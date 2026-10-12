#!/usr/bin/env python3
"""
Armilla — one complete, exact training step for a ternary layer, split across 8 harnesses (2026-10-12). Pass rules fixed before
the run.

Layer: latent int32 L (n x n), ternary W = sign(L) * (|L| > tau). Batch: N = 256 tokens, int8 inputs x, int8 targets.
  forward   y = W x            pull engine, 8 tokens per code load, 3 data rings, fractional-CRT read-out (exact)
  error     dy = int8 requantisation of (y - target): dy = (y - t) * 127 // max|y - t| (integer, deterministic)
  backward  dx = W^T dy        pull engine on W^T codes (exact)
  gradient  dW_h = sum over harness h's tokens of dy x^T: an int8 x int8 GEMM run in float32 (every partial sum < 2^24, so exact)
  all-reduce  sum_h dW_h in 4 byte rings (251, 241, 239, 233; range 2^31.6), carry-free, any order (C: ring_acc4); Garner read-out (C: garner4)
  update    L <- L - (dW >> 6)  (integer floor shift); W re-ternarised
T1 exactness: forward, backward and dW equal int64 references.                                  PASS = 0 mismatches
T2 bit-identity: two different splits of the batch over the 8 harnesses, reduced in opposite orders, give the SAME L and W
   bit for bit. The float32 version of the same step (float activations, float32 partial dW per harness, float sum in harness
   order) is run on the same two splits: number of latent entries that differ is reported.              PASS = identical
T3 speed: exact step vs float32 step (NumPy BLAS sgemm for forward, backward and dW), one core, median of 5. Reported.
Writes train_step.result.json
"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "t.so"), os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(os.path.join(tmp, "t.so")); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
DATA = np.array([251, 241, 239], np.uint16); MD = 251 * 241 * 239
kf = np.array([pow(MD // int(m), -1, int(m)) for m in DATA], np.float64) / DATA
RED = [251, 241, 239, 233]
NT, N, H, TAU, SHIFT = 8, 256, 8, 64, 6

def codes2_of(W):
    O, K = W.shape; G = K // 3; Q = G // 2
    w3 = W.reshape(O, G, 3).astype(np.int16) + 1
    cd = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
    c2 = cd.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32
    return np.ascontiguousarray(c2)

def pull_batch(W, X):
    """y[t] = W @ X[t] for all tokens, exact, 8 tokens per code load"""
    O, K = W.shape; G = K // 3; Q = G // 2; c2 = codes2_of(W)
    lut1 = np.zeros((Q, 3, 64), np.uint8); lutb = np.zeros((Q, NT, 3, 64), np.uint8)
    outb = np.zeros((NT * 3, O), np.uint8); acc = np.zeros(O * NT * 3, np.uint16); Y = np.zeros((len(X), O), np.int64)
    for t0 in range(0, len(X), NT):
        for t in range(NT):
            lib.tlut_precompute3(P(np.ascontiguousarray(X[t0 + t])), G, P(DATA), 3, P(lut1)); lutb[:, t] = lut1
        lib.tlut_gemv3_batch(P(c2), P(lutb), NT, 3, G, O, P(DATA), P(outb), P(acc), 24)
        o = outb.reshape(NT, 3, O).transpose(0, 2, 1).astype(np.float64)
        f = (o * kf).sum(axis=2) % 1.0; y = np.rint(f * MD).astype(np.int64)
        Y[t0:t0 + NT] = np.where(2 * y >= MD, y - MD, y)
    return Y

def garner(res, mods):
    dg = []
    for i in range(len(mods)):
        a = res[i].astype(np.int64)
        for j in range(i): a = (a - dg[j]) * pow(mods[j], -1, mods[i]) % mods[i]
        dg.append(a)
    y = np.zeros_like(dg[0]); w = 1
    for d, m in zip(dg, mods): y = y + d * w; w *= m
    return np.where(2 * y >= w, y - w, y)

def exact_step(L, X, Tg, split, order):
    W = (np.sign(L) * (np.abs(L) > TAU)).astype(np.int8)
    Y = pull_batch(W, X)                                                        # forward
    E = Y - Tg.astype(np.int64); mx = max(1, int(np.max(np.abs(E))))
    DY = (E * 127 // mx).astype(np.int8)                                        # error, int8
    DX = pull_batch(np.ascontiguousarray(W.T), DY)                              # backward (input gradient)
    acc = [np.zeros(L.size, np.uint8) for _ in RED]                             # ring all-reduce accumulators (bytes)
    DYf, Xf = DY.astype(np.float32), X.astype(np.float32)
    for h in order:
        idx = split[h]
        dWh = np.ascontiguousarray((DYf[idx].T @ Xf[idx]).astype(np.int32))    # exact: |partial| < 2^24
        lib.ring_acc4(P(dWh), P(acc[0]), P(acc[1]), P(acc[2]), P(acc[3]), ctypes.c_size_t(L.size))   # carry-free, any order
    dW = np.zeros(L.size, np.int64)
    lib.garner4(P(acc[0]), P(acc[1]), P(acc[2]), P(acc[3]), P(dW), ctypes.c_size_t(L.size)); dW = dW.reshape(L.shape)
    L2 = L - (dW >> SHIFT)
    return L2, Y, DX, dW, DY

def float_step(Lf, Xf, Tf, split, order):
    W = (np.sign(Lf) * (np.abs(Lf) > TAU)).astype(np.float32)
    Y = Xf @ W.T; E = Y - Tf; DY = E / max(1e-30, float(np.max(np.abs(E))))
    DX = DY @ W
    dW = np.zeros(Lf.shape, np.float32)
    for h in order:
        idx = split[h]; dW = dW + DY[idx].T @ Xf[idx]
    return Lf - dW * np.float32(127.0 / 2 ** SHIFT), DX

rng = np.random.default_rng(31415)
res = {"date": "2026-10-12", "tokens": N, "harnesses": H, "cpu": "one core, AVX-512 VBMI"}
for n in (960, 1920):
    L = rng.integers(-200, 201, (n, n)).astype(np.int64)
    X = rng.integers(-127, 128, (N, n), dtype=np.int8); Tg = rng.integers(-2000, 2001, (N, n)).astype(np.int64)
    splitA = [np.arange(h * N // H, (h + 1) * N // H) for h in range(H)]
    perm = rng.permutation(N); splitB = [np.sort(perm[h::H]) for h in range(H)]
    LA, Y, DX, dW, DY = exact_step(L, X, Tg, splitA, list(range(H)))
    LB, _, _, _, _ = exact_step(L, X, Tg, splitB, list(reversed(range(H))))
    W = (np.sign(L) * (np.abs(L) > TAU)).astype(np.int64)
    t1 = {"forward_mismatches": int(np.sum(Y != X.astype(np.int64) @ W.T)),
          "backward_mismatches": int(np.sum(DX != DY.astype(np.int64) @ W)),
          "dW_mismatches": int(np.sum(dW != DY.astype(np.int64).T @ X.astype(np.int64)))}
    Lf = L.astype(np.float32); Xf = X.astype(np.float32) / 127; Tf = Tg.astype(np.float32) / 127
    fA, _ = float_step(Lf, Xf, Tf, splitA, list(range(H))); fB, _ = float_step(Lf, Xf, Tf, splitB, list(reversed(range(H))))
    t2 = {"exact_L_bit_identical": bool(np.array_equal(LA, LB)),
          "exact_W_bit_identical": bool(np.array_equal(np.sign(LA) * (np.abs(LA) > TAU), np.sign(LB) * (np.abs(LB) > TAU))),
          "float32_latent_entries_differing": int(np.sum(fA != fB)), "of": n * n}
    te, tf = [], []
    for _ in range(5):
        s = time.perf_counter(); exact_step(L, X, Tg, splitA, list(range(H))); te.append(time.perf_counter() - s)
        s = time.perf_counter(); float_step(Lf, Xf, Tf, splitA, list(range(H))); tf.append(time.perf_counter() - s)
    res[str(n)] = {"T1": t1, "T2": t2, "T3_ms": {"exact_step": round(float(np.median(te)) * 1e3, 1), "float32_step": round(float(np.median(tf)) * 1e3, 1)},
                   "exact_vs_float32_speed": round(float(np.median(tf)) / float(np.median(te)), 2)}
    print(n, json.dumps(res[str(n)]), flush=True)
ok = all(v["T1"]["forward_mismatches"] == 0 and v["T1"]["backward_mismatches"] == 0 and v["T1"]["dW_mismatches"] == 0
         and v["T2"]["exact_L_bit_identical"] for k, v in res.items() if k.isdigit())
res["verdict"] = "PASS" if ok else "FAIL"
json.dump(res, open(os.path.join(HERE, "train_step.result.json"), "w"), indent=1); print(res["verdict"])
