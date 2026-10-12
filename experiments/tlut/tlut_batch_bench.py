#!/usr/bin/env python3
"""Forward pass with 8 tokens at once (prefill / batched decode): pull engine (one code load feeds 8 tokens x 3 rings) vs batched
bit-planes (masks loaded once per word) vs NumPy float32 GEMM (BLAS), one core. Pass rules fixed before the run:
FB1 exactness of both integer kernels vs int64 W @ X.   FB2 speed, interleaved medians of 30.  Writes tlut_batch_bench.result.json"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "t.so"), os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(os.path.join(tmp, "t.so")); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
DATA = np.array([251, 241, 239], np.uint16); M = 251 * 241 * 239; R = 3; NT = 8
kf = np.array([pow(M // int(m), -1, int(m)) for m in DATA], np.float64) / DATA
rng = np.random.default_rng(88); res = {"date": "2026-10-12", "tokens": NT}
for O, K in ((4096, 4032), (8192, 8064)):
    G = K // 3; Q = G // 2
    W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, K)); X = rng.integers(-127, 128, (NT, K), dtype=np.int8)
    ref = W.astype(np.int64) @ X.T.astype(np.int64)                          # O x NT
    w3 = W.reshape(O, G, 3).astype(np.int16) + 1
    cd = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
    c2 = cd.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32; c2 = np.ascontiguousarray(c2)
    words = K // 64
    bits = lambda B: np.packbits(B.reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
    plus, minus = bits(W == 1), bits(W == -1)
    lut1 = np.zeros((Q, R, 64), np.uint8); lutb = np.zeros((Q, NT, R, 64), np.uint8); outb = np.zeros((NT * R, O), np.uint8)
    acc = np.zeros(O * NT * R, np.uint16); Y = np.zeros((NT, O), np.int32); Wf = W.astype(np.float32); Xf = X.T.astype(np.float32).copy()
    def pre():
        for t in range(NT):
            lib.tlut_precompute3(P(np.ascontiguousarray(X[t])), G, P(DATA), R, P(lut1)); lutb[:, t] = lut1
    def engine():
        pre(); lib.tlut_gemv3_batch(P(c2), P(lutb), NT, R, G, O, P(DATA), P(outb), P(acc), 24)
        o = outb.reshape(NT, R, O)
        f = (o.transpose(0, 2, 1).astype(np.float64) * kf).sum(axis=2) % 1.0
        y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
    def bp(): Y[:] = 0; lib.gemv_bitplane_batch(P(plus), P(minus), P(X), NT, O, K, P(Y))
    e_ok = bool((engine().T == ref).all()); bp(); b_ok = bool((Y.T == ref).all())
    te, tb, tf = [], [], []
    for _ in range(30):
        s = time.perf_counter(); engine(); te.append(time.perf_counter() - s)
        s = time.perf_counter(); bp(); tb.append(time.perf_counter() - s)
        s = time.perf_counter(); Wf @ Xf; tf.append(time.perf_counter() - s)
    m = lambda v: float(np.median(v))
    res[f"{O}x{K}"] = {"engine_exact": e_ok, "bitplane_batch_exact": b_ok,
        "us_per_token": {"engine": round(m(te) / NT * 1e6, 1), "bitplane_batch": round(m(tb) / NT * 1e6, 1), "float32_gemm": round(m(tf) / NT * 1e6, 1)},
        "engine_vs_bitplane": round(m(tb) / m(te), 2), "engine_vs_float32": round(m(tf) / m(te), 2)}
    print(O, K, res[f"{O}x{K}"], flush=True)
json.dump(res, open(os.path.join(HERE, "tlut_batch_bench.result.json"), "w"), indent=1)
