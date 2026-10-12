#!/usr/bin/env python3
"""
Armilla ternary pull engine, cyclone form (2026-10-12). Pass rules fixed before the run.

  outer rings (precompute, vectorised)  ->  centre (3 data rings, L1-tiled vpermb pull)  ->  read-out (fractional CRT, ReLU)
  guard system (check ring 233) pulls on a SECOND core at the same time; every output must agree with it.

C1  exactness: all outputs equal int64 W @ x; guard agrees on every output.                     PASS = 0 mismatches.
C2  one layer (also measured interleaved, engine and bit-plane alternating, 40 pairs), data path on one core vs bit-plane ternary / int8 / NumPy float32 GEMV on one core (median of 30).
C3  compounding: three matrices that read the same input (Q, K, V) share ONE precompute.
C4  guard overlap: wall time of data path + guard on two cores vs data path alone.
Writes tlut_cyclone.result.json
"""
import ctypes, json, os, subprocess, threading, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "libtlut.so")
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", SO, os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(SO); c = ctypes.c_void_p
def P(a): return a.ctypes.data_as(c)
DATA = np.array([251, 241, 239], dtype=np.uint16); CHECK = np.array([233], dtype=np.uint16)
M = 251 * 241 * 239; kf = np.array([pow(M // int(m), -1, int(m)) for m in DATA], dtype=np.float64) / DATA
def med(f, n=30):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))
rng = np.random.default_rng(1012)
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 VBMI", "data_rings": DATA.tolist(), "check_ring": 233}
for O, K in ((4096, 4032), (8192, 8064)):
    G = K // 3; Q = G // 2
    def forms(W):
        w3 = W.reshape(O, G, 3).astype(np.int16) + 1
        codes = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
        c2 = codes.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32
        words = K // 64
        bits = lambda B: np.packbits(B.reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
        return np.ascontiguousarray(c2), bits(W == 1), bits(W == -1)
    Ws = [rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(O, K)) for _ in range(3)]
    x = rng.integers(-127, 128, K, dtype=np.int8)
    F = [forms(W) for W in Ws]
    refs = [W.astype(np.int64) @ x.astype(np.int64) for W in Ws]
    lutD = np.zeros((Q, 3, 64), np.uint8); lutC = np.zeros((Q, 1, 64), np.uint8)
    outD = [np.zeros((3, O), np.uint8) for _ in range(3)]; outC = [np.zeros((1, O), np.uint8) for _ in range(3)]
    accD = np.zeros(O * 3, np.uint16); accC = np.zeros(O, np.uint16)
    def pre(): lib.tlut_precompute3(P(x), G, P(DATA), 3, P(lutD))
    def pull(i): lib.tlut_gemv3(P(F[i][0]), P(lutD), 3, G, O, P(DATA), P(outD[i]), P(accD), 96)
    def guard(i):
        lib.tlut_precompute3(P(x), G, P(CHECK), 1, P(lutC)); lib.tlut_gemv3(P(F[i][0]), P(lutC), 1, G, O, P(CHECK), P(outC[i]), P(accC), 96)
    def readout(i):
        f = (outD[i].T.astype(np.float64) * kf).sum(axis=1) % 1.0
        y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
    def layer(i=0): pre(); pull(i); return readout(i)
    def layer_guarded(i=0):
        t = threading.Thread(target=guard, args=(i,)); t.start(); y = layer(i); t.join(); return y
    def qkv(): pre(); return [(pull(i), readout(i))[1] for i in range(3)]
    ybp = np.zeros(O, np.int32); Wi8 = Ws[0]; yi8 = np.zeros(O, np.int32); Wf = Ws[0].astype(np.float32); xf = x.astype(np.float32)
    def bp(i=0): ybp[:] = 0; lib.gemv_bitplane(P(F[i][1]), P(F[i][2]), P(x), O, K, P(ybp))
    def i8(): lib.gemv_i8(P(Wi8), P(x), O, K, P(yi8))
    # C1
    bad = 0; gbad = 0
    for i in range(3):
        y = (pre(), pull(i), readout(i))[2]; guard(i)
        bad += int(np.sum(y != refs[i])); gbad += int(np.sum((y % 233) != outC[i][0]))
    bp(); bp_exact = bool(np.array_equal(ybp, refs[0])); i8(); i8_exact = bool(np.array_equal(yi8, refs[0]))
    il_e, il_b = [], []
    for _ in range(40):
        t0 = time.perf_counter(); layer(); il_e.append(time.perf_counter() - t0)
        t0 = time.perf_counter(); bp(); il_b.append(time.perf_counter() - t0)
    t = {"interleaved_layer": float(np.median(il_e)), "interleaved_bitplane": float(np.median(il_b)), "layer_1core": med(layer), "precompute": med(pre), "pull": med(lambda: pull(0)), "readout": med(lambda: readout(0)),
         "layer_with_guard_2cores": med(layer_guarded), "guard_alone": med(lambda: guard(0)),
         "qkv_shared_precompute": med(qkv), "bitplane": med(bp), "bitplane_x3": med(lambda: [bp(i) for i in range(3)]),
         "int8": med(i8), "numpy_float32": med(lambda: Wf @ xf)}
    res[f"{O}x{K}"] = {"C1_mismatches": bad, "C1_guard_disagreements": gbad, "bitplane_exact": bp_exact, "int8_exact": i8_exact,
        "us": {k: round(v * 1e6, 1) for k, v in t.items()},
        "C2_layer_vs_bitplane": round(t["bitplane"] / t["layer_1core"], 2),
        "C2_interleaved_layer_vs_bitplane": round(t["interleaved_bitplane"] / t["interleaved_layer"], 2), "C2_layer_vs_int8": round(t["int8"] / t["layer_1core"], 2),
        "C2_layer_vs_float32": round(t["numpy_float32"] / t["layer_1core"], 2),
        "C3_qkv_vs_bitplane_x3": round(t["bitplane_x3"] / t["qkv_shared_precompute"], 2),
        "C4_guard_overhead_wall": round(t["layer_with_guard_2cores"] / t["layer_1core"], 2),
        "weight_bytes": {"codes": int(F[0][0].nbytes), "bitplanes": int(F[0][1].nbytes + F[0][2].nbytes), "int8": int(Ws[0].nbytes)}}
    print(O, K, json.dumps(res[f"{O}x{K}"]), flush=True)
res["verdict"] = "PASS" if all(v["C1_mismatches"] == 0 and v["C1_guard_disagreements"] == 0 for k, v in res.items() if k[0].isdigit()) else "FAIL"
json.dump(res, open(os.path.join(HERE, "tlut_cyclone.result.json"), "w"), indent=1)
print(res["verdict"])
