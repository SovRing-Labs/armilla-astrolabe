#!/usr/bin/env python3
"""
Armilla ternary pull engine — exactness and speed (2026-10-12). Pass rules fixed before the run.

Pipeline per layer (one input vector, the decode case):
  1. outer rings precompute: for every group of 3 inputs and every ring, the 27 possible ternary answers (tlut_precompute);
  2. the centre pulls: one vpermb per (64 rows, group, ring) takes the answers from the table (tlut_gemv);
  3. read-out: fractional CRT over the 3 data rings gives y / M in one dot product; y, its sign, ReLU and int8 requantisation
     follow without rebuilding big integers; the 4th ring (233) is a check ring: every output must agree with it.
T1  exactness: every output equals int64 W @ x, check ring agrees on every output.        PASS = 0 mismatches.
T2  speed, one core, same inputs: pull engine (precompute + pull + read-out) vs int8 GEMV (compiler-vectorised AVX-512),
    vs ternary bit-plane GEMV (AVX-512 masked adds), vs NumPy float32 GEMV (BLAS). Median of 30 runs. Reported as measured.
v2 (tlut_gemv2): all rings per code load, two groups per 64-entry table, vpmaddubsw pair sums, vector Barrett.
v3 (tlut_gemv3): v2 with L1-resident table tiles (96 group pairs per tile).
Build: gcc -O3 -march=native -shared -fPIC -o libtlut.so armilla_tlut.c
"""
import ctypes, json, os, subprocess, sys, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

HERE = os.path.dirname(os.path.abspath(__file__))
SO = os.path.join(HERE, "libtlut.so")
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", SO, os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(SO)
c = ctypes.c_void_p
def P(a): return a.ctypes.data_as(c)

DATA = [251, 241, 239]; CHECK = 233
MODS = np.array(DATA + [CHECK], dtype=np.uint16); R = len(MODS)
M = 251 * 241 * 239
k = np.array([pow(M // m, -1, m) for m in DATA], dtype=np.float64)

def median_time(f, n=30):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

rng = np.random.default_rng(12)
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 VBMI, one core", "rings": {"data": DATA, "check": CHECK},
       "range_bits": int(M).bit_length()}
for O, K in ((4096, 4032), (8192, 8064)):
    G = K // 3
    W = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(O, K), p=[1 / 3, 1 / 3, 1 / 3])
    x = rng.integers(-127, 128, K, dtype=np.int8)
    ref = W.astype(np.int64) @ x.astype(np.int64)
    # weight forms
    w3 = (W.reshape(O, G, 3).astype(np.int16) + 1)
    codes = np.ascontiguousarray((w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8))   # G x O
    words = K // 64
    plus = np.packbits((W == 1).reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
    minus = np.packbits((W == -1).reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
    Wf = W.astype(np.float32); xf = x.astype(np.float32)
    lut = np.zeros((R, G, 64), dtype=np.uint8); out = np.zeros((R, O), dtype=np.uint8)
    c2 = codes.reshape(G // 2, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32
    codes2 = np.ascontiguousarray(c2); lut2 = np.zeros((G // 2, R, 64), dtype=np.uint8); out2 = np.zeros((R, O), dtype=np.uint8)
    y_i8 = np.zeros(O, dtype=np.int32); y_bp = np.zeros(O, dtype=np.int32)

    def precompute(): lib.tlut_precompute(P(x), G, P(MODS), R, P(lut))
    def pull(): lib.tlut_gemv(P(codes), P(lut), R, G, O, P(MODS), P(out))
    def readout():
        f = (out[:3].T.astype(np.float64) * k / np.array(DATA)).sum(axis=1) % 1.0      # y / M, one dot product
        y = np.rint(f * M).astype(np.int64); y = np.where(2 * y >= M, y - M, y)          # centred value, sign included
        return y
    def engine():
        precompute(); pull(); return readout()
    def precompute2(): lib.tlut_precompute2(P(x), G, P(MODS), R, P(lut2))
    def pull2(): lib.tlut_gemv2(P(codes2), P(lut2), R, G, O, P(MODS), P(out2))
    def readout2():
        f = (out2[:3].T.astype(np.float64) * k / np.array(DATA)).sum(axis=1) % 1.0
        y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
    def engine2():
        precompute2(); pull2(); return readout2()
    acc16 = np.zeros(O * R, dtype=np.uint16); out3 = np.zeros((R, O), dtype=np.uint8)
    def pull3(): lib.tlut_gemv3(P(codes2), P(lut2), R, G, O, P(MODS), P(out3), P(acc16), 96)
    def engine3():
        precompute2(); pull3()
        f = (out3[:3].T.astype(np.float64) * k / np.array(DATA)).sum(axis=1) % 1.0
        y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
    y3 = engine3(); exact3 = int(np.sum(y3 != ref)); check3 = bool(np.all((y3 % CHECK) == out3[3]))
    y2 = engine2(); exact2 = int(np.sum(y2 != ref)); check2 = bool(np.all((y2 % CHECK) == out2[3]))
    y = engine()
    check_ok = bool(np.all((y % CHECK) == out[3]))
    exact = int(np.sum(y != ref))
    relu = np.maximum(y, 0); x_next = np.clip(np.rint(relu * (127.0 / max(1, relu.max()))), 0, 127).astype(np.int8)
    def i8(): lib.gemv_i8(P(W), P(x), O, K, P(y_i8))
    def bp():
        y_bp[:] = 0; lib.gemv_bitplane(P(plus), P(minus), P(x), O, K, P(y_bp))
    i8(); bp()
    t = {"engine3_total": median_time(engine3), "pull3": median_time(pull3), "engine2_total": median_time(engine2), "precompute2": median_time(precompute2), "pull2": median_time(pull2),
         "engine_total": median_time(engine), "precompute": median_time(precompute), "pull": median_time(pull),
         "readout": median_time(readout), "int8_gemv": median_time(i8), "bitplane_gemv": median_time(bp),
         "numpy_float32_gemv": median_time(lambda: Wf @ xf)}
    res[f"{O}x{K}"] = {
        "T1_mismatches_vs_int64": exact + exact2 + exact3, "T1_check_ring_agrees_all": check_ok and check2 and check3,
        "engine3_vs_int8": round(t["int8_gemv"] / t["engine3_total"], 2),
        "engine3_vs_bitplane": round(t["bitplane_gemv"] / t["engine3_total"], 2),
        "pull3_vs_bitplane": round(t["bitplane_gemv"] / t["pull3"], 2),
        "engine3_vs_numpy_float32": round(t["numpy_float32_gemv"] / t["engine3_total"], 2),
        "engine2_vs_int8": round(t["int8_gemv"] / t["engine2_total"], 2),
        "engine2_vs_bitplane": round(t["bitplane_gemv"] / t["engine2_total"], 2),
        "engine2_vs_numpy_float32": round(t["numpy_float32_gemv"] / t["engine2_total"], 2),
        "int8_baseline_exact": bool(np.array_equal(y_i8, ref)), "bitplane_baseline_exact": bool(np.array_equal(y_bp, ref)),
        "weight_bytes": {"pull_engine_codes": int(codes.nbytes), "int8": int(W.nbytes), "bitplanes": int(plus.nbytes + minus.nbytes),
                         "float32": int(Wf.nbytes)},
        "us": {kk: round(v * 1e6, 1) for kk, v in t.items()},
        "engine_vs_int8": round(t["int8_gemv"] / t["engine_total"], 2),
        "engine_vs_bitplane": round(t["bitplane_gemv"] / t["engine_total"], 2),
        "engine_vs_numpy_float32": round(t["numpy_float32_gemv"] / t["engine_total"], 2),
        "relu_requant_next_layer_nonzero": int(np.count_nonzero(x_next)),
    }
    print(O, K, json.dumps(res[f"{O}x{K}"]), flush=True)
res["verdict"] = "PASS" if all(v["T1_mismatches_vs_int64"] == 0 and v["T1_check_ring_agrees_all"] for kk, v in res.items() if "x" in kk and isinstance(v, dict)) else "FAIL"
json.dump(res, open(os.path.join(HERE, "tlut_bench.result.json"), "w"), indent=1)
print(res["verdict"])
