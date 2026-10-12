#!/usr/bin/env python3
"""
Armilla — 5-trit codes (THEAD's P3 packing, 1.6 bits/weight) in the pull engine vs 3-trit codes (2.67 bits) vs ternary
bit-planes (2 bits), int8 inputs, 3 data rings, one core (2026-10-12). Pass rules fixed before the run.
T5a exactness: both pull engines give the exact int64 W @ x on every output.          PASS = 0 mismatches.
T5b speed: median of 30, interleaved (3-trit, 5-trit, bit-plane alternating). Reported as measured, with weight bytes.
Writes tlut5_bench.result.json
"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "t.so"), os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(os.path.join(tmp, "t.so")); c = ctypes.c_void_p
def P(a): return a.ctypes.data_as(c)
DATA = np.array([251, 241, 239], np.uint16); M = 251 * 241 * 239
kf = np.array([pow(M // int(m), -1, int(m)) for m in DATA], np.float64) / DATA
def readout(out):
    f = (out.T.astype(np.float64) * kf).sum(axis=1) % 1.0
    y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
rng = np.random.default_rng(55); res = {"date": "2026-10-12", "rings": DATA.tolist()}
for O, K in ((4096, 3840), (8192, 7680)):            # K divisible by 5, 6 and 64
    W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, K)); x = rng.integers(-127, 128, K, dtype=np.int8)
    ref = W.astype(np.int64) @ x.astype(np.int64)
    G = K // 3; Q = G // 2; G5 = K // 5
    w3 = W.reshape(O, G, 3).astype(np.int16) + 1
    cd = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
    c2 = cd.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32; c2 = np.ascontiguousarray(c2)
    w5 = W.reshape(O, G5, 5).astype(np.int16) + 1
    c5 = np.ascontiguousarray(sum(w5[:, :, j] * 3 ** j for j in range(5)).T.astype(np.uint8))
    words = K // 64
    bits = lambda B: np.packbits(B.reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
    plus, minus = bits(W == 1), bits(W == -1)
    lut3 = np.zeros((Q, 3, 64), np.uint8); out3 = np.zeros((3, O), np.uint8); acc3 = np.zeros(O * 3, np.uint16)
    lut5 = np.zeros((G5, 3, 256), np.uint8); out5 = np.zeros((3, O), np.uint8); acc5 = np.zeros(O * 3, np.uint16)
    ybp = np.zeros(O, np.int32)
    def e3():
        lib.tlut_precompute3(P(x), G, P(DATA), 3, P(lut3)); lib.tlut_gemv3(P(c2), P(lut3), 3, G, O, P(DATA), P(out3), P(acc3), 96); return readout(out3)
    def e5():
        lib.tlut5_precompute(P(x), G5, P(DATA), 3, P(lut5)); lib.tlut5_gemv(P(c5), P(lut5), 3, G5, O, P(DATA), P(out5), P(acc5), 24); return readout(out5)
    def bp(): ybp[:] = 0; lib.gemv_bitplane(P(plus), P(minus), P(x), O, K, P(ybp))
    m3 = int(np.sum(e3() != ref)); m5 = int(np.sum(e5() != ref)); bp(); mb = int(np.sum(ybp != ref))
    T = {"3trit": [], "5trit": [], "bitplane": []}
    for _ in range(30):
        for k, f in (("3trit", e3), ("5trit", e5), ("bitplane", bp)):
            t0 = time.perf_counter(); f(); T[k].append(time.perf_counter() - t0)
    t = {k: float(np.median(v)) for k, v in T.items()}
    t5p = []
    for _ in range(30):
        t0 = time.perf_counter(); lib.tlut5_precompute(P(x), G5, P(DATA), 3, P(lut5)); t5p.append(time.perf_counter() - t0)
    res[f"{O}x{K}"] = {"mismatches": {"3trit": m3, "5trit": m5, "bitplane": mb},
        "us": {k: round(v * 1e6, 1) for k, v in t.items()}, "5trit_precompute_us": round(float(np.median(t5p)) * 1e6, 1),
        "weight_bytes": {"3trit": int(c2.nbytes), "5trit": int(c5.nbytes), "bitplane": int(plus.nbytes + minus.nbytes)},
        "5trit_vs_3trit_speed": round(t["3trit"] / t["5trit"], 2), "5trit_vs_bitplane_speed": round(t["bitplane"] / t["5trit"], 2)}
    print(O, K, json.dumps(res[f"{O}x{K}"]), flush=True)
res["verdict"] = "PASS" if all(v["mismatches"]["3trit"] == 0 and v["mismatches"]["5trit"] == 0 for k, v in res.items() if k[0].isdigit()) else "FAIL"
json.dump(res, open(os.path.join(HERE, "tlut5_bench.result.json"), "w"), indent=1); print(res["verdict"])
