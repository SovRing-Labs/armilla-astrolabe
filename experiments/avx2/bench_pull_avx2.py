#!/usr/bin/env python3
"""Legacy-CPU path (2026-10-12): AVX2-only pull (2-trit codes, pshufb; v2 row-block-major codes, L1 table tiles of 120 pairs) vs omnid's AVX2 sign_epi8 GEMV vs NumPy float32, one core.
Built with -mavx2 -mno-avx512f (no zmm in the code). Pass rules fixed before the run:
A1 exactness of both vs int64 W @ x (0 mismatches).  A2 speed, interleaved medians of 30. Writes bench_pull_avx2.result.json"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp(); so = os.path.join(tmp, "p.so")
subprocess.run(["gcc", "-O3", "-mavx2", "-mno-avx512f", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "pull_avx2.c")], check=True)
asm = subprocess.run(["gcc", "-O3", "-mavx2", "-mno-avx512f", "-S", "-o", "-", os.path.join(HERE, "pull_avx2.c")], capture_output=True, text=True).stdout
lib = ctypes.CDLL(so); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
M = 251 * 241 * 239; mods = np.array([251, 241, 239]); kf = np.array([pow(M // int(m), -1, int(m)) for m in mods], np.float64) / mods
rng = np.random.default_rng(2013); res = {"date": "2026-10-12", "build": "gcc -O3 -mavx2 -mno-avx512f", "zmm_in_asm": "zmm" in asm}
for O, K in ((4096, 4096), (8192, 8192)):
    W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, K)); x = rng.integers(-127, 128, K, dtype=np.int8)
    ref = W.astype(np.int64) @ x.astype(np.int64)
    w2 = W.reshape(O, K // 2, 2).astype(np.int16) + 1
    codes = np.ascontiguousarray((w2[:, :, 0] + 3 * w2[:, :, 1]).T.astype(np.uint8))
    codesb = np.ascontiguousarray(codes.reshape(K // 2, O // 32, 32).transpose(1, 0, 2)); acc16 = np.zeros(O * 3, np.uint16)
    lut = np.zeros((K // 2, 3, 32), np.uint8); out = np.zeros((3, O), np.uint8); y = np.zeros(O, np.int32)
    Wf = W.astype(np.float32); xf = x.astype(np.float32)
    def pull():
        lib.pull2_precompute(P(x), K, P(lut)); lib.pull2_gemv_blk(P(codesb), P(lut), K // 2, O, P(out), P(acc16), 120)
        f = (out.T.astype(np.float64) * kf).sum(axis=1) % 1.0
        v = np.rint(f * M).astype(np.int64); return np.where(2 * v >= M, v - M, v)
    def sign(): lib.gemv_sign(P(W), P(x), O, K, P(y))
    e1 = int(np.sum(pull() != ref)); sign(); e2 = int(np.sum(y != ref))
    tp, ts, tf = [], [], []
    for _ in range(30):
        s = time.perf_counter(); pull(); tp.append(time.perf_counter() - s)
        s = time.perf_counter(); sign(); ts.append(time.perf_counter() - s)
        s = time.perf_counter(); Wf @ xf; tf.append(time.perf_counter() - s)
    m = lambda v: float(np.median(v))
    res[f"{O}x{K}"] = {"pull_mismatches": e1, "omnid_sign_mismatches": e2,
        "us": {"pull_avx2": round(m(tp) * 1e6, 1), "omnid_sign_epi8": round(m(ts) * 1e6, 1), "numpy_float32": round(m(tf) * 1e6, 1)},
        "weight_bytes": {"pull_codes": int(codes.nbytes), "omnid_int8": int(W.nbytes)},
        "pull_vs_omnid": round(m(ts) / m(tp), 2), "pull_vs_float32": round(m(tf) / m(tp), 2)}
    print(O, K, res[f"{O}x{K}"], flush=True)
json.dump(res, open(os.path.join(HERE, "bench_pull_avx2.result.json"), "w"), indent=1)
