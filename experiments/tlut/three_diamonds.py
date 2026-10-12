#!/usr/bin/env python3
"""
Armilla — the three-diamond unit (J 2026-10-12): "take our diamond as 1 unit, and turn it into a three diamond unit setup, where all
the answers are then pushed from the core of each diamond that lives in cpu cache". Pass rules fixed before the run.

Mapping on this CPU (4 vCPU): one diamond = one CPU core. Its core (the 5-trit answer-table tile, 24 groups x rings x 256 bytes,
~18 KB, plus its 16-bit accumulators) lives in that core's L1; the outer rings' precompute fills the tables once per input and every
diamond pulls from them; each diamond pushes its finished residues straight into the shared output. Codes are stored row-block-major ([O/64][G5][64], one contiguous stream per block). Two ways to share the work:
  rows   each diamond owns a third of the weight rows (its own codes) and pulls all 3 data rings for them
  rings  each diamond owns one data ring (251, 241 or 239) and pulls it for every row
D1 exactness: three-diamond outputs equal int64 W @ x, both splits.                                   PASS = 0 mismatches
D2 speed (one token, interleaved medians of 30): 1 diamond vs 3 diamonds (rows) vs 3 diamonds (rings), against a bit-plane GEMV and
   NumPy float32 GEMV split the same way over 3 cores (row thirds, one thread each). Reported.
Writes three_diamonds.result.json
"""
import ctypes, json, os, subprocess, tempfile, threading, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "t.so"), os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(os.path.join(tmp, "t.so")); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
DATA = np.array([251, 241, 239], np.uint16); M = 251 * 241 * 239
kf = np.array([pow(M // int(m), -1, int(m)) for m in DATA], np.float64) / DATA
def readout(out3):
    f = (out3.T.astype(np.float64) * kf).sum(axis=1) % 1.0
    y = np.rint(f * M).astype(np.int64); return np.where(2 * y >= M, y - M, y)
def par(fns):
    ts = [threading.Thread(target=f) for f in fns]; [t.start() for t in ts]; [t.join() for t in ts]
rng = np.random.default_rng(333); res = {"date": "2026-10-12", "cpu": "4 vCPU Xeon @ 2.10 GHz, AVX-512 VBMI", "diamonds": 3}
for LAYOUT in ("group_major", "row_block_major"):
 for O, K in ((6144, 7680), (12288, 7680)):                 # O divisible by 3 * 64, K by 5 and 64
     G5 = K // 5; W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, K)); x = rng.integers(-127, 128, K, dtype=np.int8)
     ref = W.astype(np.int64) @ x.astype(np.int64)
     w5 = W.reshape(O, G5, 5).astype(np.int16) + 1
     codes = np.ascontiguousarray(sum(w5[:, :, j] * 3 ** j for j in range(5)).T.astype(np.uint8))     # [G5][O]
     Ot = O // 3
     blk = lambda c: np.ascontiguousarray(c.reshape(c.shape[0], c.shape[1] // 64, 64).transpose(1, 0, 2))   # row-block-major
     if LAYOUT == "row_block_major":
         GEMV = lib.tlut5_gemv_blk; codesb = blk(codes); slices = [blk(np.ascontiguousarray(codes[:, d * Ot:(d + 1) * Ot])) for d in range(3)]
     else:
         GEMV = lib.tlut5_gemv; codesb = codes; slices = [np.ascontiguousarray(codes[:, d * Ot:(d + 1) * Ot]) for d in range(3)]
     lut3 = np.zeros((G5, 3, 256), np.uint8)                                                           # tables, 3 rings
     lut1 = [np.zeros((G5, 1, 256), np.uint8) for _ in range(3)]                                       # one ring per diamond
     out1 = np.zeros((3, O), np.uint8); outs = [np.zeros((3, Ot), np.uint8) for _ in range(3)]; outr = [np.zeros((1, O), np.uint8) for _ in range(3)]
     acc1 = np.zeros(O * 3, np.uint16); accs = [np.zeros(Ot * 3, np.uint16) for _ in range(3)]; accr = [np.zeros(O, np.uint16) for _ in range(3)]
     def one():
         lib.tlut5_precompute(P(x), G5, P(DATA), 3, P(lut3)); GEMV(P(codesb), P(lut3), 3, G5, O, P(DATA), P(out1), P(acc1), 24)
         return readout(out1)
     def rows():
         lib.tlut5_precompute(P(x), G5, P(DATA), 3, P(lut3))
         par([lambda d=d: GEMV(P(slices[d]), P(lut3), 3, G5, Ot, P(DATA), P(outs[d]), P(accs[d]), 24) for d in range(3)])
         return readout(np.concatenate(outs, axis=1))
     def rings():
         par([lambda d=d: (lib.tlut5_precompute(P(x), G5, P(DATA[d:d + 1]), 1, P(lut1[d])),
                           GEMV(P(codesb), P(lut1[d]), 1, G5, O, P(DATA[d:d + 1]), P(outr[d]), P(accr[d]), 24)) for d in range(3)])
         return readout(np.concatenate(outr, axis=0))
     words = K // 64
     bits = lambda B: np.packbits(B.reshape(O, words, 64)[:, :, ::-1], axis=2, bitorder="big").view(">u8").reshape(O, words).astype(np.uint64)
     plus, minus = bits(W == 1), bits(W == -1); ybp = np.zeros(O, np.int32)
     psl = [(np.ascontiguousarray(plus[d * Ot:(d + 1) * Ot]), np.ascontiguousarray(minus[d * Ot:(d + 1) * Ot])) for d in range(3)]
     ysl = [np.zeros(Ot, np.int32) for _ in range(3)]
     def bp3():
         for y in ysl: y[:] = 0
         par([lambda d=d: lib.gemv_bitplane(P(psl[d][0]), P(psl[d][1]), P(x), Ot, K, P(ysl[d])) for d in range(3)])
     Wf = [np.ascontiguousarray(W[d * Ot:(d + 1) * Ot].astype(np.float32)) for d in range(3)]; xf = x.astype(np.float32)
     def f3(): par([lambda d=d: Wf[d] @ xf for d in range(3)])
     e = {"one_diamond": int(np.sum(one() != ref)), "three_rows": int(np.sum(rows() != ref)), "three_rings": int(np.sum(rings() != ref))}
     bp3(); e["bitplane_3core"] = int(np.sum(np.concatenate(ysl) != ref))
     T = {k: [] for k in ("one_diamond", "three_rows", "three_rings", "bitplane_3core", "float32_3core")}
     for _ in range(30):
         for k, f in (("one_diamond", one), ("three_rows", rows), ("three_rings", rings), ("bitplane_3core", bp3), ("float32_3core", f3)):
             s = time.perf_counter(); f(); T[k].append(time.perf_counter() - s)
     md = {k: round(float(np.median(v)) * 1e6, 1) for k, v in T.items()}
     res[f"{LAYOUT}:{O}x{K}"] = {"D1_mismatches": e, "D2_us": md,
         "three_rows_vs_one": round(md["one_diamond"] / md["three_rows"], 2), "three_rings_vs_one": round(md["one_diamond"] / md["three_rings"], 2),
         "three_rows_vs_bitplane_3core": round(md["bitplane_3core"] / md["three_rows"], 2), "three_rows_vs_float32_3core": round(md["float32_3core"] / md["three_rows"], 2),
         "weight_MB": {"5trit_codes": round(codes.nbytes / 1e6, 1), "bitplanes": round((plus.nbytes + minus.nbytes) / 1e6, 1), "float32": round(W.size * 4 / 1e6, 1)}}
     print(LAYOUT, O, K, json.dumps(res[f"{LAYOUT}:{O}x{K}"]), flush=True)
res["verdict"] = "PASS" if all(all(v == 0 for v in r["D1_mismatches"].values()) for k, r in res.items() if ":" in k) else "FAIL"
json.dump(res, open(os.path.join(HERE, "three_diamonds.result.json"), "w"), indent=1); print(res["verdict"])
