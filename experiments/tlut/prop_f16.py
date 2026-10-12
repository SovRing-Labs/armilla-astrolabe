#!/usr/bin/env python3
"""
Armilla — forward AND backward propagation through a ternary layer in rings, f16 activations and gradients (2026-10-12).
Pass rules fixed before the run.

Layer: y = W x (forward), dx = W^T dy (backward, the input gradient), W ternary (n x n), x and dy f16.
Both directions use the same machine: outer rings precompute 27-answer tables per (Gelosia digit, ring), the centre pulls them with
vpermb from one code load (rendered kernel), the diagonals are summed in-ring, one Garner read-out carries.
The backward pass reads W^T codes (groups of 3 along the output side), the form THEAD's float table backward used
(RAM-KIT-0 #1, registered "NOT bit-identical", shipped OFF).

Two exactness modes:
  full    every f16 value as X = v * 2^24 (exact), 6 digits, 7 rings (42 table sets)
  window  per-vector scale: X = round(v * 2^s), s chosen so max |X| < 2^23; 3 digits, 5 rings (15 table sets). Exact for every
          value on the window's grid; values finer than the grid are rounded once on entry and COUNTED.
P1  full mode: forward and backward outputs equal the exact integer products; bit-identical over 3 reorderings. PASS = 0 / identical
P2  window mode: outputs equal the exact integer product of the windowed integers (0 mismatches); entry rounding counted;
    relative error vs the exact f16 product reported next to float32's.                                    PASS = 0 mismatches
P3  speed, one core, median of 15: each mode vs NumPy float32 (forward W@x, backward W.T@dy). Reported as measured.
Writes prop_f16.result.json
"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "b.so"), os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(os.path.join(tmp, "b.so")); c = ctypes.c_void_p
def P(a): return a.ctypes.data_as(c)
ALL = [251, 241, 239, 233, 229, 227, 223]

def render(rings, D, G, O, QT=48, CH=8):
    R = len(rings); RT = R * D; Q = G // 2
    mods = [rings[t % R] for t in range(RT)]
    chunks = [list(range(s, min(s + CH, RT))) for s in range(0, RT, CH)]
    L = ["#include <immintrin.h>", "#include <stdint.h>",
         "static inline __m512i br(__m512i a,int m){__m512i q=_mm512_mulhi_epu16(a,_mm512_set1_epi16((short)(65536/m)));"
         "__m512i r=_mm512_sub_epi16(a,_mm512_mullo_epi16(q,_mm512_set1_epi16((short)m)));__m512i mm=_mm512_set1_epi16((short)m);"
         "r=_mm512_min_epu16(r,_mm512_sub_epi16(r,mm));return _mm512_min_epu16(r,_mm512_sub_epi16(r,mm));}",
         f"/* rendered: rings={rings} D={D} G={G} O={O} tile={QT} chunk={CH} */",
         "void pull(const uint8_t *codes2,const uint8_t *lut,uint8_t *out,uint16_t *acc16){const __m512i ones=_mm512_set1_epi8(1);",
         f"for(int i=0;i<{O // 32 * RT * 32};i++)acc16[i]=0;"]
    for ch in chunks:
        L.append(f"for(int q0=0;q0<{Q};q0+={QT}){{int q1=q0+{QT}<{Q}?q0+{QT}:{Q};for(int bl=0;bl<{O // 32};bl++){{")
        L += [f"__m512i a{i}=_mm512_loadu_si512(acc16+((size_t)bl*{RT}+{t})*32);" for i, t in enumerate(ch)]
        L.append(f"for(int q=q0;q<q1;q++){{__m512i idx=_mm512_loadu_si512(codes2+((size_t)q*{O}+bl*32)*2);const uint8_t*T=lut+(size_t)q*{RT * 64};")
        L += [f"a{i}=_mm512_add_epi16(a{i},_mm512_maddubs_epi16(_mm512_permutexvar_epi8(idx,_mm512_loadu_si512(T+{t * 64})),ones));" for i, t in enumerate(ch)]
        L.append("}")
        L += [f"_mm512_storeu_si512(acc16+((size_t)bl*{RT}+{t})*32,br(a{i},{mods[t]}));" for i, t in enumerate(ch)]
        L.append("}}")
    L.append(f"for(int bl=0;bl<{O // 32};bl++)for(int t=0;t<{RT};t++)for(int l=0;l<32;l++)out[(size_t)t*{O}+bl*32+l]=(uint8_t)acc16[((size_t)bl*{RT}+t)*32+l];}}")
    src = "\n".join(L)
    f = os.path.join(tmp, f"r{abs(hash((tuple(rings), D, G, O)))}.c"); open(f, "w").write(src)
    so = f[:-2] + ".so"
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, f], check=True)
    return ctypes.CDLL(so)

def codes_of(W):            # rows of W pull; groups of 3 along its columns
    O, K = W.shape; G = K // 3; Q = G // 2
    w3 = W.reshape(O, G, 3).astype(np.int16) + 1
    cd = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
    c2 = cd.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32
    return np.ascontiguousarray(c2)

def digits(X, D):
    ds = []; v = X.copy()
    for _ in range(D):
        d = ((v + 128) % 256) - 128; ds.append(d.astype(np.int8)); v = (v - d) // 256
    assert np.all(v == 0)
    return np.ascontiguousarray(np.stack(ds))

def garner(res, mods):
    dg = []
    for i in range(len(mods)):
        a = res[i].astype(np.int64)
        for j in range(i): a = (a - dg[j]) * pow(mods[j], -1, mods[i]) % mods[i]
        dg.append(a)
    y = np.zeros_like(dg[0]); w = 1
    for d, m in zip(dg, mods): y = y + d * w; w *= m
    return np.where(2 * y >= w, y - w, y)

class Engine:
    def __init__(self, W, rings, D):
        self.O, self.K = W.shape; self.G = self.K // 3; self.rings = rings; self.R = len(rings); self.D = D
        self.codes = codes_of(W); self.jit = render(rings, D, self.G, self.O)
        self.mods = np.array(rings, dtype=np.uint16); RT = self.R * D
        self.lut = np.zeros((self.G // 2, RT, 64), np.uint8); self.out = np.zeros((RT, self.O), np.uint8)
        self.acc = np.zeros(self.O * RT, np.uint16)
        self.pw = np.array([[pow(256, d, m) for m in rings] for d in range(D)], dtype=np.int64)
    def __call__(self, X):
        xd = digits(X, self.D)
        lib.tlut_precompute_digits(P(xd), self.K, self.D, P(self.mods), self.R, P(self.lut))
        self.jit.pull(P(self.codes), P(self.lut), P(self.out), P(self.acc))
        o = self.out.astype(np.int64).reshape(self.D, self.R, self.O)
        Y = (o * self.pw[:, :, None]).sum(axis=0) % np.array(self.rings)[:, None]
        return garner(Y, self.rings)

def med(f, n=15):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
    return float(np.median(ts))

rng = np.random.default_rng(1013)
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 VBMI, one core"}
for n in (4032, 8064):
    W = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(n, n))
    x16 = (rng.standard_normal(n) * np.exp(rng.uniform(-8, 3, n))).astype(np.float16)
    dy16 = (rng.standard_normal(n) * np.exp(rng.uniform(-10, 0, n))).astype(np.float16)
    Wf, WTf = W.astype(np.float32), np.ascontiguousarray(W.T).astype(np.float32)
    row = {}
    # ---- full mode
    Xf = (x16.astype(np.float64) * 2.0 ** 24).astype(np.int64); Df = (dy16.astype(np.float64) * 2.0 ** 24).astype(np.int64)
    fwd, bwd = Engine(W, ALL, 6), Engine(np.ascontiguousarray(W.T), ALL, 6)
    yr, dxr = W.astype(np.int64) @ Xf, W.T.astype(np.int64) @ Df
    y, dx = fwd(Xf), bwd(Df)
    same = True
    for _ in range(3):
        g = rng.permutation(n // 3); idx = (g[:, None] * 3 + np.arange(3)).ravel()
        same &= bool(np.array_equal(Engine(np.ascontiguousarray(W[:, idx]), ALL, 6)(Xf[idx]), y))
    f32y = Wf @ x16.astype(np.float32); f32dx = WTf @ dy16.astype(np.float32)
    row["P1_full"] = {"fwd_mismatches": int(np.sum(y != yr)), "bwd_mismatches": int(np.sum(dx != dxr)), "reorder_bit_identical": same,
        "float32_fwd_outputs_not_exact": int(np.sum(f32y.astype(np.float64) * 2 ** 24 != yr)),
        "float32_bwd_outputs_not_exact": int(np.sum(f32dx.astype(np.float64) * 2 ** 24 != dxr)),
        "us": {"fwd": round(med(lambda: fwd(Xf)) * 1e6), "bwd": round(med(lambda: bwd(Df)) * 1e6),
               "float32_fwd": round(med(lambda: Wf @ x16.astype(np.float32)) * 1e6),
               "float32_bwd": round(med(lambda: WTf @ dy16.astype(np.float32)) * 1e6)}}
    # ---- window mode
    def window(v16):
        v = v16.astype(np.float64); s = 22 - int(np.floor(np.log2(np.max(np.abs(v)))))
        X = np.rint(v * 2.0 ** s).astype(np.int64); assert np.max(np.abs(X)) < 2 ** 23
        rounded = int(np.sum(X.astype(np.float64) / 2.0 ** s != v))
        return X, s, rounded
    Xw, sx, rx = window(x16); Dw, sd, rd = window(dy16)
    fw, bw = Engine(W, ALL[:5], 3), Engine(np.ascontiguousarray(W.T), ALL[:5], 3)
    yw, dxw = fw(Xw), bw(Dw)
    ex_y, ex_dx = yr / 2.0 ** 24, dxr / 2.0 ** 24
    rel = lambda a, b: float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-30)))
    row["P2_window"] = {"fwd_mismatches_vs_windowed_exact": int(np.sum(yw != W.astype(np.int64) @ Xw)),
        "bwd_mismatches_vs_windowed_exact": int(np.sum(dxw != W.T.astype(np.int64) @ Dw)),
        "entry_rounded_values": {"x": rx, "dy": rd, "of": n},
        "max_rel_err_vs_exact_f16_product": {"window_fwd": rel(yw / 2.0 ** sx, ex_y), "float32_fwd": rel(f32y.astype(np.float64), ex_y),
                                             "window_bwd": rel(dxw / 2.0 ** sd, ex_dx), "float32_bwd": rel(f32dx.astype(np.float64), ex_dx)},
        "us": {"fwd": round(med(lambda: fw(Xw)) * 1e6), "bwd": round(med(lambda: bw(Dw)) * 1e6)}}
    u = row["P1_full"]["us"]
    row["P3_speed_vs_float32"] = {"full_fwd": round(u["float32_fwd"] / u["fwd"], 2), "full_bwd": round(u["float32_bwd"] / u["bwd"], 2),
        "window_fwd": round(u["float32_fwd"] / row["P2_window"]["us"]["fwd"], 2), "window_bwd": round(u["float32_bwd"] / row["P2_window"]["us"]["bwd"], 2)}
    res[f"{n}x{n}"] = row
    print(n, json.dumps(row), flush=True)
ok = all(v["P1_full"]["fwd_mismatches"] == 0 and v["P1_full"]["bwd_mismatches"] == 0 and v["P1_full"]["reorder_bit_identical"]
         and v["P2_window"]["fwd_mismatches_vs_windowed_exact"] == 0 and v["P2_window"]["bwd_mismatches_vs_windowed_exact"] == 0
         for k, v in res.items() if k[0].isdigit())
res["verdict"] = "PASS" if ok else "FAIL"
json.dump(res, open(os.path.join(HERE, "prop_f16.result.json"), "w"), indent=1)
print(res["verdict"])
