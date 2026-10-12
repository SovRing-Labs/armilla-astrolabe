#!/usr/bin/env python3
"""
Armilla — the Gelosia lattice rendered into the JIT centre: EXACT f16-activation x ternary GEMV (2026-10-12).
Pass rules fixed before the run.

Gelosia, as used here: the paper method cuts both factors into digits, fills every box of the lattice independently, adds along
the diagonals, and carries once at the end. Here
  - the weight side: ternary digits (trits) in groups of 3 = one pattern code (27 patterns);
  - the activation side: every f16 value is an exact fixed-point integer X = x * 2^24 (f16's smallest step is 2^-24), cut into
    D balanced base-256 digits (each in [-128, 127]);
  - the boxes: the outer rings precompute, for every digit and ring, the 27 answers of every group (tlut_precompute_digits);
  - the pull: the centre pulls each row's box answers for every (digit, ring) with vpermb from ONE code load;
  - the diagonals: digit d carries weight 256^d; the diagonal sum is formed INSIDE each ring (Y_r = sum_d y_{d,r} * (256^d mod m_r)),
    carry-free;
  - the one carry pass: an exact mixed-radix (Garner) read-out over the rings gives y as an integer.
The pull kernel is RENDERED for its shape: moduli, ring count, digit count, table tiling and loop bounds are constants in
generated C, compiled and loaded at run time (THEAD's t50.rs renders raw AVX-512 bytes the same way; porting this kernel to that
byte emitter is the next step).

F1  exactness: y equals the exact integer W @ X for every output (int64 reference).                  PASS = 0 mismatches.
F2  float comparison: NumPy float32 GEMV on the same f16 inputs: number of outputs that differ from the exact value. Reported.
F3  order independence: y over 4 random permutations of the input order is bit-identical.             PASS = identical.
F4  speed (one core, median of 15): rendered engine vs NumPy float32 GEMV vs THEAD-style float path. Reported as measured.
Writes gelosia_f16.result.json
"""
import ctypes, json, os, subprocess, tempfile, time
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
tmp = tempfile.mkdtemp()
base = os.path.join(tmp, "base.so")
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", base, os.path.join(HERE, "armilla_tlut.c")], check=True)
lib = ctypes.CDLL(base); c = ctypes.c_void_p
def P(a): return a.ctypes.data_as(c)

RINGS = [251, 241, 239, 233, 229, 227, 223]            # 7 byte rings: range ~ 2^55 > 2 * 8064 * 2^40
D = 6                                                  # balanced base-256 digits for |X| < 2^40 (f16 * 2^24)

def render(R, D, G, O, QT=48, CH=8):
    """C source for one shape: table sets (digit, ring) processed in chunks of CH per code load."""
    RT = R * D; Q = G // 2
    mods = [RINGS[t % R] for t in range(RT)]                 # table t = d*R + r
    chunks = [list(range(s, min(s + CH, RT))) for s in range(0, RT, CH)]
    L = ["#include <immintrin.h>", "#include <stdint.h>",
         "static inline __m512i br(__m512i a, int m){__m512i q=_mm512_mulhi_epu16(a,_mm512_set1_epi16((short)(65536/m)));"
         "__m512i r=_mm512_sub_epi16(a,_mm512_mullo_epi16(q,_mm512_set1_epi16((short)m)));__m512i mm=_mm512_set1_epi16((short)m);"
         "r=_mm512_min_epu16(r,_mm512_sub_epi16(r,mm));return _mm512_min_epu16(r,_mm512_sub_epi16(r,mm));}",
         f"/* rendered: R={R} rings, D={D} digits, G={G} groups, O={O} rows, tile={QT} group pairs, chunk={CH} */",
         "void pull(const uint8_t *codes2, const uint8_t *lut, uint8_t *out, uint16_t *acc16) {",
         "  const __m512i ones = _mm512_set1_epi8(1);",
         f"  for (int i = 0; i < {O // 32 * RT * 32}; i++) acc16[i] = 0;"]
    for ch in chunks:
        n = len(ch)
        L.append(f"  for (int q0 = 0; q0 < {Q}; q0 += {QT}) {{ int q1 = q0 + {QT} < {Q} ? q0 + {QT} : {Q};")
        L.append(f"    for (int bl = 0; bl < {O // 32}; bl++) {{")
        for i, t in enumerate(ch):
            L.append(f"      __m512i a{i} = _mm512_loadu_si512(acc16 + ((size_t)bl * {RT} + {t}) * 32);")
        L.append(f"      for (int q = q0; q < q1; q++) {{")
        L.append(f"        __m512i idx = _mm512_loadu_si512(codes2 + ((size_t)q * {O} + bl * 32) * 2);")
        L.append(f"        const uint8_t *T = lut + (size_t)q * {RT * 64};")
        for i, t in enumerate(ch):
            L.append(f"        a{i} = _mm512_add_epi16(a{i}, _mm512_maddubs_epi16(_mm512_permutexvar_epi8(idx, _mm512_loadu_si512(T + {t * 64})), ones));")
        L.append("      }")
        for i, t in enumerate(ch):
            L.append(f"      _mm512_storeu_si512(acc16 + ((size_t)bl * {RT} + {t}) * 32, br(a{i}, {mods[t]}));")
        L.append("    }\n  }")
    L.append(f"  for (int bl = 0; bl < {O // 32}; bl++) for (int t = 0; t < {RT}; t++) for (int l = 0; l < 32; l++)")
    L.append(f"    out[(size_t)t * {O} + bl * 32 + l] = (uint8_t)acc16[((size_t)bl * {RT} + t) * 32 + l];")
    L.append("}")
    return "\n".join(L), mods

def balanced_digits(X, D):
    ds = []; v = X.copy()
    for _ in range(D):
        d = ((v + 128) % 256) - 128; ds.append(d.astype(np.int8)); v = (v - d) // 256
    assert np.all(v == 0)
    return np.ascontiguousarray(np.stack(ds))

def garner(res, mods):
    """exact mixed-radix read-out, vectorised over outputs (int64; product of moduli < 2^63)"""
    n = len(mods); digits = []
    for i in range(n):
        a = res[i].astype(np.int64)
        for j in range(i):
            a = (a - digits[j]) * pow(mods[j], -1, mods[i]) % mods[i]
        digits.append(a)
    y = np.zeros_like(digits[0]); w = 1
    for dgt, m in zip(digits, mods):
        y = y + dgt * w; w *= m
    M = w
    return np.where(2 * y >= M, y - M, y)

rng = np.random.default_rng(16)
res = {"date": "2026-10-12", "rings": RINGS, "digits": D, "activation": "f16, exact fixed point X = x * 2^24"}
for O, K in ((4096, 4032), (8192, 8064)):
    G = K // 3; Q = G // 2; R = len(RINGS); RT = R * D
    src, mods = render(R, D, G, O)
    t0 = time.perf_counter()
    open(os.path.join(tmp, "r.c"), "w").write(src)
    subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, f"r{O}.so"), os.path.join(tmp, "r.c")], check=True)
    jit = ctypes.CDLL(os.path.join(tmp, f"r{O}.so")); t_render = time.perf_counter() - t0
    W = rng.choice(np.array([-1, 0, 1], dtype=np.int8), size=(O, K))
    x16 = (rng.standard_normal(K) * np.exp(rng.uniform(-8, 3, K))).astype(np.float16)
    X = (x16.astype(np.float64) * 2.0 ** 24).astype(np.int64)
    assert np.all(X.astype(np.float64) / 2.0 ** 24 == x16.astype(np.float64))          # fixed point is exact
    ref = W.astype(np.int64) @ X
    w3 = W.reshape(O, G, 3).astype(np.int16) + 1
    codes = (w3[:, :, 0] + 3 * w3[:, :, 1] + 9 * w3[:, :, 2]).T.astype(np.uint8)
    c2 = codes.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2[:, :, 1] += 32; codes2 = np.ascontiguousarray(c2)
    MODS = np.array(RINGS, dtype=np.uint16)
    lut = np.zeros((Q, RT, 64), np.uint8); out = np.zeros((RT, O), np.uint8); acc16 = np.zeros(O * RT, np.uint16)
    pw = np.array([[pow(256, d, m) for m in RINGS] for d in range(D)], dtype=np.int64)
    def engine(Xv=X):
        xd = balanced_digits(Xv, D)
        lib.tlut_precompute_digits(P(xd), K, D, P(MODS), R, P(lut))
        jit.pull(P(codes2), P(lut), P(out), P(acc16))
        o = out.astype(np.int64).reshape(D, R, O)
        Y = (o * pw[:, :, None]).sum(axis=0) % np.array(RINGS)[:, None]          # the diagonals, inside each ring
        return garner(Y, RINGS)                                                   # the one carry pass
    y = engine()
    f1 = int(np.sum(y != ref))
    yf32 = W.astype(np.float32) @ x16.astype(np.float32)
    f32_wrong = int(np.sum(yf32.astype(np.float64) * 2.0 ** 24 != ref.astype(np.float64)))
    rel = float(np.max(np.abs(yf32.astype(np.float64) - ref / 2.0 ** 24) / np.maximum(1e-30, np.abs(ref / 2.0 ** 24))))
    same = True
    for _ in range(4):
        perm = rng.permutation(K // 3)
        idx = (perm[:, None] * 3 + np.arange(3)).ravel()                           # permute whole groups of 3
        Wp = W[:, idx]; w3p = Wp.reshape(O, G, 3).astype(np.int16) + 1
        cp = (w3p[:, :, 0] + 3 * w3p[:, :, 1] + 9 * w3p[:, :, 2]).T.astype(np.uint8)
        c2p = cp.reshape(Q, 2, O).transpose(0, 2, 1).copy(); c2p[:, :, 1] += 32
        keep = codes2.copy(); codes2[:] = c2p
        same &= bool(np.array_equal(engine(X[idx]), y)); codes2[:] = keep
    def med(f, n=15):
        ts = []
        for _ in range(n):
            t0 = time.perf_counter(); f(); ts.append(time.perf_counter() - t0)
        return float(np.median(ts))
    Wf = W.astype(np.float32); xf = x16.astype(np.float32)
    t = {"engine": med(engine), "numpy_float32_gemv": med(lambda: Wf @ xf)}
    xd = balanced_digits(X, D)
    t["precompute"] = med(lambda: lib.tlut_precompute_digits(P(xd), K, D, P(MODS), R, P(lut)))
    t["pull"] = med(lambda: jit.pull(P(codes2), P(lut), P(out), P(acc16)))
    res[f"{O}x{K}"] = {"F1_mismatches_vs_exact": f1, "F2_float32_outputs_not_exact": f32_wrong, "F2_float32_max_rel_err": rel,
        "F3_order_independent_bit_identical": same, "render_compile_ms": round(t_render * 1e3, 1), "table_sets": RT,
        "us": {k: round(v * 1e6, 1) for k, v in t.items()}, "engine_vs_float32": round(t["numpy_float32_gemv"] / t["engine"], 2)}
    print(O, K, json.dumps(res[f"{O}x{K}"]), flush=True)
res["verdict"] = "PASS" if all(v["F1_mismatches_vs_exact"] == 0 and v["F3_order_independent_bit_identical"] for k, v in res.items() if k[0].isdigit()) else "FAIL"
json.dump(res, open(os.path.join(HERE, "gelosia_f16.result.json"), "w"), indent=1)
print(res["verdict"])
