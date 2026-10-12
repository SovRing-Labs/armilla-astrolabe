#!/usr/bin/env python3
"""Times the AVX2 Gelosia lattice core against GMP (gmpy2) and Python int; checks every product exactly.
Usage: bench_gelosia_kernel.py <path-to-libgelosia.so> <label>   (build: see gelosia_kernel.c / README)"""
import ctypes, json, math, os, random, sys, time
import numpy as np
import gmpy2
HERE = os.path.dirname(os.path.abspath(__file__))
lib = ctypes.CDLL(sys.argv[1]); label = sys.argv[2]
P = [3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,
     131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233]
L = 64; mods = np.ones(L, dtype=np.uint32); mods[:50] = P
M = math.prod(P); Mi = [M // p for p in P]; ki = [pow(M // p, -1, p) for p in P]
W = [gmpy2.mpz(a * b) for a, b in zip(Mi, ki)]; Mg = gmpy2.mpz(M)
u32p = ctypes.POINTER(ctypes.c_uint32)
def ptr(a): return a.ctypes.data_as(u32p)
def limbs(x, lb): return [int(v) for v in gmpy2.mpz(x).digits(2 ** lb if False else 16)] if False else [(x >> (i * lb)) & ((1 << lb) - 1) for i in range(-(-x.bit_length() // lb))]
def to_planes(ls):
    R = np.zeros((len(ls), L), dtype=np.uint32)
    for i, v in enumerate(ls):
        g = gmpy2.mpz(v); R[i, :50] = [int(g % p) for p in P]
    return R
def run(a, b, lb, T):
    t0 = time.perf_counter(); RA = to_planes(limbs(a, lb)); RB = to_planes(limbs(b, lb)); T["in"] += time.perf_counter() - t0
    D = np.zeros((RA.shape[0] + RB.shape[0] - 1, L), dtype=np.uint32)
    t0 = time.perf_counter()
    for _ in range(T["reps"]): lib.gelosia_core(ptr(RA), RA.shape[0], ptr(RB), RB.shape[0], ptr(mods), ptr(D))
    T["core"] += (time.perf_counter() - t0) / T["reps"]
    t0 = time.perf_counter(); out = gmpy2.mpz(0)
    for d in range(D.shape[0]):
        v = sum(int(r) * w for r, w in zip(D[d, :50], W)) % Mg   # one read-out per diagonal
        out += v << (d * lb)                                     # the one carry pass
    T["out"] += time.perf_counter() - t0
    return out
rng = random.Random(7); res = {"label": label, "host_cpus": os.cpu_count(), "gmp": gmpy2.mp_version()}
for bits, lb in ((2000, 150), (8000, 150)):
    T = {"in": 0, "core": 0, "out": 0, "reps": 200}; bad = 0; n = 50; tg = tp = 0
    for _ in range(n):
        a = rng.getrandbits(bits) | 1 << (bits - 1); b = rng.getrandbits(bits) | 1 << (bits - 1)
        bad += run(a, b, lb, T) != a * b
        ga, gb = gmpy2.mpz(a), gmpy2.mpz(b)
        t0 = time.perf_counter()
        for _ in range(2000): ga * gb
        tg += (time.perf_counter() - t0) / 2000
        t0 = time.perf_counter()
        for _ in range(2000): a * b
        tp += (time.perf_counter() - t0) / 2000
    K = -(-bits // lb)
    res[f"{bits}bit"] = {"pairs": n, "mismatches": bad, "limb_bits": lb, "boxes": K * K,
        "core_us": round(T["core"] / n * 1e6, 2), "convert_in_us": round(T["in"] / n * 1e6, 1),
        "convert_out_us": round(T["out"] / n * 1e6, 1), "gmp_mul_us": round(tg / n * 1e6, 3),
        "python_int_mul_us": round(tp / n * 1e6, 3),
        "core_vs_gmp": round((T["core"] / n) / (tg / n), 1)}
res["verdict"] = "PASS (exact)" if all(v["mismatches"] == 0 for k, v in res.items() if isinstance(v, dict)) else "FAIL"
print(json.dumps(res, indent=1))
json.dump(res, open(os.path.join(HERE, f"bench_gelosia_kernel.{label}.result.json"), "w"), indent=1)
