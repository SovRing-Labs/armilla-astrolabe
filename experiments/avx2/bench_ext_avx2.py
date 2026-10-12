#!/usr/bin/env python3
"""Winograd vs plain base extension on an AVX2-only build (no AVX-512 / IFMA), 28-bit rings, 4 messages per pass.
Same CPU, compiled with -mavx2 -mno-avx512f: the legacy instruction mix, not a legacy CPU's timing. Pass rule fixed before the run:
X1 both kernels exact against Python big integers.  X2 time per extension (median of 200, interleaved). Writes bench_ext_avx2.result.json"""
import ctypes, json, os, random, subprocess, tempfile, time
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp(); so = os.path.join(tmp, "e.so")
subprocess.run(["gcc", "-O3", "-mavx2", "-mno-avx512f", "-mtune=haswell", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "ext_avx2.c")], check=True)
asm = subprocess.run(["gcc", "-O3", "-mavx2", "-mno-avx512f", "-S", "-o", "-", os.path.join(HERE, "ext_avx2.c")], capture_output=True, text=True).stdout
lib = ctypes.CDLL(so); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
rng = random.Random(2)
pr = []; c = gmpy2.mpz((1 << 28) - 1)
while len(pr) < 700:
    if gmpy2.is_prime(c): pr.append(int(c))
    c -= 2
res = {"date": "2026-10-12", "build": "gcc -O3 -mavx2 -mno-avx512f", "zmm_or_ifma_in_asm": ("zmm" in asm) or ("vpmadd52" in asm)}
for n in (80, 152, 304):                       # 28-bit rings for ~2,048 / 4,096 / 8,192-bit bases
    A, B = pr[:n], pr[n:2 * n]
    T = np.array([[rng.randrange(1 << 28) % b for b in B] for _ in A], np.uint64)
    xi = np.array([[rng.randrange(a) for a in A] for _ in range(4)], np.uint64)
    bb = np.array(B, np.uint64)
    CT = np.array([sum(int(T[2 * k, j]) * int(T[2 * k + 1, j]) for k in range(n // 2)) % B[j] for j in range(n)], np.uint64)
    ref = np.array([[sum(int(xi[q, i]) * int(T[i, j]) for i in range(n)) % B[j] for j in range(n)] for q in range(4)], np.uint64)
    o1 = np.zeros((4, n), np.uint64); o2 = np.zeros((4, n), np.uint64)
    lib.ext_plain(P(xi), P(T), P(bb), n, P(o1)); lib.ext_wino(P(xi), P(T), P(bb), P(CT), n, P(o2))
    t1, t2 = [], []
    for _ in range(200):
        s = time.perf_counter(); lib.ext_plain(P(xi), P(T), P(bb), n, P(o1)); t1.append(time.perf_counter() - s)
        s = time.perf_counter(); lib.ext_wino(P(xi), P(T), P(bb), P(CT), n, P(o2)); t2.append(time.perf_counter() - s)
    res[str(n)] = {"plain_exact": bool((o1 == ref).all()), "wino_exact": bool((o2 == ref).all()),
                   "plain_us": round(float(np.median(t1)) * 1e6, 2), "wino_us": round(float(np.median(t2)) * 1e6, 2),
                   "wino_speedup": round(float(np.median(t1)) / float(np.median(t2)), 2)}
    print(n, res[str(n)], flush=True)
json.dump(res, open(os.path.join(HERE, "bench_ext_avx2.result.json"), "w"), indent=1)
