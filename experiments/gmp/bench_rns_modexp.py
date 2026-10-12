#!/usr/bin/env python3
"""
Armilla vs GMP — RNS Montgomery modular exponentiation compiled for AVX-512 (2026-10-12). Pass rules fixed before the run.
Two bases of n 28-bit primes (residues in 64-bit lanes); C kernel rns_avx512.c (v2: 32-bit conversion tables, 4 output vectors per pass); GMP 6.3 through gmpy2 as the reference.
G1 exactness: rns powmod == gmpy2.powmod for 20 random (base, exponent) pairs at 2,048 and 4,096 bits.   PASS = 0 mismatches
G2 speed: microseconds per Montgomery multiplication inside one 4,096-bit exponentiation, RNS vs GMP (median of 5). Reported.
Writes bench_rns_modexp.result.json
"""
import ctypes, json, math, os, random, subprocess, tempfile, time
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
so = os.path.join(tmp, "rns.so")
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "rns_avx512.c"), "-lm"], check=True)
lib = ctypes.CDLL(so); vp = ctypes.c_void_p
def P(a): return a.ctypes.data_as(vp)
rng = random.Random(28)

def primes_28(k):
    out = []; c = gmpy2.mpz((1 << 28) - 1)
    while len(out) < k:
        c = gmpy2.prev_prime(c) if hasattr(gmpy2, "prev_prime") else c - 2
        if gmpy2.is_prime(c): out.append(int(c))
    return out

def setup(N, bits):
    n = ((bits + 4) // 28 + 1 + 31) // 32 * 32                     # A > 4N with margin, multiple of 32
    pr = primes_28(2 * n); A = pr[0::2]; B = pr[1::2]
    MA, MB = math.prod(A), math.prod(B)
    assert MA > 16 * N and MB > 16 * N
    u = lambda v: np.array(v, dtype=np.uint64)
    C = dict(n=n, A=A, B=B, MA=MA, MB=MB,
        a=u(A), b=u(B), ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]),
        TAB=np.array([[(MA // a) % b for b in B] for a in A], np.uint32).ravel(), MAb=u([MA % b for b in B]), Nb=u([N % b for b in B]),
        dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=np.array([[(MB // b) % a for a in A] for b in B], np.uint32).ravel(), MBa=u([MB % a for a in A]))
    return C

def to_rns(C, v): return np.array([v % m for m in C["A"]], np.uint64), np.array([v % m for m in C["B"]], np.uint64)
def from_rns_b(C, rb):
    MB = C["MB"]; return sum(int(r) * (MB // b) * pow(MB // b, -1, b) for r, b in zip(rb, C["B"])) % MB

def powmod(C, N, x, e):
    R = C["MA"] % N; R2 = R * R % N
    xa, xb = to_rns(C, x * R % N); oa, ob = to_rns(C, R)              # Montgomery forms (entry, once)
    bits = np.array([int(ch) for ch in bin(e)[2:]], np.uint8)
    outa = np.zeros(C["n"], np.uint64); outb = np.zeros(C["n"], np.uint64); cnt = ctypes.c_int(0)
    args = [C["n"]] + [P(C[k]) for k in ("a", "b", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa")]
    t0 = time.perf_counter()
    lib.rns_modexp(*args, P(xa), P(xb), P(oa), P(ob), P(bits), len(bits), P(outa), P(outb), ctypes.byref(cnt))
    t = time.perf_counter() - t0
    # leave the Montgomery domain: multiply by 1 (in residues) once more
    one_a, one_b = to_rns(C, 1); ra = np.zeros(C["n"], np.uint64); rb = np.zeros(C["n"], np.uint64)
    lib.rns_mont(*args, P(outa), P(outb), P(one_a), P(one_b), P(ra), P(rb))
    return from_rns_b(C, rb) % N, t, cnt.value

res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 (DQ, IFMA present, not yet used)", "gmp": gmpy2.mp_version()}
for bits in (2048, 4096):
    N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1)))
    C = setup(N, bits); bad = 0
    for _ in range(20):
        x = rng.randrange(N); e = rng.getrandbits(bits) | 1 << (bits - 1)
        got, _, _ = powmod(C, N, x, e); bad += got != int(gmpy2.powmod(x, e, N))
    x = rng.randrange(N); e = rng.getrandbits(bits) | 1 << (bits - 1)
    tr = []; tg = []
    for _ in range(5):
        _, t, cnt = powmod(C, N, x, e); tr.append(t / cnt)
        t0 = time.perf_counter(); gmpy2.powmod(x, e, N); tg.append((time.perf_counter() - t0) / cnt)
    r_us, g_us = float(np.median(tr)) * 1e6, float(np.median(tg)) * 1e6
    res[f"{bits}"] = {"moduli_per_base": C["n"], "mismatches": bad, "rns_us_per_modmul": round(r_us, 3),
                      "gmp_us_per_modmul": round(g_us, 3), "gmp_faster_by": round(r_us / g_us, 2)}
    print(bits, res[f"{bits}"], flush=True)
res["verdict"] = "PASS (exact)" if all(res[k]["mismatches"] == 0 for k in ("2048", "4096")) else "FAIL"
json.dump(res, open(os.path.join(HERE, "bench_rns_modexp.result.json"), "w"), indent=1); print(res["verdict"])
