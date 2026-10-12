#!/usr/bin/env python3
"""
Armilla vs GMP — RNS Montgomery modexp with AVX-512 IFMA and pseudo-Mersenne rings 2^48 - c (2026-10-12).
Pass rules fixed before the run.
I1 exactness: == gmpy2.powmod for 20 random (base, exponent) pairs at 1,024, 2,048, 3,072 and 4,096 bits.  PASS = 0 mismatches
I2 speed: per Montgomery multiplication inside one full-length exponentiation, RNS-IFMA vs GMP, measured INTERLEAVED
   (15 pairs, median) because this shared cloud CPU varies between runs. Reported as measured.
Writes bench_ifma_modexp.result.json
"""
import ctypes, json, math, os, random, subprocess, tempfile, time
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
so = os.path.join(tmp, "ifma.so")
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "rns_ifma.c"), "-lm"], check=True)
lib = ctypes.CDLL(so); vp = ctypes.c_void_p
def P(a): return a.ctypes.data_as(vp)
rng = random.Random(48)
PR = []; cc = 1
while len(PR) < 400:
    if gmpy2.is_prime((1 << 48) - cc): PR.append((1 << 48) - cc)
    cc += 2

def setup(N, bits):
    n = ((bits + 6 + 47) // 48 + 7) // 8 * 8
    A, B = PR[0:2 * n:2], PR[1:2 * n:2]; MA, MB = math.prod(A), math.prod(B)
    assert MA > 32 * N and MB > 32 * N
    u = lambda v: np.array(v, dtype=np.uint64)
    return dict(n=n, A=A, B=B, MA=MA, MB=MB, a=u(A), b=u(B), ca=u([(1 << 48) - m for m in A]), cb=u([(1 << 48) - m for m in B]),
        ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]),
        TAB=u([[(MA // a) % b for b in B] for a in A]).ravel(), MAb=u([MA % b for b in B]), Nb=u([N % b for b in B]),
        dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=u([[(MB // b) % a for a in A] for b in B]).ravel(), MBa=u([MB % a for a in A]))

KEYS = ("a", "b", "ca", "cb", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa")
def to_rns(C, v): return np.array([v % m for m in C["A"]], np.uint64), np.array([v % m for m in C["B"]], np.uint64)
def powmod(C, N, x, e):
    R = C["MA"] % N
    xa, xb = to_rns(C, x * R % N); oa, ob = to_rns(C, R)
    bits = np.array([int(ch) for ch in bin(e)[2:]], np.uint8)
    outa = np.zeros(C["n"], np.uint64); outb = np.zeros(C["n"], np.uint64); cnt = ctypes.c_int(0)
    args = [C["n"]] + [P(C[k]) for k in KEYS]
    t0 = time.perf_counter()
    lib.ifma_modexp(*args, P(xa), P(xb), P(oa), P(ob), P(bits), len(bits), P(outa), P(outb), ctypes.byref(cnt))
    t = time.perf_counter() - t0
    one_a, one_b = to_rns(C, 1); ra = np.zeros(C["n"], np.uint64); rb = np.zeros(C["n"], np.uint64)
    lib.ifma_mont(*args, P(outa), P(outb), P(one_a), P(one_b), P(ra), P(rb))
    MB = C["MB"]; v = sum(int(r) * (MB // b) * pow(MB // b, -1, b) for r, b in zip(rb, C["B"])) % MB
    return v % N, t, cnt.value

res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 IFMA, one core", "gmp": gmpy2.mp_version(),
       "rings": "primes 2^48 - c, c < 2^16"}
for bits in (1024, 2048, 3072, 4096):
    N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1))); C = setup(N, bits); bad = 0
    for _ in range(20):
        x = rng.randrange(N); e = rng.getrandbits(bits) | 1 << (bits - 1)
        bad += powmod(C, N, x, e)[0] != int(gmpy2.powmod(x, e, N))
    x = rng.randrange(N); e = rng.getrandbits(bits) | 1 << (bits - 1); gx, ge, gn = gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)
    tr, tg = [], []
    for _ in range(15):
        _, t, cnt = powmod(C, N, x, e); tr.append(t / cnt)
        t0 = time.perf_counter(); gmpy2.powmod(gx, ge, gn); tg.append((time.perf_counter() - t0) / cnt)
    r, g = float(np.median(tr)) * 1e6, float(np.median(tg)) * 1e6
    res[str(bits)] = {"moduli_per_base": C["n"], "mismatches": bad, "rns_ifma_us_per_modmul": round(r, 3),
                      "gmp_us_per_modmul": round(g, 3), "rns_vs_gmp_speed": round(g / r, 2),
                      "rns_ms_per_modexp": round(r * cnt / 1e3, 3), "gmp_ms_per_modexp": round(g * cnt / 1e3, 3)}
    print(bits, res[str(bits)], flush=True)
res["verdict"] = "PASS (exact)" if all(res[k]["mismatches"] == 0 for k in ("1024", "2048", "3072", "4096")) else "FAIL"
json.dump(res, open(os.path.join(HERE, "bench_ifma_modexp.result.json"), "w"), indent=1); print(res["verdict"])
