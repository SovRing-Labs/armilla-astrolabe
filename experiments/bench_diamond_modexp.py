#!/usr/bin/env python3
"""
Armilla — modular exponentiation that stays in residues end to end, on the five-layer diamond (2026-10-12).

Base A = upper half (7 systems x 55 rings), base B = lower half (7 x 55): 385 rings each, 16-bit primes.
RNS Montgomery multiplication (Posch & Posch 1995; Bajard, Didier & Kornerup 1998) with FAST base extension: the extension is one
matrix product (Kawamura et al. 2000 form; the correction count alpha is read from a float sum). All sums are exact in float64
(products < 2^32, at most 385 terms < 2^41 < 2^53), so the extension runs as a BLAS matrix product. Numbers enter the rings once
and leave once; every multiply in between stays in residues.

E1  exactness: every result equals gmpy2.powmod.                                   PASS = 0 mismatches
E2  speed, measured as is: batch of 1 and batch of 256 independent exponentiations (same 4,096-bit exponent, different bases,
    as when one key signs many messages), against GMP powmod one by one on the same core.   Reported.
Writes experiments/bench_diamond_modexp.result.json
"""
import json, math, os, random, sys, time
import numpy as np
import gmpy2

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(4096)
os.environ.setdefault("OMP_NUM_THREADS", "1")

def primes_below(n):
    s = bytearray([1]) * n; s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n) if s[i]]

P16 = [p for p in primes_below(65536) if p > 2][::-1][:15 * 55]
layout = (["top"] + [f"second.{i}" for i in range(3)] + ["centre"] + [f"middle.{i}" for i in range(6)]
          + [f"secondmid.{i}" for i in range(3)] + ["bottom"])
sysm = {n: P16[i::15] for i, n in enumerate(layout)}
upper = ["top", "second.0", "second.1", "second.2", "middle.0", "middle.1", "middle.2"]
lower = ["bottom", "secondmid.0", "secondmid.1", "secondmid.2", "middle.3", "middle.4", "middle.5"]
A = [m for n in upper for m in sysm[n]]; B = [m for n in lower for m in sysm[n]]
MA, MB = math.prod(A), math.prod(B)
a = np.array(A, dtype=np.int64); b = np.array(B, dtype=np.int64)

def rand_prime(bits):
    while True:
        c = gmpy2.mpz(rng.getrandbits(bits) | 1 << (bits - 1) | 1)
        if gmpy2.is_prime(c): return int(c)
N = rand_prime(2048) * rand_prime(2048)
assert MA > 4 * N and MB > 2 * N

# constants
Ai_inv = np.array([pow(MA // ai, -1, ai) for ai in A], dtype=np.int64)
TAB = np.array([[(MA // ai) % bj for bj in B] for ai in A], dtype=np.float64)       # nA x nB
MA_b = np.array([MA % bj for bj in B], dtype=np.int64)
Bj_inv = np.array([pow(MB // bj, -1, bj) for bj in B], dtype=np.int64)
TBA = np.array([[(MB // bj) % ai for ai in A] for bj in B], dtype=np.float64)       # nB x nA
MB_a = np.array([MB % ai for ai in A], dtype=np.int64)
negNinv_a = np.array([(-pow(N, -1, ai)) % ai for ai in A], dtype=np.int64)
N_b = np.array([N % bj for bj in B], dtype=np.int64)
MAinv_b = np.array([pow(MA, -1, bj) for bj in B], dtype=np.int64)
inv_a = 1.0 / a.astype(np.float64); inv_b = 1.0 / b.astype(np.float64)

def ext(xi, T, inv_src, M_dst, mods_dst, exact):
    """base extension of xi (already multiplied by M_i^-1 mod m_i) as one matrix product"""
    s = xi.astype(np.float64) @ T                                  # exact: < 2^41
    f = xi.astype(np.float64) @ inv_src                            # sum xi / m_i = alpha + value / M
    alpha = np.rint(f) if exact else np.floor(f)
    v = (s.astype(np.int64) % mods_dst - (alpha.astype(np.int64)[:, None] * M_dst) % mods_dst) % mods_dst
    return v

def mont(xa, xb, ya, yb):
    q = (xa * ya % a) * negNinv_a % a
    qb = ext(q * Ai_inv % a, TAB, inv_a, MA_b, b, exact=False)    # q may come out as q + MA (allowed: r grows by N)
    rb = (xb * yb % b + qb * N_b % b) % b * MAinv_b % b
    ra = ext(rb * Bj_inv % b, TBA, inv_b, MB_a, a, exact=True)     # r < 3N << MB: alpha = rint(sum) is exact
    return ra, rb

def to_rns(xs):
    return (np.array([[x % m for m in A] for x in xs], dtype=np.int64),
            np.array([[x % m for m in B] for x in xs], dtype=np.int64))

def from_rns_b(rb):
    Mb = gmpy2.mpz(MB); W = [gmpy2.mpz(MB // bj) * pow(MB // bj, -1, bj) for bj in B]
    return [int(sum(int(r) * w for r, w in zip(row, W)) % Mb) for row in rb]

def rns_powmod(bases, e):
    n = len(bases)
    R2 = MA * MA % N
    r2a, r2b = to_rns([R2] * n); xa, xb = to_rns(bases)
    xa, xb = mont(xa, xb, r2a, r2b)                    # into the Montgomery domain
    oa, ob = to_rns([1] * n)
    acc_a, acc_b = mont(oa, ob, r2a, r2b)
    for bit in bin(e)[2:]:
        acc_a, acc_b = mont(acc_a, acc_b, acc_a, acc_b)
        if bit == "1": acc_a, acc_b = mont(acc_a, acc_b, xa, xb)
    ra, rb = mont(acc_a, acc_b, oa, ob)                # out of the Montgomery domain
    return [v % N for v in from_rns_b(rb)]

R = {"date": "2026-10-12", "layout": "1-3-7-3-1 diamond, base A upper 385 rings, base B lower 385 rings",
     "modulus_bits": N.bit_length(), "numpy": np.__version__, "gmp": gmpy2.mp_version(), "threads": 1}
e = rng.getrandbits(4096) | 1 << 4095
for batch in (1, 256):
    bases = [rng.randrange(N) for _ in range(batch)]
    t0 = time.perf_counter(); got = rns_powmod(bases, e); t_rns = time.perf_counter() - t0
    gN, gE = gmpy2.mpz(N), gmpy2.mpz(e)
    t0 = time.perf_counter(); want = [int(gmpy2.powmod(gmpy2.mpz(x), gE, gN)) for x in bases]; t_gmp = time.perf_counter() - t0
    bad = sum(g != w for g, w in zip(got, want))
    R[f"batch_{batch}"] = {"exponent_bits": 4096, "mismatches": bad, "rns_seconds": round(t_rns, 3),
        "rns_ms_per_modexp": round(t_rns / batch * 1e3, 2), "gmp_ms_per_modexp": round(t_gmp / batch * 1e3, 2),
        "gmp_faster_by": round(t_rns / t_gmp, 1)}
    print(batch, R[f"batch_{batch}"], flush=True)
R["verdict"] = "PASS (exact)" if all(R[k]["mismatches"] == 0 for k in R if k.startswith("batch")) else "FAIL"
json.dump(R, open(os.path.join(HERE, "bench_diamond_modexp.result.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
