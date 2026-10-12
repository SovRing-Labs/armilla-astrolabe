#!/usr/bin/env python3
"""
Armilla Astrolabe — registered probes (2026-10-12).

P0  the 2026-10-11 headline claims, re-checked (8,000-bit ring product; unweighted core sum)
P1  sign / ReLU in the centred range (fractional CRT + exact mixed-radix fallback) against exact integers
P2  the hexad: six ring systems with disjoint primes (triangle A,B,C + centre + top + bottom),
    two-level CRT (each system reads its own value, the centre joins the six) against exact integers
P3  the evolved Gelosia: limbs in the lattice, every box one carry-free product in the rings,
    diagonals summed in the rings, one read-out per diagonal, one carry pass; against Python int
P4  RNS Montgomery modular multiplication on the hexad (base A = triangle, base B = top + bottom,
    centre = check channel) against Python pow(); correctness only, no security claim

Writes experiments/probes.result.json. Pass/fail rules are fixed in the code before the run.
"""
import json, math, os, random, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "engine"))
from gelosia_astrolabe import AstrolabeSphericalProjector, T50_PRIMES  # noqa: E402

def primes_below(n):
    s = bytearray([1]) * n; s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n) if s[i]]

def crt(res, mods):
    M = math.prod(mods)
    return sum(r * (M // m) * pow(M // m, -1, m) for r, m in zip(res, mods)) % M, M

def mrc_extend(res, src, dst):
    """Exact base extension: mixed-radix digits in the source base, evaluated in each target modulus."""
    d = []
    for i, m in enumerate(src):
        a = res[i]
        for j in range(i):
            a = (a - d[j]) * pow(src[j], -1, m) % m
        d.append(a)
    out = []
    for p in dst:
        v, w = 0, 1
        for di, m in zip(d, src):
            v = (v + di * w) % p; w = w * m % p
        out.append(v)
    return out

R = {"date": "2026-10-12", "host": {"cpu_count": os.cpu_count(), "python": sys.version.split()[0], "numpy": np.__version__}}

# ---------------- P0 the 2026-10-11 claims, re-checked ----------------
astro0 = AstrolabeSphericalProjector(T50_PRIMES)
P50_ = [int(m) for m in T50_PRIMES]; M50_ = math.prod(P50_)
a0 = (1 << 8000) - 1; b0 = (1 << 8000) - 3
rc = [(a0 % m) * (b0 % m) % m for m in P50_]
back = crt(rc, P50_)[0]
r0 = random.Random(0); xs = [r0.randrange(M50_) for _ in range(4000)]
ph = [astro0.phase_sum(np.array([x % m for m in P50_], dtype=np.int64)) for x in xs]
R["P0_old_claims"] = {
    "ring_range_bits": M50_.bit_length(),
    "8000bit_ring_product_equals_true_product": back == a0 * b0,
    "8000bit_ring_product_equals_product_mod_M": back == (a0 * b0) % M50_,
    "old_core_sum_r_over_m_corr_with_X_over_M_n4000": round(float(np.corrcoef(ph, [x / M50_ for x in xs])[0, 1]), 4),
    "old_relu_zeroed_share_of_2000_positive_values_below_1e6": round(sum(
        astro0.phase_sum(np.array([x % m for m in P50_], dtype=np.int64)) < 0.5
        for x in [r0.randrange(1, 10 ** 6) for _ in range(2000)]) / 2000, 3),
    "verdict": "old 8000-bit and Zero-CRT ReLU claims do not hold"}

# ---------------- P1 sign / ReLU ----------------
astro = AstrolabeSphericalProjector(T50_PRIMES)
P50 = [int(m) for m in T50_PRIMES]; M50 = math.prod(P50)
rng = random.Random(12)
vals = list(range(-2000, 2001)) + [M50 // 2 + d for d in range(-1000, 1001)]
vals += [rng.randrange(-(M50 // 2), M50 // 2 + 1) for _ in range(20000)]
wrong = 0; fallback = 0
t0 = time.perf_counter()
for v in vals:
    r = np.array([v % m for m in P50], dtype=np.int64)
    f = astro.fractional_crt(r)
    if min(f, abs(f - 0.5), 1 - f) < 1e-9: fallback += 1
    want = 1 if (v % M50) * 2 < M50 else -1
    wrong += astro.rns_sign(r) != want
dt = time.perf_counter() - t0
rand = vals[-20000:]
t0 = time.perf_counter()
for v in rand[:5000]:
    astro.rns_sign(np.array([v % m for m in P50], dtype=np.int64))
fast_dt = (time.perf_counter() - t0)
R["P1_sign_relu"] = {"rings": 50, "range_bits": M50.bit_length(), "values": len(vals),
    "edge_values_near_0_and_M_over_2": 4001 + 2001, "wrong": wrong, "exact_fallbacks": fallback,
    "us_per_sign_random_values_incl_residue_build": round(fast_dt / 5000 * 1e6, 2),
    "verdict": "PASS" if wrong == 0 else "FAIL"}

# ---------------- P2 hexad ----------------
P16 = [p for p in primes_below(65536) if p > 2]
pool = P16[::-1][:420]                       # 420 largest primes below 2^16
names = ["A", "B", "C", "centre", "top", "bottom"]
sysm = {n: pool[i::6] for i, n in enumerate(names)}  # dealt round-robin: six disjoint systems of 70 rings
allm = [m for n in names for m in sysm[n]]
Mall = math.prod(allm)
bad = 0; t0 = time.perf_counter()
for _ in range(300):
    x = rng.randrange(1 << 3300); y = rng.randrange(1 << 3300)
    rx = [x % m for m in allm]; ry = [y % m for m in allm]
    for op in ("add", "mul"):
        rz = [(a + b) % m if op == "add" else a * b % m for a, b, m in zip(rx, ry, allm)]
        # level 1: each system reads its own value; level 2: the centre joins the six
        local, i = [], 0
        for n in names:
            k = len(sysm[n]); v, Ms = crt(rz[i:i + k], sysm[n]); local.append((v, Ms)); i += k
        z, _ = crt([v for v, _ in local], [Ms for _, Ms in local])
        bad += z != ((x + y) if op == "add" else (x * y)) % Mall
R["P2_hexad"] = {"systems": {n: {"rings": len(sysm[n]), "bits": round(sum(math.log2(m) for m in sysm[n]), 1)} for n in names},
    "lane": "uint16 residues (primes 2^15..2^16)", "total_range_bits": Mall.bit_length(),
    "trials": 600, "mismatches": bad, "seconds": round(time.perf_counter() - t0, 2),
    "verdict": "PASS" if bad == 0 else "FAIL"}

# ---------------- P3 evolved Gelosia ----------------
mods = np.array(P50, dtype=np.int64)
def to_rns_vec(v): return np.array([v % m for m in P50], dtype=np.int64)
def gelosia_mul(a, b, lb):
    mask = (1 << lb) - 1
    A = [(a >> (i * lb)) & mask for i in range(-(-a.bit_length() // lb))]
    B = [(b >> (i * lb)) & mask for i in range(-(-b.bit_length() // lb))]
    K = min(len(A), len(B))
    if K * mask * mask >= M50: raise ValueError("a diagonal could wrap: limb too large for the range")
    RA = np.stack([to_rns_vec(v) for v in A]); RB = np.stack([to_rns_vec(v) for v in B])
    boxes = (RA[:, None, :] * RB[None, :, :]) % mods
    out = 0
    for d in range(len(A) + len(B) - 1):
        i0, i1 = max(0, d - len(B) + 1), min(d, len(A) - 1)
        s = boxes[np.arange(i0, i1 + 1), d - np.arange(i0, i1 + 1)].sum(axis=0) % mods
        out += crt([int(v) for v in s], P50)[0] << (d * lb)
    return out, len(A) * len(B)
g = {}
for bits, lb in ((2000, 150), (8000, 150), (8000, 128)):
    bad = 0; tg = 0.0; tb = 0.0
    for _ in range(20):
        a = rng.getrandbits(bits) | 1 << (bits - 1); b = rng.getrandbits(bits) | 1 << (bits - 1)
        t0 = time.perf_counter(); z, boxes = gelosia_mul(a, b, lb); tg += time.perf_counter() - t0
        t0 = time.perf_counter(); zz = a * b; tb += time.perf_counter() - t0
        bad += z != zz
    worst = math.ceil(bits / lb) * ((1 << lb) - 1) ** 2
    g[f"{bits}bit_limb{lb}"] = {"pairs": 20, "mismatches": bad, "boxes": boxes,
        "worst_diagonal_bits": worst.bit_length(), "range_bits": M50.bit_length(),
        "gelosia_ms_python": round(tg / 20 * 1e3, 3), "python_int_ms": round(tb / 20 * 1e3, 4)}
try:
    gelosia_mul((1 << 8000) - 1, (1 << 8000) - 1, 200); g["limb_200_guard"] = "NOT RAISED (FAIL)"
except ValueError:
    g["limb_200_guard"] = "raised: limb too large for a 311-bit range (as designed)"
g["verdict"] = "PASS" if all(v["mismatches"] == 0 for k, v in g.items() if isinstance(v, dict)) else "FAIL"
R["P3_evolved_gelosia"] = g

# ---------------- P4 RNS Montgomery on the hexad ----------------
A = sysm["A"] + sysm["B"] + sysm["C"]          # the triangle
Bb = sysm["top"] + sysm["bottom"]               # top + bottom
Cc = sysm["centre"]                             # check channel
MA, MB = math.prod(A), math.prod(Bb)
def rand_prime(bits):
    while True:
        c = rng.getrandbits(bits) | 1 << (bits - 1) | 1
        if all(pow(w, c - 1, c) == 1 for w in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)): return c
N = rand_prime(1024) * rand_prime(1024)           # an RSA-2048-shaped modulus (test only)
assert MA > 4 * N and MB > 2 * N
Ninv_A = [(-pow(N, -1, m)) % m for m in A]
MAinv_B = [pow(MA, -1, m) for m in Bb]
def mont(x, y):
    """returns x*y*MA^-1 mod N (< 2N), computed in residues; base A -> B -> A/centre by exact extension."""
    xa = [x % m for m in A]; ya = [y % m for m in A]
    q = [a * b * n % m for a, b, n, m in zip(xa, ya, Ninv_A, A)]     # q = -xy/N mod MA, channel-wise
    qB = mrc_extend(q, A, Bb)
    rB = [((x % m) * (y % m) + qb * (N % m)) * ai % m for qb, ai, m in zip(qB, MAinv_B, Bb)]
    rA = mrc_extend(rB, Bb, A); rC = mrc_extend(rB, Bb, Cc)
    r = crt(rB, Bb)[0]
    check = all(r % m == v for m, v in zip(Cc, rC)) and all(r % m == v for m, v in zip(A, rA))
    return r, check
bad = 0; checks = 0; t0 = time.perf_counter()
for _ in range(200):
    x = rng.randrange(N); y = rng.randrange(N)
    r, ok = mont(x, y); checks += ok
    bad += (r % N) != (x * y * pow(MA, -1, N)) % N or r >= 2 * N
t_mul = (time.perf_counter() - t0) / 200
# a short modular exponentiation in the Montgomery domain
e = rng.getrandbits(64) | 1 << 63; base = rng.randrange(N)
R2 = MA * MA % N
xm, _ = mont(base, R2); acc, _ = mont(1, R2)
for bit in bin(e)[2:]:
    acc, _ = mont(acc, acc)
    if bit == "1": acc, _ = mont(acc, xm)
res, _ = mont(acc, 1)
R["P4_rns_montgomery"] = {"modulus_bits": N.bit_length(), "base_A_triangle_bits": MA.bit_length(),
    "base_B_top_bottom_bits": MB.bit_length(), "centre_check_bits": math.prod(Cc).bit_length(),
    "base_extension": "exact mixed-radix (reference method, not the fast approximate one)",
    "modmuls": 200, "mismatches": bad, "centre_check_agreed": checks,
    "modexp_64bit_exponent_matches_pow": (res % N) == pow(base, e, N),
    "ms_per_modmul_python_reference": round(t_mul * 1e3, 2),
    "security_claim": "none: correctness probe only; not constant-time",
    "verdict": "PASS" if bad == 0 and checks == 200 and (res % N) == pow(base, e, N) else "FAIL"}

out = os.path.join(HERE, "probes.result.json")
json.dump(R, open(out, "w"), indent=1)
print(json.dumps(R, indent=1))
