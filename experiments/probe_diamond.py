#!/usr/bin/env python3
"""
Armilla — the five-layer diamond (J 2026-10-12): top 1 · second 3 · middle 7 (the centre and the six around it) · second-mid 3 ·
bottom 1 = 15 ring systems in one cyclone; every system flows into the centre. 55 rings per system, 16-bit primes, none shared.
Pass rules fixed before the run:
D1  range; mirror bases: base A = top + second + 3 of the middle hexagon (7 systems), base B = the mirror image (7 systems),
    centre = converter / check. Reported (arithmetic).
D2  RNS Montgomery on the mirror bases with a 4,096-bit modulus (RSA-4096 shaped, test only): 100 modular multiplications and a
    64-bit-exponent modexp against Python pow(), centre check on every result.   PASS = 0 mismatches, centre agrees 100/100.
D3  guard mode, r check systems carrying the largest primes; r = 3 (top, bottom, centre) and r = 4 (top, bottom + both
    second-layer middles). Decoder leaves out up to floor(r/2) systems. 200 trials each.
    PASS = clean 200/200; up to t = floor(r/2) faulty systems corrected 200/200; t+1 .. r-t detected 200/200 (field bound when
    correcting t); r faulty systems reported, not gated.
Writes experiments/probe_diamond.result.json
"""
import itertools, json, math, os, random, time

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(15)

def primes_below(n):
    s = bytearray([1]) * n; s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n) if s[i]]

def crt(res, mods):
    M = math.prod(mods)
    return sum(r * (M // m) * pow(M // m, -1, m) for r, m in zip(res, mods)) % M

def mrc_extend(res, src, dst):
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

P16 = [p for p in primes_below(65536) if p > 2][::-1]
RINGS = 55
layout = (["top"] + [f"second.{i}" for i in range(3)] + ["centre"] + [f"middle.{i}" for i in range(6)]
          + [f"secondmid.{i}" for i in range(3)] + ["bottom"])
pool = P16[:15 * RINGS]
sysm = {n: pool[i::15] for i, n in enumerate(layout)}
bits = {n: sum(math.log2(p) for p in s) for n, s in sysm.items()}
upper = ["top", "second.0", "second.1", "second.2", "middle.0", "middle.1", "middle.2"]
lower = ["bottom", "secondmid.0", "secondmid.1", "secondmid.2", "middle.3", "middle.4", "middle.5"]
R = {"date": "2026-10-12", "layout": "1-3-7-3-1 (15 systems)", "rings_per_system": RINGS, "rings_total": 15 * RINGS}

# D1
A = [m for n in upper for m in sysm[n]]; B = [m for n in lower for m in sysm[n]]; C = sysm["centre"]
MA, MB = math.prod(A), math.prod(B)
R["D1_range_and_bases"] = {"range_bits": round(sum(bits.values())), "base_A_upper_bits": MA.bit_length(),
    "base_B_lower_bits": MB.bit_length(), "centre_bits": math.prod(C).bit_length(),
    "max_montgomery_modulus_bits": min(MA.bit_length() - 3, MB.bit_length() - 2),
    "hops_to_centre": "every system is at most 2 hops from the centre (top/bottom -> second layer -> centre)"}

# D2
def rand_prime(b):
    while True:
        c = rng.getrandbits(b) | 1 << (b - 1) | 1
        if all(pow(w, c - 1, c) == 1 for w in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)): return c
N = rand_prime(2048) * rand_prime(2048)
assert MA > 4 * N and MB > 2 * N
NinvA = [(-pow(N, -1, m)) % m for m in A]; MAinvB = [pow(MA, -1, m) for m in B]
def mont(x, y):
    q = [(x % m) * (y % m) * n % m for n, m in zip(NinvA, A)]
    qB = mrc_extend(q, A, B)
    rB = [((x % m) * (y % m) + qb * (N % m)) * ai % m for qb, ai, m in zip(qB, MAinvB, B)]
    rC = mrc_extend(rB, B, C); r = crt(rB, B)
    return r, all(r % m == v for m, v in zip(C, rC))
bad = ok = 0; t0 = time.perf_counter()
for _ in range(100):
    x, y = rng.randrange(N), rng.randrange(N); r, c = mont(x, y); ok += c
    bad += (r % N) != x * y * pow(MA, -1, N) % N or r >= 2 * N
tm = (time.perf_counter() - t0) / 100
e = rng.getrandbits(64) | 1 << 63; base = rng.randrange(N); R2 = MA * MA % N
xm, _ = mont(base, R2); acc, _ = mont(1, R2)
for bit in bin(e)[2:]:
    acc, _ = mont(acc, acc)
    if bit == "1": acc, _ = mont(acc, xm)
res, _ = mont(acc, 1)
R["D2_montgomery_4096"] = {"modulus_bits": N.bit_length(), "modmuls": 100, "mismatches": bad, "centre_check_agreed": ok,
    "modexp_64bit_exponent_matches_pow": res % N == pow(base, e, N), "ms_per_modmul_python_reference": round(tm * 1e3, 1),
    "security_claim": "none: correctness only, not constant-time",
    "verdict": "PASS" if bad == 0 and ok == 100 and res % N == pow(base, e, N) else "FAIL"}

# D3
def guard(checks):
    big = P16[:len(checks) * RINGS]; rest = P16[len(checks) * RINGS:15 * RINGS]
    data_names = [n for n in layout if n not in checks]
    sm = {n: big[i::len(checks)] for i, n in enumerate(checks)}
    sm.update({n: rest[i::len(data_names)] for i, n in enumerate(data_names)})
    names = data_names + checks
    Ms = [math.prod(sm[n]) for n in names]; MD = math.prod(Ms[:len(data_names)]); ns = len(Ms)
    r = len(checks); t = r // 2
    def decode(v):
        x = crt(v, Ms)
        if x < MD: return ("ok", x)
        for k in range(1, t + 1):
            for out in itertools.combinations(range(ns), k):
                keep = [j for j in range(ns) if j not in out]
                y = crt([v[j] for j in keep], [Ms[j] for j in keep])
                if y < MD: return ("corrected", y)
        return ("detected", None)
    st = {"check_systems": checks, "data_range_bits": MD.bit_length(), "clean_ok": 0}
    for f in range(1, r + 1):
        st[f"{f}_faulty_right"] = 0; st[f"{f}_faulty_wrong_value"] = 0
    T = 200
    for _ in range(T):
        x = rng.randrange(MD); v = [x % m for m in Ms]
        st["clean_ok"] += decode(v) == ("ok", x)
        for f in range(1, r + 1):
            w = list(v)
            for i in rng.sample(range(ns), f): w[i] = (w[i] + rng.randrange(1, Ms[i])) % Ms[i]
            d = decode(w)
            right = (d == ("corrected", x)) if f <= t else (d[0] == "detected")
            st[f"{f}_faulty_right" if right else f"{f}_faulty_wrong_value"] += 1
    st["rule"] = f"corrects up to {t}; while correcting, detects up to {r - t}"
    st["verdict"] = "PASS" if st["clean_ok"] == T and all(st[f"{f}_faulty_right"] == T for f in range(1, r - t + 1)) else "FAIL"
    return st
R["D3_guard_r3"] = guard(["top", "bottom", "centre"])
R["D3_guard_r4"] = guard(["top", "bottom", "second.0", "secondmid.0"])
json.dump(R, open(os.path.join(HERE, "probe_diamond.result.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
