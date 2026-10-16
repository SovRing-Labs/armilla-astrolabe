#!/usr/bin/env python3
"""
PAILLIER-MERGE-0 (CYCLONE-SHIELD-0 D1.7, 2026-10-14): two sentinels merge their count vectors as ciphertext; nobody decodes the
other's counts; only the key holder decrypts the SUM. Paillier (1999), additive homomorphism: Enc(a)·Enc(b) mod n² = Enc(a+b).
g = n + 1, so g^m = 1 + m·n mod n²; encryption's one modexp is r^n mod n²; decryption's is c^λ mod n².
Modexp backend: Armilla RNS Montgomery (method of experiments/probes.py P4: bases A / B of 16-bit primes, exact mixed-radix base
extension, centre check channel) — every modexp also checked against Python pow().
Pass rules fixed before the run:
  E1 exactness (n = 512 bits, n² = 1,024): every RNS modexp == pow(); every centre check agrees; Dec(Enc(m)) == m;
     merged vector: Dec(∏ Enc) == plaintext sum, 2 sentinels × 8 bins.                                   PASS = 0 mismatches
  E2 full size (n = 2,048 bits) with pow() only: Dec(Enc(a)·Enc(b)) == a + b; ms per encryption / decryption reported.
  Speed vs GMP on AVX-512 IFMA = NOT measured here (laptop i5-8365U has AVX2 only) — open item.
No security claim: test keys, Python big integers, not constant-time.
Writes paillier_merge.result.json beside this file.
"""
import json, math, os, random, time
HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(1014)


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
        for j in range(i): a = (a - d[j]) * pow(src[j], -1, m) % m
        d.append(a)
    out = []
    for p in dst:
        v, w = 0, 1
        for di, m in zip(d, src): v = (v + di * w) % p; w = w * m % p
        out.append(v)
    return out


class RnsMont:
    """x·y·MA⁻¹ mod N in residues (result < 2N), base A → B → A + centre check."""

    def __init__(self, N):
        pool = [p for p in primes_below(65536) if p > 2][::-1]
        need = (N.bit_length() + 4) // 15 + 2
        self.A, self.B, self.C = pool[0:2 * need:2], pool[1:2 * need:2], pool[2 * need:2 * need + 4]
        self.MA = math.prod(self.A)
        assert self.MA > 4 * N and math.prod(self.B) > 2 * N
        self.N = N
        self.Ninv_A = [(-pow(N, -1, m)) % m for m in self.A]
        self.MAinv_B = [pow(self.MA, -1, m) for m in self.B]
        self.R2 = self.MA * self.MA % N
        self.checks = self.calls = 0

    def mul(self, x, y):
        A, B, C, N = self.A, self.B, self.C, self.N
        q = [(x % m) * (y % m) * n % m for n, m in zip(self.Ninv_A, A)]
        qB = mrc_extend(q, A, B)
        rB = [((x % m) * (y % m) + qb * (N % m)) * ai % m for qb, ai, m in zip(qB, self.MAinv_B, B)]
        rC = mrc_extend(rB, B, C)
        r = crt(rB, B)
        self.calls += 1; self.checks += all(r % m == v for m, v in zip(C, rC))
        return r

    def pow(self, base, e):
        xm = self.mul(base % self.N, self.R2); acc = self.mul(1, self.R2)
        for bit in bin(e)[2:]:
            acc = self.mul(acc, acc)
            if bit == "1": acc = self.mul(acc, xm)
        return self.mul(acc, 1) % self.N


def rand_prime(bits):
    while True:
        c = rng.getrandbits(bits) | 1 << (bits - 1) | 1
        if all(pow(w, c - 1, c) == 1 for w in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)): return c


def keygen(bits):
    while True:
        p, q = rand_prime(bits // 2), rand_prime(bits // 2)
        n = p * q
        if p != q and math.gcd(n, (p - 1) * (q - 1)) == 1: break
    lam = math.lcm(p - 1, q - 1)
    return n, lam, pow(lam, -1, n)


class Paillier:
    def __init__(self, n, lam, mu, modexp):
        self.n, self.n2, self.lam, self.mu, self.modexp = n, n * n, lam, mu, modexp
        self.mismatch = 0

    def _pow(self, b, e):
        v = self.modexp(b, e)
        if v != pow(b, e, self.n2): self.mismatch += 1
        return v

    def enc(self, m):
        while True:
            r = rng.randrange(1, self.n)
            if math.gcd(r, self.n) == 1: break
        return (1 + m * self.n) * self._pow(r, self.n) % self.n2

    def add(self, c1, c2): return c1 * c2 % self.n2

    def dec(self, c):
        u = self._pow(c, self.lam)
        return (u - 1) // self.n * self.mu % self.n


out = {"date": "2026-10-14", "host": "laptop i5-8365U (AVX2, no AVX-512 IFMA)"}
t0 = time.time()
# E1 — exactness on the Armilla RNS Montgomery backend
n, lam, mu = keygen(512)
rm = RnsMont(n * n)
pa = Paillier(n, lam, mu, rm.pow)
bins = 8
s1 = [rng.randrange(0, 10 ** 6) for _ in range(bins)]; s2 = [rng.randrange(0, 10 ** 6) for _ in range(bins)]
te = time.perf_counter(); c1 = [pa.enc(x) for x in s1]; c2 = [pa.enc(x) for x in s2]; t_enc = (time.perf_counter() - te) / (2 * bins)
merged = [pa.add(a, b) for a, b in zip(c1, c2)]
td = time.perf_counter(); dec = [pa.dec(c) for c in merged]; t_dec = (time.perf_counter() - td) / bins
single = pa.dec(c1[0])
e1 = {"n_bits": n.bit_length(), "n2_bits": (n * n).bit_length(), "rings_A": len(rm.A), "rings_B": len(rm.B), "check_rings": len(rm.C),
      "modexp_mismatches_vs_pow": pa.mismatch, "mont_calls": rm.calls, "centre_checks_agreed": rm.checks,
      "dec_enc_roundtrip": single == s1[0], "merged_equals_plain_sum": dec == [a + b for a, b in zip(s1, s2)],
      "ms_per_enc_rns_python": round(t_enc * 1e3, 1), "ms_per_dec_rns_python": round(t_dec * 1e3, 1)}
e1["verdict"] = "PASS" if (pa.mismatch == 0 and rm.checks == rm.calls and e1["dec_enc_roundtrip"] and e1["merged_equals_plain_sum"]) else "FAIL"
out["E1_rns_backend_512"] = e1
# E2 — full size, pow() backend
n, lam, mu = keygen(2048)
pb = Paillier(n, lam, mu, lambda b, e: pow(b, e, n * n))
a, b = rng.randrange(10 ** 9), rng.randrange(10 ** 9)
te = time.perf_counter(); ca, cb = pb.enc(a), pb.enc(b); t2e = (time.perf_counter() - te) / 2
td = time.perf_counter(); s = pb.dec(pb.add(ca, cb)); t2d = time.perf_counter() - td
out["E2_full_2048_pow"] = {"n_bits": n.bit_length(), "sum_ok": s == a + b, "ms_per_enc_pow": round(t2e * 1e3, 1), "ms_per_dec_pow": round(t2d * 1e3, 1)}
out["open"] = "Armilla IFMA modexp vs GMP inside Paillier at 2,048-bit n: needs an AVX-512 IFMA host (pod)"
out["verdict"] = "PASS" if e1["verdict"] == "PASS" and out["E2_full_2048_pow"]["sum_ok"] else "FAIL"
out["seconds"] = round(time.time() - t0, 1)
json.dump(out, open(os.path.join(HERE, "paillier_merge.result.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
