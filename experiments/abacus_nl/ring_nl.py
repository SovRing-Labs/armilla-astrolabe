#!/usr/bin/env python3
"""
Armilla — the non-linear math of a training step done INSIDE the rings (ABACUS-NL-0 step 1, 2026-10-13).

The abacus already multiplies and adds carry-free and exactly. Here exp, 1/x, ln, softmax and log-sum-exp are written as
algorithms that use only what the rings can do exactly:
  ADD / SUB / MUL          per ring, carry-free
  SCALE                    exact floor(X / S) by RNS scaling (Szabo & Tanaka 1967; Barsi & Maestrini): X mod S from the scale
                           group, subtract, multiply by S^-1 on the other rings, mixed-radix base extension back. Signed
                           values pass through a bias B = S * 2^100.
  COMPARE                  exact, by mixed-radix digits (the fractional CRT of TECHNICAL-DISCLOSURE section 5 is the fast path)
  PULL                     a constant chosen by a small exact index (the pull engine of section 8)
Rings: the T50 OUTER rings = the 20 largest T50 primes 131 ... 233 (range ~2^150). Fixed point: value = X / S where S is
the product of the SCALE GROUP = the 5 largest rings (~2^39), not a power of two.
Ring floating point: exp(-w) = mantissa * 2^-z with z an exact small integer, so precision is relative, not absolute.

  exp(-w):  z = floor(w / ln2) (two SCALEs), r = w - z ln2, e^-r by Horner on c_k = round(S / k!), 2^-z kept as the exponent
  sum:      mantissa * PULL T[z] (T[z] = round(S^2 2^-z)), summed unscaled, SCALEd twice (lazy reduction, as in the Gelosia)
  1/s:      K from COMPAREs against 2^j S; t = s 2^-(K+1) by PULL; Newton y <- y (2 - t y) from 48/17 - 32/17 t
  ln s:     (K+1) ln2 + 2 atanh((t-1)/(t+1)), with 1/(t+1) by Newton
  softmax_i = SCALE(mantissa_i * y) * 2^-(z_i + K + 1);  LSE = max + ln s;  cross-entropy = LSE - x_target

The algorithm is written ONCE (class Algo) over a backend. Backend IntOps = Python integers with floor division by S (the
integer twin). Backend RingOps = residue vectors on the 20 rings. Pass rules fixed before the run:
  T1 bit-identity      RingOps == IntOps on every output integer (mantissas, exponents, LSE, loss)        PASS = 0 mismatches
  T2 order-free        a shuffled row order of the sum gives the same bits                                PASS = identical
  T3 accuracy          softmax max relative error and LSE max absolute error vs float64 (math.exp/log)    reported; bar 1e-9
Writes ring_nl.result.json next to this file.
"""
import json, math, os, random, sys, time
from decimal import Decimal, getcontext
import numpy as np

getcontext().prec = 80
T50 = [p for p in range(3, 234) if all(p % q for q in range(2, int(p ** 0.5) + 1))]
assert len(T50) == 50 and T50[-1] == 233
OUTER = sorted(T50[-20:], reverse=True)               # 233 ... 131
G = OUTER[:5]                                          # scale group
RR = OUTER[5:]                                         # the other 15 rings
MODS = RR + G                                          # ring order used everywhere (rest first, scale group last)
S = math.prod(G); M = math.prod(MODS); MR = math.prod(RR)
BIAS_K = 100; B = S << BIAS_K                          # |X| < B and X + B < M are required for SCALE / COMPARE
assert 2 * B < M

LN2 = Decimal(2).ln()
def rnd(d): return int(d.to_integral_value())          # round half even on an 80-digit decimal
C_INVLN2 = rnd(Decimal(S) / LN2)
C_LN2 = rnd(Decimal(S) * LN2)
C_LN2_2 = rnd(Decimal(S) * Decimal(S) * LN2)      # ln2 at S^2 so z * ln2 keeps full precision for large z
EXP_D = 14; C_EXP = [rnd(Decimal(S) / math.factorial(k)) for k in range(EXP_D + 1)]
ZMAX = 90; T2 = [rnd(Decimal(S) * Decimal(S) / (Decimal(2) ** z)) for z in range(ZMAX + 1)]   # PULL table, S^2 2^-z
KMAX = 20                                              # rows up to 2^20 entries
C48 = rnd(Decimal(48) * S / 17); C32 = rnd(Decimal(32) * S / 17)
C_1_175 = rnd(Decimal(S) / Decimal("1.75"))
ATANH_J = 14; C_ATANH = [rnd(Decimal(S) / (2 * j + 1)) for j in range(ATANH_J + 1)]
NEWTON = 5

class Algo:
    """The one algorithm. `o` is the backend; every arithmetic step goes through it."""
    def __init__(self, o): self.o = o

    def scale2(self, x): return self.o.scale(self.o.scale(x))

    def rowmax(self, X):                               # X: backend array (rows, n); tournament of exact COMPAREs
        o = self.o; cur = X
        while o.n(cur) > 1:
            n = o.n(cur)
            if n % 2: cur = o.cat(cur, o.take(cur, [n - 1]))
            a = o.take(cur, list(range(0, o.n(cur), 2))); b = o.take(cur, list(range(1, o.n(cur), 2)))
            cur = o.select(o.ge(a, b), a, b)
        return cur                                     # (rows, 1)

    def exp_neg(self, W):                              # W >= 0 fixed -> (mantissa fixed in (0.5, 1], exponent z ints)
        o = self.o
        Q = o.scale(o.mulc(W, C_INVLN2))               # ~ w / ln2, fixed
        Z = o.scale(Q)                                 # floor: integer z (as a ring value)
        R = o.sub(W, o.scale(o.mulc(Z, C_LN2_2)))      # r = w - z ln2, fixed
        NR = o.neg(R); acc = o.const_like(W, C_EXP[EXP_D])
        for k in range(EXP_D - 1, -1, -1):
            acc = o.addc(o.scale(o.mul(acc, NR)), C_EXP[k])
        return acc, o.small_int(Z)                     # z read out exactly (it indexes the PULL table)

    def recip_newton(self, Tt, y0, iters):
        o = self.o; y = y0
        for _ in range(iters):
            y = o.scale(o.mul(y, o.rsubc(o.scale(o.mul(Tt, y)), 2 * S)))
        return y

    def softmax_lse(self, X, target=None):
        o = self.o
        MX = self.rowmax(X)
        W = o.sub(o.bcast(MX, o.n(X)), X)              # w_i = max - x_i >= 0
        Mant, z = self.exp_neg(W)
        zc = np.minimum(z, ZMAX)
        SUM = self.scale2(o.rowsum(o.mul(Mant, o.pull(T2, zc))))   # s = sum m_i 2^-z_i, fixed; value in [1, n]
        K = np.zeros(o.rows(X), dtype=np.int64)
        for j in range(1, KMAX + 1):                   # K = #{j : s >= 2^j}
            K += o.ge_const(SUM, S << j).astype(np.int64)
        Tt = self.scale2(o.mul(SUM, o.pull(T2, K + 1)))            # t = s 2^-(K+1) in [0.5, 1)
        y0 = o.rsubc(o.scale(o.mulc(Tt, C32)), C48)
        Y = self.recip_newton(Tt, y0, NEWTON)                      # 1/t in (1, 2]
        P = o.scale(o.mul(Mant, o.bcast(Y, o.n(X))))               # softmax mantissa
        E = z + (K + 1)[:, None]                                   # softmax exponent: p_i = P_i / S * 2^-E_i
        Tp1 = o.addc(Tt, S)                                        # t + 1 in [1.5, 2)
        V = self.recip_newton(Tp1, o.const_like(Tt, C_1_175), NEWTON)
        U = o.scale(o.mul(o.addc(Tt, -S), V))                      # u = (t-1)/(t+1) in [-1/3, 0)
        U2 = o.scale(o.mul(U, U)); acc = o.const_like(Tt, C_ATANH[ATANH_J])
        for j in range(ATANH_J - 1, -1, -1):
            acc = o.addc(o.scale(o.mul(acc, U2)), C_ATANH[j])
        LNT = o.mulc(o.scale(o.mul(acc, U)), 2)                    # ln t
        LSE = o.add(o.add(MX, o.mulk(o.const_like(Tt, C_LN2), K + 1)), LNT)
        out = {"P": P, "E": E, "LSE": LSE}
        if target is not None:
            out["LOSS"] = o.sub(LSE, o.take_rows(X, target))
        return out

class IntOps:
    """The integer twin: Python integers in numpy object arrays, SCALE = floor division by S."""
    def lift(self, Xint): return np.array(Xint, dtype=object)
    def n(self, a): return a.shape[1]
    def rows(self, a): return a.shape[0]
    def cat(self, a, b): return np.concatenate([a, b], axis=1)
    def take(self, a, idx): return a[:, idx]
    def take_rows(self, a, t): return a[np.arange(a.shape[0]), t][:, None]
    def bcast(self, a, n): return np.repeat(a, n, axis=1)
    def select(self, c, a, b): return np.where(c, a, b)
    def ge(self, a, b): return np.vectorize(lambda u, v: u >= v, otypes=[bool])(a, b)
    def ge_const(self, a, c): return np.array([bool(v >= c) for v in a[:, 0]])
    def scale(self, a): return a // S
    def mul(self, a, b): return a * b
    def mulc(self, a, c): return a * c
    def mulk(self, a, k): return a * np.array(k, dtype=object)[:, None]
    def add(self, a, b): return a + b
    def addc(self, a, c): return a + c
    def sub(self, a, b): return a - b
    def neg(self, a): return -a
    def rsubc(self, a, c): return c - a
    def const_like(self, a, c): return np.full(a.shape, c, dtype=object)
    def rowsum(self, a):                               # integer sum, any order
        return np.array([[sum(r)] for r in a], dtype=object)
    def pull(self, table, idx):
        idx = np.asarray(idx); return np.vectorize(lambda i: table[i], otypes=[object])(idx if idx.ndim == 2 else idx[:, None])
    def small_int(self, a): return a.astype(np.int64)
    def ints(self, a): return a

class RingOps:
    """Residue vectors on the 20 outer rings: arrays of shape (20, rows, n), int64, every value in [0, m)."""
    def __init__(self):
        self.m = np.array(MODS, dtype=np.int64)[:, None, None]
        self.nm = len(MODS); self.nr = len(RR)
        # MRC inverses: inv[i][j] = m_j^-1 mod m_i (j < i)
        self.inv = [[pow(MODS[j], -1, MODS[i]) if j < i else 0 for j in range(self.nm)] for i in range(self.nm)]
        self.g_inv = [pow(S // g, -1, g) for g in G]                  # CRT over the scale group
        self.s_inv_r = [pow(S, -1, m) for m in RR]
        self.Bres = self.res_of(B); self.BSres = self.res_of(B // S)
        self.B_digits = self.mrc_int(B)
    def res_of(self, c): return np.array([c % m for m in MODS], dtype=np.int64)[:, None, None]
    def lift(self, Xint):
        X = np.array(Xint, dtype=object)
        return np.stack([(X % m).astype(np.int64) for m in MODS])
    def n(self, a): return a.shape[2]
    def rows(self, a): return a.shape[1]
    def cat(self, a, b): return np.concatenate([a, b], axis=2)
    def take(self, a, idx): return a[:, :, idx]
    def take_rows(self, a, t): return a[:, np.arange(a.shape[1]), t][:, :, None]
    def bcast(self, a, n): return np.repeat(a, n, axis=2)
    def select(self, c, a, b): return np.where(c[None], a, b)
    def mul(self, a, b): return a * b % self.m
    def mulc(self, a, c): return a * self.res_of(c) % self.m
    def mulk(self, a, k): return a * (np.array(k, dtype=np.int64)[None, :, None] % self.m) % self.m
    def add(self, a, b): return (a + b) % self.m
    def addc(self, a, c): return (a + self.res_of(c)) % self.m
    def sub(self, a, b): return (a - b) % self.m
    def neg(self, a): return (-a) % self.m
    def rsubc(self, a, c): return (self.res_of(c) - a) % self.m
    def const_like(self, a, c): return np.broadcast_to(self.res_of(c), a.shape).copy()
    def rowsum(self, a): return (a.sum(axis=2, keepdims=True)) % self.m    # residues < 233, n < 2^40: no overflow
    def pull(self, table, idx):
        idx = np.asarray(idx); idx = idx if idx.ndim == 2 else idx[:, None]
        tab = np.stack([np.array([t % m for t in table], dtype=np.int64) for m in MODS])   # (20, len)
        return tab[:, idx]
    def mrc(self, a, k=None):                          # mixed-radix digits over the first k rings
        k = self.nm if k is None else k; d = []
        for i in range(k):
            x = a[i].copy(); mi = MODS[i]
            for j in range(i):
                x = (x - d[j]) * self.inv[i][j] % mi
            d.append(x)
        return d
    def mrc_int(self, c):
        d = []
        for i, mi in enumerate(MODS):
            x = c % mi
            for j in range(i): x = (x - d[j]) * self.inv[i][j] % mi
            d.append(x)
        return d
    def scale(self, a):                                # exact floor(X / S) for |X| < B
        x = (a + self.Bres) % self.m
        acc = np.zeros(x.shape[1:], dtype=np.int64)
        for gi, g in enumerate(G):                     # X' mod S by CRT over the scale group
            acc += (x[self.nr + gi] * self.g_inv[gi] % g) * (S // g)
        amod = acc % S
        y = np.empty_like(x)
        for i, m in enumerate(RR):
            y[i] = (x[i] - amod % m) % m * self.s_inv_r[i] % m
        d = self.mrc(y, self.nr)                       # extend floor(X'/S) from the 15 rings to the scale group
        for gi, g in enumerate(G):
            v = np.zeros_like(d[0])
            for i in range(self.nr - 1, -1, -1):
                v = (v * RR[i] + d[i]) % g
            y[self.nr + gi] = v
        return (y - self.BSres) % self.m
    def nonneg(self, a):                               # X >= 0 for |X| < B: compare X + B with B by mixed-radix digits
        d = self.mrc((a + self.Bres) % self.m)
        gt = np.zeros(a.shape[1:], dtype=bool); lt = np.zeros(a.shape[1:], dtype=bool)
        for i in range(self.nm - 1, -1, -1):
            und = ~(gt | lt)
            gt |= und & (d[i] > self.B_digits[i]); lt |= und & (d[i] < self.B_digits[i])
        return ~lt
    def ge(self, a, b): return self.nonneg(self.sub(a, b))
    def ge_const(self, a, c): return self.nonneg(self.addc(a, -c))[:, 0]
    def small_int(self, a):                            # exact read-out of a small non-negative integer (< m0 * m1)
        d = self.mrc(a, 3)
        assert not d[2].any(), "value too large for a 2-digit read-out"
        return d[0] + d[1] * MODS[0]
    def ints(self, a):                                 # full CRT back to signed Python ints (checking only)
        d = self.mrc(a); out = np.zeros(a.shape[1:], dtype=object); w = 1
        for i, mi in enumerate(MODS):
            out = out + d[i].astype(object) * w; w *= mi
        return np.where(out > M // 2, out - M, out)


def quantise(xs): return [[int(round(float(v) * float(S))) for v in row] for row in xs]

def to_float(P, E): return np.vectorize(lambda p, e: math.ldexp(float(p) / float(S), -int(e)), otypes=[float])(P, E)

def run(rows=64, n=512, seed=0, spread=8.0):
    rnd_ = random.Random(seed)
    xs = [[rnd_.gauss(0, spread) for _ in range(n)] for _ in range(rows)]
    tgt = np.array([rnd_.randrange(n) for _ in range(rows)])
    Xi = quantise(xs)
    t0 = time.perf_counter(); oi = Algo(IntOps()).softmax_lse(IntOps().lift(Xi), tgt); ti = time.perf_counter() - t0
    ro = RingOps(); t0 = time.perf_counter(); orr = Algo(ro).softmax_lse(ro.lift(Xi), tgt); tr = time.perf_counter() - t0
    mism = 0
    for k in ("P", "LSE", "LOSS"):
        mism += int((ro.ints(orr[k]) != oi[k]).sum())
    mism += int((orr["E"] != np.asarray(oi["E"], dtype=np.int64)).sum())
    # T2: shuffled order of the row (the sum and the max see a different order)
    perm = list(range(n)); rnd_.shuffle(perm)
    Xs = [[r[j] for j in perm] for r in Xi]; inv = np.argsort(perm)
    o2 = Algo(ro).softmax_lse(ro.lift(Xs), np.array([int(np.where(np.array(perm) == t)[0][0]) for t in tgt]))
    same = bool((ro.ints(o2["P"])[:, inv] == ro.ints(orr["P"])).all()) and bool((ro.ints(o2["LSE"]) == ro.ints(orr["LSE"])).all())
    # T3: accuracy vs float64 on the quantised inputs
    p = to_float(oi["P"], oi["E"]); worst_rel = 0.0; worst_lse = 0.0
    for r in range(rows):
        xq = [v / S for v in Xi[r]]; mx = max(xq); e = [math.exp(v - mx) for v in xq]; s = math.fsum(e)
        ref = [v / s for v in e]
        worst_rel = max(worst_rel, max(abs(p[r][i] - ref[i]) / ref[i] for i in range(n) if ref[i] > 1e-300))
        worst_lse = max(worst_lse, abs(float(oi["LSE"][r][0]) / S - (mx + math.log(s))))
    return {"rows": rows, "n": n, "spread": spread, "T1_ring_vs_int_mismatches": mism, "T2_shuffled_identical": same,
            "T3_softmax_max_rel_err": worst_rel, "T3_lse_max_abs_err": worst_lse,
            "secs_int": round(ti, 2), "secs_ring_numpy": round(tr, 2)}

if __name__ == "__main__":
    R = {"date": "2026-10-13", "rings": MODS, "scale_group": G, "S_bits": S.bit_length(), "range_bits": M.bit_length(),
         "exp_terms": EXP_D, "newton_iters": NEWTON, "atanh_terms": ATANH_J}
    R["runs"] = [run(64, 512, 0, 8.0), run(64, 512, 1, 30.0), run(200, 40, 2, 25.0)]
    # teeth: a ring backend with one wrong scaling constant must be caught by T1
    _init = RingOps.__init__
    def bad_init(self):
        _init(self); self.s_inv_r[3] = (self.s_inv_r[3] + 1) % RR[3]
    RingOps.__init__ = bad_init
    try:
        tb = run(8, 64, 5, 8.0); caught = tb["T1_ring_vs_int_mismatches"]
    except AssertionError as e:
        caught = "caught by the read-out range guard: " + str(e)
    RingOps.__init__ = _init
    R["T1_teeth_broken_ring"] = caught
    ok = caught != 0 and all(r["T1_ring_vs_int_mismatches"] == 0 and r["T2_shuffled_identical"] for r in R["runs"])
    acc = all(r["T3_softmax_max_rel_err"] < 1e-9 and r["T3_lse_max_abs_err"] < 1e-9 for r in R["runs"])
    R["verdict"] = "PASS" if ok and acc else ("PASS_EXACT_FAIL_ACCURACY" if ok else "FAIL")
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ring_nl.result.json")
    json.dump(R, open(out, "w"), indent=1); print(json.dumps(R, indent=1))
