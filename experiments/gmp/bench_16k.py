#!/usr/bin/env python3
"""
Item 3 (2026-10-12): the 16,384-bit limit. Pass rules fixed before the run.
L1 step 3 (with folds every 240 terms) at 16,384 bits: exact vs gmpy2.powmod (8 messages); ms per exponentiation vs GMP powmod /
   powmod_sec and vs step 4, one core, 3 interleaved repetitions.
L2 fixed-base comb at 16,384 bits: exact (8 messages); ms per exponentiation.
L3 fast Chinese remaindering (subproduct tree, Borodin-Moenck 1974) as a base extension: time to take n residues to the other base
   through a product tree + remainder tree on GMP integers, against the measured O(n^2) IFMA extension cost, at 16,384 and 32,768
   bits, to locate the crossover. Reported.
Writes bench_16k.result.json
"""
import ctypes, json, math, os, random, subprocess, tempfile, time
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
def build(src):
    so = os.path.join(tmp, src + ".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", so, os.path.join(HERE, src), "-lm"], check=True)
    return ctypes.CDLL(so)
L3, L4 = build("rns_ifma_batch.c"), build("rns_ifma_wino.c"); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
rng = random.Random(16384)
PR = []; cc = 1
while len(PR) < 1400:
    if gmpy2.is_prime((1 << 48) - cc): PR.append((1 << 48) - cc)
    cc += 2
def setup(N, bits):
    n = ((bits + 6 + 47) // 48 + 7) // 8 * 8
    A, B = PR[0:2 * n:2], PR[1:2 * n:2]; MA, MB = math.prod(A), math.prod(B)
    u = lambda v: np.array(v, dtype=np.uint64)
    TAB = [[(MA // a) % b for b in B] for a in A]; TBA = [[(MB // b) % a for a in A] for b in B]
    return dict(n=n, A=A, B=B, MA=MA, MB=MB, a=u(A), b=u(B), ca=u([(1 << 48) - m for m in A]), cb=u([(1 << 48) - m for m in B]),
        ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]), TAB=u(TAB).ravel(), MAb=u([MA % b for b in B]),
        Nb=u([N % b for b in B]), dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=u(TBA).ravel(), MBa=u([MB % a for a in A]),
        CTAB=u([sum(TAB[2 * k][j] * TAB[2 * k + 1][j] for k in range(n // 2)) % B[j] for j in range(n)]),
        CTBA=u([sum(TBA[2 * k][i] * TBA[2 * k + 1][i] for k in range(n // 2)) % A[i] for i in range(n)]))
K3 = ("a", "b", "ca", "cb", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa"); K4 = K3 + ("CTAB", "CTBA")
def rns(C, v): return [v % m for m in C["A"]] + [v % m for m in C["B"]]
def finish(lib, keys, C, N, out):
    ones = np.array([rns(C, 1)] * 8, np.uint64); n = C["n"]; MB = C["MB"]; r = np.zeros((8, 2 * n), np.uint64)
    lib.batch_mont(n, *[P(C[k]) for k in keys], P(np.ascontiguousarray(out)), P(ones), P(r))
    return [sum(int(v) * (MB // b) * pow(MB // b, -1, b) for v, b in zip(row[n:], C["B"])) % MB % N for row in r]
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 IFMA, one core"}
bits = 16384; N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1))); C = setup(N, bits); n = C["n"]; S = 2 * n
R = C["MA"] % N; nwin = (bits + 4) // 5; msgs = 8
xs = [rng.randrange(N) for _ in range(msgs)]; es = [rng.getrandbits(bits) | 1 << (bits - 1) for _ in range(msgs)]
X = np.array([rns(C, x * R % N) for x in xs], np.uint64); O = np.array([rns(C, R)] * msgs, np.uint64)
W = np.array([[(e >> (5 * (nwin - 1 - k))) & 31 for k in range(nwin)] for e in es], np.uint8)
want = [int(gmpy2.powmod(x, e, N)) for x, e in zip(xs, es)]
def run(lib, keys):
    out = np.zeros_like(X); s = time.perf_counter()
    lib.batch_modexp_ct(n, *[P(C[k]) for k in keys], 1, P(X), P(O), P(W), nwin, P(out), 1)
    return out, (time.perf_counter() - s) / msgs
o3, _ = run(L3, K3); o4, _ = run(L4, K4)
bad3 = sum(a != b for a, b in zip(finish(L3, K3, C, N, o3), want)); bad4 = sum(a != b for a, b in zip(finish(L4, K4, C, N, o4), want))
T = {"step3": [], "step4": [], "gmp": [], "gmp_sec": []}
for _ in range(3):
    T["step3"].append(run(L3, K3)[1]); T["step4"].append(run(L4, K4)[1])
    s = time.perf_counter(); [gmpy2.powmod(gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)) for x, e in zip(xs, es)]; T["gmp"].append((time.perf_counter() - s) / msgs)
    s = time.perf_counter(); [gmpy2.powmod_sec(gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)) for x, e in zip(xs, es)]; T["gmp_sec"].append((time.perf_counter() - s) / msgs)
md = {k: round(float(np.median(v)) * 1e3, 1) for k, v in T.items()}
res["L1_16384"] = {"moduli_per_base": n, "mismatches_step3": bad3, "mismatches_step4": bad4, "ms_per_modexp": md,
                   "step3_vs_gmp_powmod": round(md["gmp"] / md["step3"], 2), "step3_vs_gmp_powmod_sec": round(md["gmp_sec"] / md["step3"], 2)}
print("L1", res["L1_16384"], flush=True)
# L2 comb
g = rng.randrange(2, N); tab = np.zeros((nwin, 32, S), np.uint64); base = gmpy2.mpz(g); t0 = time.perf_counter()
for k in range(nwin):
    v = gmpy2.mpz(R)
    for j in range(32):
        iv = int(v); tab[k, j, :n] = [iv % m for m in C["A"]]; tab[k, j, n:] = [iv % m for m in C["B"]]; v = v * base % N
    base = gmpy2.powmod(base, 32, N)
t_tab = time.perf_counter() - t0
Wl = np.array([[(e >> (5 * k)) & 31 for k in range(nwin)] for e in es], np.uint8)
out = np.zeros_like(X); s = time.perf_counter()
L3.batch_fixedbase_ct(n, *[P(C[k]) for k in K3], 1, P(tab), P(Wl), nwin, P(out), 1); tc = (time.perf_counter() - s) / msgs
badc = sum(a != int(gmpy2.powmod(g, e, N)) for a, e in zip(finish(L3, K3, C, N, out), es))
res["L2_comb_16384"] = {"mismatches": badc, "table_MB": round(tab.nbytes / 1e6, 1), "table_build_s": round(t_tab, 1),
                        "ms_per_modexp": round(tc * 1e3, 1), "vs_gmp_powmod": round(md["gmp"] / (tc * 1e3), 2), "vs_gmp_powmod_sec": round(md["gmp_sec"] / (tc * 1e3), 2)}
print("L2", res["L2_comb_16384"], flush=True)
del tab
# L3 fast CRT tree vs O(n^2) extension
def tree_ext(res_a, A, B):
    # product tree over A, CRT combine bottom-up, then remainder tree over B
    lv = [(gmpy2.mpz(r), gmpy2.mpz(a)) for r, a in zip(res_a, A)]
    inv = [gmpy2.invert(gmpy2.mpz(math.prod(A) // a) % a, a) for a in A] if False else None
    while len(lv) > 1:
        nx = []
        for i in range(0, len(lv) - 1, 2):
            (x1, m1), (x2, m2) = lv[i], lv[i + 1]
            t = ((x2 - x1) * gmpy2.invert(m1, m2)) % m2; nx.append((x1 + m1 * t, m1 * m2))
        if len(lv) % 2: nx.append(lv[-1])
        lv = nx
    Xv = lv[0][0]
    # remainder tree over B
    prods = [[gmpy2.mpz(b) for b in B]]
    while len(prods[-1]) > 1:
        p = prods[-1]; prods.append([p[i] * p[i + 1] if i + 1 < len(p) else p[i] for i in range(0, len(p), 2)])
    rem = [Xv % prods[-1][0]]
    for level in reversed(prods[:-1]):
        rem = [rem[i // 2] % level[i] for i in range(len(level))]
    return [int(r) for r in rem]
l3 = {}
for b2 in (16384, 32768):
    nn = ((b2 + 6 + 47) // 48 + 7) // 8 * 8; A, B = PR[0:2 * nn:2], PR[1:2 * nn:2]
    v = rng.getrandbits(b2); ra = [v % a for a in A]
    got = tree_ext(ra, A, B); ok = got == [v % b for b in B]
    ts = []
    for _ in range(5):
        s = time.perf_counter(); tree_ext(ra, A, B); ts.append(time.perf_counter() - s)
    l3[str(b2)] = {"moduli": nn, "exact": ok, "tree_extension_us_python_gmp": round(float(np.median(ts)) * 1e6, 1)}
# measured O(n^2) extension cost at 16,384: from L1 (two extensions + small work per Montgomery multiply)
mults = 16384 + nwin + 31
l3["ifma_extension_us_est_16384"] = round(md["step3"] * 1e3 / mults / 2, 1)
res["L3_fast_crt_tree"] = l3; print("L3", l3, flush=True)
json.dump(res, open(os.path.join(HERE, "bench_16k.result.json"), "w"), indent=1)
