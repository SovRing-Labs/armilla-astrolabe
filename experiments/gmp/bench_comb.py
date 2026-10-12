#!/usr/bin/env python3
"""Fixed-base comb ("Napier's bones" for exponents) on the step-3 RNS-IFMA kernel vs GMP powmod / powmod_sec (2026-10-12).
One fixed base g, many exponents (signing / key generation / Diffie-Hellman). GMP has no fixed-base precompute API, so the GMP
columns are what a GMP user gets. Pass rules fixed before the run:
CB1 exactness: g^e mod N equals gmpy2.powmod for every message.   CB2 ms per exponentiation, interleaved medians, 1 and 4 cores.
Writes bench_comb.result.json"""
import ctypes, json, math, os, random, subprocess, tempfile, time
from multiprocessing import Pool
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp(); so = os.path.join(tmp, "c.so")
subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "rns_ifma_batch.c"), "-lm"], check=True)
lib = ctypes.CDLL(so); vp = ctypes.c_void_p; P = lambda a: a.ctypes.data_as(vp)
rng = random.Random(1617)
PR = []; cc = 1
while len(PR) < 300:
    if gmpy2.is_prime((1 << 48) - cc): PR.append((1 << 48) - cc)
    cc += 2
def setup(N, bits):
    n = ((bits + 6 + 47) // 48 + 7) // 8 * 8
    A, B = PR[0:2 * n:2], PR[1:2 * n:2]; MA, MB = math.prod(A), math.prod(B)
    u = lambda v: np.array(v, dtype=np.uint64)
    return dict(n=n, A=A, B=B, MA=MA, MB=MB, a=u(A), b=u(B), ca=u([(1 << 48) - m for m in A]), cb=u([(1 << 48) - m for m in B]),
        ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]), TAB=u([[(MA // a) % b for b in B] for a in A]).ravel(),
        MAb=u([MA % b for b in B]), Nb=u([N % b for b in B]),
        dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=u([[(MB // b) % a for a in A] for b in B]).ravel(), MBa=u([MB % a for a in A]))
KEYS = ("a", "b", "ca", "cb", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa")
def gmp_job(a):
    g, e, N, sec = a; return int((gmpy2.powmod_sec if sec else gmpy2.powmod)(gmpy2.mpz(g), gmpy2.mpz(e), gmpy2.mpz(N)))
res = {"date": "2026-10-12", "method": "fixed-base comb, 5-bit windows, constant-time table scan, batches of 8, RNS-IFMA step 3"}
pool = Pool(4)
for bits in (2048, 4096):
    N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1))); C = setup(N, bits); n = C["n"]; S = 2 * n
    g = rng.randrange(2, N); nwin = (bits + 4) // 5; R = C["MA"] % N
    t0 = time.perf_counter()
    Ag, Bg = np.array(C["A"], dtype=object), np.array(C["B"], dtype=object)
    tab = np.zeros((nwin, 32, S), np.uint64); base = gmpy2.mpz(g)
    for k in range(nwin):
        v = gmpy2.mpz(R)                                                      # j = 0 -> R (Montgomery one)
        for j in range(32):
            iv = int(v); tab[k, j, :n] = [iv % m for m in C["A"]]; tab[k, j, n:] = [iv % m for m in C["B"]]
            v = v * base % N
        base = gmpy2.powmod(base, 32, N)
    t_pre = time.perf_counter() - t0
    msgs = 64; es = [rng.getrandbits(bits) | 1 << (bits - 1) for _ in range(msgs)]
    Wn = np.array([[(e >> (5 * k)) & 31 for k in range(nwin)] for e in es], np.uint8)
    def run(m, th):
        out = np.zeros((m, S), np.uint64); s = time.perf_counter()
        lib.batch_fixedbase_ct(n, *[P(C[k]) for k in KEYS], m // 8, P(tab), P(np.ascontiguousarray(Wn[:m])), nwin, P(out), th)
        return out, time.perf_counter() - s
    out, _ = run(msgs, 4)
    ones = np.array([[1 % m for m in C["A"]] + [1 % m for m in C["B"]]] * 8, np.uint64); got = []
    for bt in range(msgs // 8):
        r = np.zeros((8, S), np.uint64)
        lib.batch_mont(n, *[P(C[k]) for k in KEYS], P(np.ascontiguousarray(out[bt * 8:(bt + 1) * 8])), P(ones), P(r))
        got += [sum(int(v) * (C["MB"] // b) * pow(C["MB"] // b, -1, b) for v, b in zip(row[n:], C["B"])) % C["MB"] % N for row in r]
    bad = sum(x != int(gmpy2.powmod(g, e, N)) for x, e in zip(got, es))
    T = {k: [] for k in ("comb1", "gmp1", "sec1", "comb4", "gmp4", "sec4")}
    for _ in range(5):
        T["comb1"].append(run(16, 1)[1] / 16)
        s = time.perf_counter(); [gmp_job((g, e, N, False)) for e in es[:16]]; T["gmp1"].append((time.perf_counter() - s) / 16)
        s = time.perf_counter(); [gmp_job((g, e, N, True)) for e in es[:16]]; T["sec1"].append((time.perf_counter() - s) / 16)
        T["comb4"].append(run(msgs, 4)[1] / msgs)
        s = time.perf_counter(); pool.map(gmp_job, [(g, e, N, False) for e in es], chunksize=8); T["gmp4"].append((time.perf_counter() - s) / msgs)
        s = time.perf_counter(); pool.map(gmp_job, [(g, e, N, True) for e in es], chunksize=8); T["sec4"].append((time.perf_counter() - s) / msgs)
    md = {k: round(float(np.median(v)) * 1e3, 3) for k, v in T.items()}
    res[str(bits)] = {"messages_checked": msgs, "mismatches": bad, "table_MB": round(tab.nbytes / 1e6, 1), "table_build_s_python": round(t_pre, 1),
        "ms_per_modexp": md, "1core_vs_powmod": round(md["gmp1"] / md["comb1"], 2), "1core_vs_powmod_sec": round(md["sec1"] / md["comb1"], 2),
        "4core_vs_powmod": round(md["gmp4"] / md["comb4"], 2), "4core_vs_powmod_sec": round(md["sec4"] / md["comb4"], 2)}
    print(bits, json.dumps(res[str(bits)]), flush=True)
json.dump(res, open(os.path.join(HERE, "bench_comb.result.json"), "w"), indent=1)
