#!/usr/bin/env python3
"""
Armilla vs GMP, step 3 (2026-10-12): batched, windowed, constant-time, multi-core RNS-IFMA modexp. Pass rules fixed before the run.
B1 exactness: every message (own base, own full-length exponent) equals gmpy2.powmod, at 2,048 and 4,096 bits. PASS = 0 mismatches
B2 speed per exponentiation, interleaved medians, against GMP 6.3 powmod (fast, not constant-time) and powmod_sec (constant-time):
   one core (RNS 4 batches x 8 vs GMP one by one) and four cores (RNS 16 batches x 8 on 4 threads vs GMP in 4 processes).
B3 constant time (dudect-style): one batch, exponents all-zero vs random, 300 interleaved samples; Welch t.   PASS = |t| < 4.5
Writes bench_batch_ct.result.json
"""
import ctypes, json, math, os, random, subprocess, tempfile, time
from multiprocessing import Pool
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
so = os.path.join(tmp, "b.so")
subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", so, os.path.join(HERE, "rns_ifma_batch.c"), "-lm"], check=True)
lib = ctypes.CDLL(so); vp = ctypes.c_void_p
def P(a): return a.ctypes.data_as(vp)
rng = random.Random(4848)
PR = []; cc = 1
while len(PR) < 300:
    if gmpy2.is_prime((1 << 48) - cc): PR.append((1 << 48) - cc)
    cc += 2
def setup(N, bits):
    n = ((bits + 6 + 47) // 48 + 7) // 8 * 8
    A, B = PR[0:2 * n:2], PR[1:2 * n:2]; MA, MB = math.prod(A), math.prod(B); assert MA > 32 * N and MB > 32 * N
    u = lambda v: np.array(v, dtype=np.uint64)
    return dict(n=n, A=A, B=B, MA=MA, MB=MB, a=u(A), b=u(B), ca=u([(1 << 48) - m for m in A]), cb=u([(1 << 48) - m for m in B]),
        ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]), TAB=u([[(MA // a) % b for b in B] for a in A]).ravel(),
        MAb=u([MA % b for b in B]), Nb=u([N % b for b in B]),
        dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=u([[(MB // b) % a for a in A] for b in B]).ravel(), MBa=u([MB % a for a in A]))
KEYS = ("a", "b", "ca", "cb", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa")
def rns(C, v): return [v % m for m in C["A"]] + [v % m for m in C["B"]]
def prep(C, N, xs, es, bits):
    R = C["MA"] % N; nwin = (bits + 4) // 5
    X = np.array([rns(C, x * R % N) for x in xs], np.uint64); O = np.array([rns(C, R)] * len(xs), np.uint64)
    W = np.array([[(e >> (5 * (nwin - 1 - k))) & 31 for k in range(nwin)] for e in es], np.uint8)
    return X, O, W, nwin
def run(C, X, O, W, nwin, threads):
    nb = len(X) // 8; out = np.zeros_like(X)
    t0 = time.perf_counter()
    lib.batch_modexp_ct(C["n"], *[P(C[k]) for k in KEYS], nb, P(X), P(O), P(W), nwin, P(out), threads)
    return out, time.perf_counter() - t0
def finish(C, N, out):
    ones = np.array([rns(C, 1)] * 8, np.uint64); res = []; MB = C["MB"]; n = C["n"]
    for bt in range(len(out) // 8):
        r = np.zeros((8, 2 * n), np.uint64)
        lib.batch_mont(C["n"], *[P(C[k]) for k in KEYS], P(np.ascontiguousarray(out[bt * 8:(bt + 1) * 8])), P(ones), P(r))
        for row in r:
            res.append(sum(int(v) * (MB // b) * pow(MB // b, -1, b) for v, b in zip(row[n:], C["B"])) % MB % N)
    return res
def gmp_job(args):
    x, e, N, sec = args; f = gmpy2.powmod_sec if sec else gmpy2.powmod
    return int(f(gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)))
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 IFMA", "gmp": gmpy2.mp_version(),
       "method": "RNS-IFMA, batches of 8, fixed 5-bit windows, constant-time table scan, OpenMP over batches"}
pool = Pool(4)
for bits in (2048, 4096):
    N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1))); C = setup(N, bits)
    M1, M4 = 32, 128
    xs = [rng.randrange(N) for _ in range(M4)]; es = [rng.getrandbits(bits) | 1 << (bits - 1) for _ in range(M4)]
    X, O, W, nwin = prep(C, N, xs, es, bits)
    out, _ = run(C, X, O, W, nwin, 4); got = finish(C, N, out)
    bad = sum(g != int(gmpy2.powmod(x, e, N)) for g, x, e in zip(got, xs, es))
    t = {k: [] for k in ("rns1", "gmp1", "gsec1", "rns4", "gmp4", "gsec4")}
    for _ in range(5):
        _, tt = run(C, X[:M1], O[:M1], W[:M1], nwin, 1); t["rns1"].append(tt / M1)
        t0 = time.perf_counter(); [gmp_job((x, e, N, False)) for x, e in zip(xs[:M1], es[:M1])]; t["gmp1"].append((time.perf_counter() - t0) / M1)
        t0 = time.perf_counter(); [gmp_job((x, e, N, True)) for x, e in zip(xs[:M1], es[:M1])]; t["gsec1"].append((time.perf_counter() - t0) / M1)
        _, tt = run(C, X, O, W, nwin, 4); t["rns4"].append(tt / M4)
        t0 = time.perf_counter(); pool.map(gmp_job, [(x, e, N, False) for x, e in zip(xs, es)], chunksize=8); t["gmp4"].append((time.perf_counter() - t0) / M4)
        t0 = time.perf_counter(); pool.map(gmp_job, [(x, e, N, True) for x, e in zip(xs, es)], chunksize=8); t["gsec4"].append((time.perf_counter() - t0) / M4)
    md = {k: float(np.median(v)) * 1e3 for k, v in t.items()}
    row = {"moduli_per_base": C["n"], "messages_checked": M4, "mismatches": bad, "ms_per_modexp": {k: round(v, 3) for k, v in md.items()},
           "one_core_vs_gmp_powmod": round(md["gmp1"] / md["rns1"], 2), "one_core_vs_gmp_powmod_sec": round(md["gsec1"] / md["rns1"], 2),
           "four_cores_vs_gmp_powmod": round(md["gmp4"] / md["rns4"], 2), "four_cores_vs_gmp_powmod_sec": round(md["gsec4"] / md["rns4"], 2)}
    # B3 constant time: one batch, all-zero exponents vs random exponents
    Z = np.zeros_like(W[:8]); Rw = W[:8]
    s0, s1 = [], []
    for _ in range(300):
        for cls, ww in ((0, Z), (1, Rw)):
            o = np.zeros_like(X[:8]); t0 = time.perf_counter()
            lib.batch_modexp_ct(C["n"], *[P(C[k]) for k in KEYS], 1, P(X[:8]), P(O[:8]), P(ww), nwin, P(o), 1)
            (s0 if cls == 0 else s1).append(time.perf_counter() - t0)
    a, b = np.array(s0), np.array(s1)
    tw = float((a.mean() - b.mean()) / math.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    row["B3_welch_t_zero_vs_random_exponent"] = round(tw, 2); row["B3_means_ms"] = [round(a.mean() * 1e3, 3), round(b.mean() * 1e3, 3)]
    res[str(bits)] = row; print(bits, json.dumps(row), flush=True)
res["verdict"] = "PASS" if all(res[k]["mismatches"] == 0 and abs(res[k]["B3_welch_t_zero_vs_random_exponent"]) < 4.5 for k in ("2048", "4096")) else "CHECK"
json.dump(res, open(os.path.join(HERE, "bench_batch_ct.result.json"), "w"), indent=1); print(res["verdict"])
