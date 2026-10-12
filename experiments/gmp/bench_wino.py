#!/usr/bin/env python3
"""
Armilla vs GMP, step 4 (2026-10-12): Winograd's inner product (1968) inside both base extensions, against step 3 and GMP,
from 2,048 to 16,384 bits. One core, batches of 8, constant-time windows. Pass rules fixed before the run.
W1 exactness: every message equals gmpy2.powmod (step 3 and step 4).                                       PASS = 0 mismatches
W2 speed: ms per exponentiation, interleaved medians: step 4 vs step 3 vs GMP powmod vs GMP powmod_sec.   Reported.
Writes bench_wino.result.json
"""
import ctypes, json, math, os, random, subprocess, tempfile, time
import numpy as np, gmpy2
HERE = os.path.dirname(os.path.abspath(__file__)); tmp = tempfile.mkdtemp()
def build(src):
    so = os.path.join(tmp, src + ".so")
    subprocess.run(["gcc", "-O3", "-march=native", "-fopenmp", "-shared", "-fPIC", "-o", so, os.path.join(HERE, src), "-lm"], check=True)
    return ctypes.CDLL(so)
L3, L4 = build("rns_ifma_batch.c"), build("rns_ifma_wino.c"); vp = ctypes.c_void_p
def P(a): return a.ctypes.data_as(vp)
rng = random.Random(1968)
PR = []; cc = 1
while len(PR) < 720:
    if gmpy2.is_prime((1 << 48) - cc): PR.append((1 << 48) - cc)
    cc += 2
def setup(N, bits):
    n = ((bits + 6 + 47) // 48 + 7) // 8 * 8
    A, B = PR[0:2 * n:2], PR[1:2 * n:2]; MA, MB = math.prod(A), math.prod(B); assert MA > 32 * N and MB > 32 * N
    u = lambda v: np.array(v, dtype=np.uint64)
    TAB = [[(MA // a) % b for b in B] for a in A]; TBA = [[(MB // b) % a for a in A] for b in B]
    return dict(n=n, A=A, B=B, MA=MA, MB=MB, a=u(A), b=u(B), ca=u([(1 << 48) - m for m in A]), cb=u([(1 << 48) - m for m in B]),
        ia=np.array([1.0 / m for m in A]), ib=np.array([1.0 / m for m in B]),
        cA=u([(-pow(N, -1, a)) * pow(MA // a, -1, a) % a for a in A]), TAB=u(TAB).ravel(),
        MAb=u([MA % b for b in B]), Nb=u([N % b for b in B]),
        dB=u([pow(MA, -1, b) * pow(MB // b, -1, b) % b for b in B] + [(MB // b) % b for b in B]),
        TBA=u(TBA).ravel(), MBa=u([MB % a for a in A]),
        CTAB=u([sum(TAB[2 * k][j] * TAB[2 * k + 1][j] for k in range(n // 2)) % B[j] for j in range(n)]),
        CTBA=u([sum(TBA[2 * k][i] * TBA[2 * k + 1][i] for k in range(n // 2)) % A[i] for i in range(n)]))
K3 = ("a", "b", "ca", "cb", "ia", "ib", "cA", "TAB", "MAb", "Nb", "dB", "TBA", "MBa"); K4 = K3 + ("CTAB", "CTBA")
def rns(C, v): return [v % m for m in C["A"]] + [v % m for m in C["B"]]
def run(lib, keys, C, X, O, W, nwin):
    out = np.zeros_like(X); t0 = time.perf_counter()
    lib.batch_modexp_ct(C["n"], *[P(C[k]) for k in keys], len(X) // 8, P(X), P(O), P(W), nwin, P(out), 1)
    return out, time.perf_counter() - t0
def finish(lib, keys, C, N, out):
    ones = np.array([rns(C, 1)] * 8, np.uint64); MB = C["MB"]; n = C["n"]; res = []
    for bt in range(len(out) // 8):
        r = np.zeros((8, 2 * n), np.uint64)
        lib.batch_mont(C["n"], *[P(C[k]) for k in keys], P(np.ascontiguousarray(out[bt * 8:(bt + 1) * 8])), P(ones), P(r))
        res += [sum(int(v) * (MB // b) * pow(MB // b, -1, b) for v, b in zip(row[n:], C["B"])) % MB % N for row in r]
    return res
res = {"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, AVX-512 IFMA, one core", "gmp": gmpy2.mp_version()}
for bits, msgs, reps, do3 in ((2048, 32, 5, True), (4096, 32, 5, True), (8192, 16, 3, True), (16384, 8, 3, False)):
    N = int(gmpy2.mpz(rng.getrandbits(bits)) | 1 | (1 << (bits - 1))); C = setup(N, bits); R = C["MA"] % N; nwin = (bits + 4) // 5
    xs = [rng.randrange(N) for _ in range(msgs)]; es = [rng.getrandbits(bits) | 1 << (bits - 1) for _ in range(msgs)]
    X = np.array([rns(C, x * R % N) for x in xs], np.uint64); O = np.array([rns(C, R)] * msgs, np.uint64)
    W = np.array([[(e >> (5 * (nwin - 1 - k))) & 31 for k in range(nwin)] for e in es], np.uint8)
    want = [int(gmpy2.powmod(x, e, N)) for x, e in zip(xs, es)]
    bad4 = sum(g != w for g, w in zip(finish(L4, K4, C, N, run(L4, K4, C, X, O, W, nwin)[0]), want))
    bad3 = sum(g != w for g, w in zip(finish(L3, K3, C, N, run(L3, K3, C, X, O, W, nwin)[0]), want)) if do3 else None
    t = {"step4": [], "step3": [], "gmp": [], "gmp_sec": []}
    for _ in range(reps):
        t["step4"].append(run(L4, K4, C, X, O, W, nwin)[1] / msgs)
        if do3: t["step3"].append(run(L3, K3, C, X, O, W, nwin)[1] / msgs)
        t0 = time.perf_counter(); [gmpy2.powmod(gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)) for x, e in zip(xs, es)]; t["gmp"].append((time.perf_counter() - t0) / msgs)
        t0 = time.perf_counter(); [gmpy2.powmod_sec(gmpy2.mpz(x), gmpy2.mpz(e), gmpy2.mpz(N)) for x, e in zip(xs, es)]; t["gmp_sec"].append((time.perf_counter() - t0) / msgs)
    md = {k: round(float(np.median(v)) * 1e3, 3) for k, v in t.items() if v}
    row = {"moduli_per_base": C["n"], "messages": msgs, "mismatches_step4": bad4, "mismatches_step3": bad3, "ms_per_modexp": md,
           "step4_vs_step3": round(md["step3"] / md["step4"], 2) if do3 else None,
           "step4_vs_gmp_powmod": round(md["gmp"] / md["step4"], 2), "step4_vs_gmp_powmod_sec": round(md["gmp_sec"] / md["step4"], 2)}
    res[str(bits)] = row; print(bits, json.dumps(row), flush=True)
res["verdict"] = "PASS (exact)" if all(v["mismatches_step4"] == 0 and v["mismatches_step3"] in (0, None) for k, v in res.items() if k.isdigit()) else "FAIL"
json.dump(res, open(os.path.join(HERE, "bench_wino.result.json"), "w"), indent=1); print(res["verdict"])
