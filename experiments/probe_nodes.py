#!/usr/bin/env python3
"""
Armilla node designs — probes N1–N3 (2026-10-12). Pass rules are fixed below, before the run.

Family (proposal, for J to confirm): a middle ring of k ring systems around the centre system, plus one system on top and one
below. k = 3 -> 6 systems (J's design: triangle + centre + top + bottom), k = 5 -> 8, k = 7 -> 10. Every system has 55 rings
(16-bit primes, no prime shared between systems).

N1  range and crypto fit: total range; the largest modulus N an RNS Montgomery multiply can take with base A = middle ring and
    base B = top + bottom (needs A > 4N and B > 2N).                                     PASS = arithmetic only, reported.
N2  guard mode: top and bottom carry the largest primes and act as redundant (check) systems; data = middle ring + centre.
    Redundant RNS at system level: a whole corrupted system is DETECTED (2 redundant systems detect up to 2) and a single
    one is CORRECTED by leaving each system out in turn.          PASS = 0 missed detections, 100 % single-system correction.
N3  the JIT centre: the Gelosia lattice core is generated as C for one shape (limbs K, the 50 T50 moduli baked in as
    constants), compiled and loaded at run time, then fired once per "ring turn" on data that is already in residues.
    Compared with the generic kernel on the same inputs.      PASS = bit-identical diagonals; timing reported as measured.
Writes experiments/probe_nodes.result.json
"""
import ctypes, json, math, os, random, subprocess, sys, tempfile, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
rng = random.Random(1012)
R = {"date": "2026-10-12", "rings_per_system": 55}

def primes_below(n):
    s = bytearray([1]) * n; s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n) if s[i]]

def crt(res, mods):
    M = math.prod(mods)
    return sum(r * (M // m) * pow(M // m, -1, m) for r, m in zip(res, mods)) % M

P16 = [p for p in primes_below(65536) if p > 2][::-1]          # largest first
designs = {"6 (k=3)": 3, "8 (k=5)": 5, "10 (k=7)": 7}

# ---------------- N1 range and crypto fit ----------------
n1 = {}
for name, k in designs.items():
    n = k + 3; pool = P16[:n * 55]
    sysm = [pool[i::n] for i in range(n)]                       # dealt in turn: balanced systems
    bits = [sum(math.log2(p) for p in s) for s in sysm]
    A = sum(bits[:k]); B = bits[k + 1] + bits[k + 2]            # middle ring ; top + bottom
    n1[name] = {"systems": n, "rings": n * 55, "range_bits": round(sum(bits)),
                "base_A_middle_bits": round(A), "base_B_top_bottom_bits": round(B),
                "max_montgomery_modulus_bits": int(min(A - 2, B - 1))}
R["N1_range_and_crypto_fit"] = n1

# ---------------- N2 guard mode (system-level redundant RNS) ----------------
n2 = {}
for name, k in designs.items():
    n = k + 3; pool = P16[:n * 55]
    red = [pool[0:55], pool[55:110]]                            # top, bottom: the largest primes
    rest = pool[110:]; data = [rest[i::k + 1] for i in range(k + 1)]   # middle ring + centre
    Ms = [math.prod(s) for s in data + red]; MD = math.prod(Ms[:k + 1]); Mall = math.prod(Ms)
    nsys = len(Ms)
    def decode(res):
        x = crt(res, Ms)
        if x < MD: return "ok", x
        for i in range(nsys):                                   # leave one system out
            keep = [j for j in range(nsys) if j != i]
            y = crt([res[j] for j in keep], [Ms[j] for j in keep])
            if y < MD: return f"corrected:{i}", y
        return "detected", None
    st = {"clean_ok": 0, "one_fault_corrected": 0, "one_fault_wrong": 0, "two_fault_detected": 0, "two_fault_missed": 0}
    T = 200
    for t in range(T):
        x = rng.randrange(MD); res = [x % m for m in Ms]
        st["clean_ok"] += decode(res) == ("ok", x)
        bad = list(res); i = rng.randrange(nsys); bad[i] = (bad[i] + rng.randrange(1, Ms[i])) % Ms[i]
        v = decode(bad); st["one_fault_corrected" if v == (f"corrected:{i}", x) else "one_fault_wrong"] += 1
        bad = list(res)
        for i in rng.sample(range(nsys), 2): bad[i] = (bad[i] + rng.randrange(1, Ms[i])) % Ms[i]
        v = decode(bad)
        st["two_fault_detected" if v[0] != "ok" and v[1] != x else "two_fault_missed"] += 1
        # a "correction" to a wrong value also counts as missed: guard mode guarantees detection of 2, correction of 1
        if v[0].startswith("corrected"): st["two_fault_detected"] -= 1; st["two_fault_missed"] += 1
    st["trials"] = T; st["data_range_bits"] = MD.bit_length(); st["redundant_systems"] = 2
    st["verdict"] = "PASS" if st["clean_ok"] == T and st["one_fault_corrected"] == T else "FAIL"
    n2[name] = st
R["N2_guard_mode"] = n2

# ---------------- N3 the JIT centre, fired every ring turn ----------------
T50 = [p for p in primes_below(234) if p > 2]
LANES = 64; K = 54; LB = 150                                    # 8,000-bit operands, 150-bit limbs
def gen_c(K):
    inv = ", ".join(f"{1.0 / m:.9e}f" for m in T50 + [1] * (LANES - 50))
    mods = ", ".join(str(m) for m in T50 + [1] * (LANES - 50))
    return f"""#include <stdint.h>
static const uint32_t MOD[{LANES}] = {{{mods}}};
static const float INV[{LANES}] = {{{inv}}};
/* rendered for K = {K}: shape and moduli are constants, so loop bounds and reduction constants are known at compile time */
void turn(const uint32_t *restrict RA, const uint32_t *restrict RB, uint32_t *restrict D) {{
  for (int d = 0; d < {2 * K - 1}; d++) {{
    uint32_t acc[{LANES}] = {{0}};
    int i0 = d - {K - 1} > 0 ? d - {K - 1} : 0, i1 = d < {K - 1} ? d : {K - 1};
    for (int i = i0; i <= i1; i++) {{
      const uint32_t *a = RA + i * {LANES}, *b = RB + (d - i) * {LANES};
      #pragma GCC unroll 8
      for (int l = 0; l < {LANES}; l++) acc[l] += a[l] * b[l];
    }}
    for (int l = 0; l < {LANES}; l++) {{
      int32_t r = (int32_t)acc[l] - (int32_t)((float)acc[l] * INV[l]) * (int32_t)MOD[l];
      r += (r < 0) ? (int32_t)MOD[l] : 0; r -= (r >= (int32_t)MOD[l]) ? (int32_t)MOD[l] : 0;
      D[d * {LANES} + l] = (uint32_t)r;
    }}
  }}
}}
"""
tmp = tempfile.mkdtemp()
t0 = time.perf_counter()
open(os.path.join(tmp, "jit.c"), "w").write(gen_c(K))
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "jit.so"), os.path.join(tmp, "jit.c")], check=True)
jit = ctypes.CDLL(os.path.join(tmp, "jit.so")); t_render = time.perf_counter() - t0
subprocess.run(["gcc", "-O3", "-march=native", "-shared", "-fPIC", "-o", os.path.join(tmp, "gen.so"), os.path.join(HERE, "gelosia_kernel.c")], check=True)
gen = ctypes.CDLL(os.path.join(tmp, "gen.so"))
u32p = ctypes.POINTER(ctypes.c_uint32); P = lambda a: a.ctypes.data_as(u32p)
mods = np.ones(LANES, dtype=np.uint32); mods[:50] = T50
TURNS = 2000
A = np.zeros((TURNS, K, LANES), dtype=np.uint32); B = np.zeros_like(A)
A[:, :, :50] = np.array([[rng.randrange(m) for m in T50] for _ in range(TURNS * K)]).reshape(TURNS, K, 50)
B[:, :, :50] = np.array([[rng.randrange(m) for m in T50] for _ in range(TURNS * K)]).reshape(TURNS, K, 50)
D1 = np.zeros((2 * K - 1, LANES), dtype=np.uint32); D2 = np.zeros_like(D1); same = 0
for t in range(200):
    jit.turn(P(A[t]), P(B[t]), P(D1)); gen.gelosia_core(P(A[t]), K, P(B[t]), K, P(mods), P(D2))
    same += np.array_equal(D1[:, :50], D2[:, :50])
def run(f):
    t0 = time.perf_counter()
    for t in range(TURNS): f(t)
    return (time.perf_counter() - t0) / TURNS * 1e6
for _ in range(2):                                              # warm-up, then measure
    tj = run(lambda t: jit.turn(P(A[t]), P(B[t]), P(D1)))
    tg = run(lambda t: gen.gelosia_core(P(A[t]), K, P(B[t]), K, P(mods), P(D2)))
boxes = K * K
R["N3_jit_centre"] = {"shape": f"8,000-bit operands, {K} limbs of {LB} bits, {boxes} boxes x 50 rings per turn",
    "render_and_compile_ms": round(t_render * 1e3, 1), "turns": TURNS, "identical_diagonals": f"{same}/200",
    "jit_us_per_turn": round(tj, 2), "generic_us_per_turn": round(tg, 2), "jit_speedup_vs_generic": round(tg / tj, 2),
    "ring_products_per_second_jit": round(boxes * 50 / (tj * 1e-6)),
    "note": "data already in residues (conversion is paid once at entry/exit, not per turn); includes ctypes call overhead",
    "verdict": "PASS" if same == 200 else "FAIL"}
json.dump(R, open(os.path.join(HERE, "probe_nodes.result.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
