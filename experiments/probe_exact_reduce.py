#!/usr/bin/env python3
"""
Armilla — exact, order-independent gradient sums in rings (2026-10-12). The AI case for the abacus: training on many devices
adds the same gradients in different orders; float sums then differ in the last bits, so runs are not bit-reproducible.
In residues every ring sum is exact, so any order gives the same bits.

G1  sum 1,000,000 gradient values (float32, typical spread) in 8 different orders, as 8 harnesses would:
    float32 and float64 sums: how many distinct results; ring sum (values as fixed-point integers, scale 2^-60, held on the
    50 T50 rings in the centred range, read back by CRT): distinct results.        PASS = ring sum 1 distinct result, equal to
    the exact rational sum of the fixed-point values.
G2  wide exact dot products: int8 x int8 dot products of length 2^20 accumulate past int32 when every term is large; rings
    hold them exactly.                                                              PASS = ring result equals Python int.
Writes experiments/probe_exact_reduce.result.json
"""
import json, math, os, random, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = np.array([3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,
              151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233], dtype=np.int64)
M = math.prod(int(p) for p in P)
rng = np.random.default_rng(8)
R = {"date": "2026-10-12"}

# G1
n = 1_000_000
g = (rng.standard_normal(n) * np.exp(rng.uniform(-12, 2, n))).astype(np.float32)   # gradients spread over ~6 decades
SCALE = 60
fixed = [int(round(float(v) * 2 ** SCALE)) for v in g]                               # exact fixed-point copy
exact = sum(fixed)
# residues of all values, built once (entry to the rings)
t0 = time.perf_counter()
fx = np.array(fixed, dtype=object)
RES = np.stack([np.array([f % int(p) for f in fixed], dtype=np.int64) for p in P], axis=1)
t_entry = time.perf_counter() - t0
f32, f64, ring = set(), set(), set()
for k in range(8):
    order = rng.permutation(n)
    f32.add(float(np.add.reduce(g[order], dtype=np.float32)))
    f64.add(float(np.cumsum(g[order].astype(np.float64))[-1]))
    r = RES[order].sum(axis=0) % P                                                   # one ring sum per harness order
    ring.add(tuple(int(x) for x in r))
def crt(res):
    return sum(int(r) * (M // int(p)) * pow(M // int(p), -1, int(p)) for r, p in zip(res, P)) % M
ring_val = [crt(r) for r in ring]; ring_val = [v - M if 2 * v >= M else v for v in ring_val]
R["G1_order_independent_sum"] = {"values": n, "orders": 8, "float32_distinct_results": len(f32),
    "float64_distinct_results": len(f64), "ring_distinct_results": len(ring),
    "ring_equals_exact_fixed_point_sum": ring_val == [exact],
    "float32_spread": max(f32) - min(f32), "float64_spread": max(f64) - min(f64),
    "fixed_point_scale": f"2^-{SCALE}", "entry_seconds_python": round(t_entry, 1),
    "verdict": "PASS" if len(ring) == 1 and ring_val == [exact] else "FAIL"}

# G2
L = 1 << 20
x = rng.integers(100, 128, L, dtype=np.int64); y = rng.integers(100, 128, L, dtype=np.int64)
true = int(np.sum(x * y))
i32 = int(np.sum((x * y).astype(np.int32), dtype=np.int32))
ringdot = (x[:, None] % P * (y[:, None] % P) % P).sum(axis=0) % P
R["G2_wide_dot"] = {"length": L, "true_bits": true.bit_length(), "int32_accumulator_correct": i32 == true,
    "ring_correct": crt(ringdot) == true, "verdict": "PASS" if crt(ringdot) == true else "FAIL"}
json.dump(R, open(os.path.join(HERE, "probe_exact_reduce.result.json"), "w"), indent=1)
print(json.dumps(R, indent=1))
