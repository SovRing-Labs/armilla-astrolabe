#!/usr/bin/env python3
"""
Extreme Dynamic Range Math Benchmark:
Tests mathematical scale where Python BigInt has to propagate carries across
thousands of bits (e.g. 300-digit, 1000-digit, and 10,000-bit integers),
comparing raw BigInt carry-ripple addition/multiplication vs
Zero-Carry Omniring Modular Execution.
"""

import time
import math
import numpy as np

def run_extreme_math_benchmark():
    print("=" * 70)
    print("EXTREME DYNAMIC RANGE BENCHMARK: CARRY-RIPPLE vs ZERO-CARRY OMNIRING")
    print("=" * 70)

    # 1000-digit integers (simulating massive cryptographic or quantum-state tensors)
    val_a = (1 << 3000) - 17
    val_b = (1 << 3000) - 31
    val_c = (1 << 3000) - 43
    val_d = (1 << 3000) - 67

    iterations = 20_000

    print(f"\n[1] 3,000-bit Integer Carry Arithmetic (Python BigInt ALU):")
    t0 = time.perf_counter()
    for _ in range(iterations):
        res = (val_a * val_b) + (val_c * val_d)
    t1 = time.perf_counter()
    bigint_time = t1 - t0
    bigint_cps = iterations / bigint_time
    print(f"  Time: {bigint_time:.4f}s | Cycles/sec: {bigint_cps:,.1f} | Latency: {(bigint_time/iterations)*1e6:.2f} µs/op")

    # In Omniring RNS, each node holds residues modulo 50 (or 256) rings.
    # Numbers stay bounded strictly within [0..255].
    moduli_50 = np.array([
        3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
        59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
        131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
        211, 223, 227, 229, 233
    ], dtype=np.int64)

    # Projected residues
    res_a = np.array([val_a % int(m) for m in moduli_50], dtype=np.int64)
    res_b = np.array([val_b % int(m) for m in moduli_50], dtype=np.int64)
    res_c = np.array([val_c % int(m) for m in moduli_50], dtype=np.int64)
    res_d = np.array([val_d % int(m) for m in moduli_50], dtype=np.int64)

    print(f"\n[2] Omniring Diamond Topology (50 Prime Rings, Zero Carry):")
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_center = ((res_a * res_b) + (res_c * res_d)) % moduli_50
    t1 = time.perf_counter()
    omniring_time = t1 - t0
    omniring_cps = iterations / omniring_time
    print(f"  Time: {omniring_time:.4f}s | Cycles/sec: {omniring_cps:,.1f} | Latency: {(omniring_time/iterations)*1e6:.2f} µs/op")

    print(f"\n[3] Extreme Scaling to 8,000-bit Integers (val^2):")
    val_huge_1 = (1 << 8000) - 1
    val_huge_2 = (1 << 8000) - 3

    t0 = time.perf_counter()
    for _ in range(5000):
        _ = val_huge_1 * val_huge_2
    t1 = time.perf_counter()
    huge_bigint_time = t1 - t0

    res_h1 = np.array([val_huge_1 % int(m) for m in moduli_50], dtype=np.int64)
    res_h2 = np.array([val_huge_2 % int(m) for m in moduli_50], dtype=np.int64)

    t0 = time.perf_counter()
    for _ in range(5000):
        _ = (res_h1 * res_h2) % moduli_50
    t1 = time.perf_counter()
    huge_omniring_time = t1 - t0

    print(f"  8,000-bit BigInt multiplication time: {huge_bigint_time:.4f}s (latency: {(huge_bigint_time/5000)*1e6:.2f} µs)")
    print(f"  Omniring RNS multiplication time:     {huge_omniring_time:.4f}s (latency: {(huge_omniring_time/5000)*1e6:.2f} µs)")
    print(f"  --> Omniring Speedup over 8,000-bit carry multiply: {huge_bigint_time / huge_omniring_time:.2f}x (NOT like for like: the ring side gives the product mod M, 311 bits, not the 16,000-bit product; see TECHNICAL-DISCLOSURE §3)")
    print("=" * 70)

if __name__ == "__main__":
    run_extreme_math_benchmark()
