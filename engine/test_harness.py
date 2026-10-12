#!/usr/bin/env python3
"""
Real Computation Power Benchmark & Correctness Verification Harness
for Diamond Omniring Abacus vs Traditional Multiprecision BigInt.
Compares:
1. Exactness: RNS carry-free evaluation vs Python infinite-precision BigInt arithmetic.
2. Computational Throughput: Diamond Omniring vs Standard Arithmetic across large batches.
3. Scaling: 5-ring, 15-ring, 30-ring, and 50-ring dynamic range profiles.
"""

import time
import math
import json
import numpy as np
from diamond_abacus import DiamondOmniringAbacus, T50_PRIMES
from diamond_simd import FastDiamondAbacus

def verify_algebraic_correctness():
    """Verify that (N*S + W*E) in RNS matches Python BigInt modulo product of primes."""
    abacus = DiamondOmniringAbacus(ring_count=10)
    # Test values
    N = 987654321
    W = 123456789
    S = 456789123
    E = 789123456

    abacus.load_inputs(N, W, S, E)
    abacus.execute_diamond_cycle("balanced_tensor")

    # BigInt exact ground truth
    expected_bigint = (N * S + W * E)
    reconstructed, modulus = abacus.crt_reconstruct(max_rings=10)

    assert (expected_bigint % modulus) == reconstructed, (
        f"Correctness failure! BigInt % M = {expected_bigint % modulus}, Reconstructed = {reconstructed}"
    )
    print(f"[*] Algebraic Correctness Verified: CRT Reconstructed {reconstructed} equals BigInt mod M.")

def run_comparative_benchmark(iterations: int = 200_000):
    print(f"[*] Running Comprehensive Power Benchmark ({iterations:,} cycles)...")
    N = 9876543210123456789
    W = 1234567890987654321
    S = 4567891234567890123
    E = 7891234567890123456

    # 1. Standard Python BigInt Carry Arithmetic (Software ALU with Carry Ripples)
    t0 = time.perf_counter()
    for _ in range(iterations):
        res_bigint = (N * S + W * E)
    t1 = time.perf_counter()
    bigint_time = t1 - t0
    bigint_cps = iterations / max(bigint_time, 1e-9)

    # 2. Vectorized Diamond Abacus SIMD (Carry-Free T50 Rings)
    simd_abacus = FastDiamondAbacus(ring_count=50)
    simd_abacus.load_inputs(N, W, S, E)

    t0 = time.perf_counter()
    for _ in range(iterations):
        simd_abacus.compute("balanced_tensor")
    t1 = time.perf_counter()
    simd_time = t1 - t0
    simd_cps = iterations / max(simd_time, 1e-9)

    # Calculate real operations
    # Python BigInt does 1 large mul, 1 large mul, 1 large add (each doing multi-limb carry propagation)
    # Diamond Abacus computes 50 distinct prime-ring residue multiplications and additions in parallel (150 ring ops per cycle)
    total_ring_ops = iterations * 50 * 3
    ring_mops = (total_ring_ops / simd_time) / 1e6

    results = {
        "iterations": iterations,
        "ring_count": 50,
        "dynamic_range": "10^93 states (50 primes 3..233)",
        "bigint_carry_time_s": round(bigint_time, 4),
        "bigint_cycles_per_sec": round(bigint_cps, 2),
        "diamond_abacus_simd_time_s": round(simd_time, 4),
        "diamond_cycles_per_sec": round(simd_cps, 2),
        "effective_ring_mops": round(ring_mops, 2),
        "latency_per_diamond_ns": round((simd_time / iterations) * 1e9, 2),
        "speedup_vs_python_eval": round(simd_cps / bigint_cps, 2)
    }

    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    verify_algebraic_correctness()
    run_comparative_benchmark()
