"""
High-Performance Vectorized SIMD Omniring Diamond Abacus using NumPy INT64/INT32 arrays.
Simulates parallel AVX2/AVX-512 register operations on 5-node diamond topology.
"""

import time
import numpy as np
from typing import Dict, Any

T50_PRIMES = np.array([
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233
], dtype=np.int64)

class FastDiamondAbacus:
    def __init__(self, ring_count: int = 50):
        self.moduli = T50_PRIMES[:ring_count].copy()
        self.ring_count = len(self.moduli)
        # 4 satellite registers + 1 center accumulator: shape (5, ring_count)
        # index 0: North, 1: West, 2: South, 3: East, 4: Center
        self.state = np.zeros((5, self.ring_count), dtype=np.int64)

    def load_inputs(self, n_val: int, w_val: int, s_val: int, e_val: int):
        vals = [n_val, w_val, s_val, e_val]
        for idx, val in enumerate(vals):
            # Safe projection even for arbitrary Python BigInts
            res = [val % int(m) for m in self.moduli]
            self.state[idx] = np.array(res, dtype=np.int64)

    def compute(self, formula: str = "balanced_tensor") -> np.ndarray:
        # 0:N, 1:W, 2:S, 3:E
        N = self.state[0]
        W = self.state[1]
        S = self.state[2]
        E = self.state[3]
        M = self.moduli

        if formula == "balanced_tensor":
            # (N * S + W * E) % M
            self.state[4] = (N * S + W * E) % M
        elif formula == "superposition_sum":
            # (N + W + S + E) % M
            self.state[4] = (N + W + S + E) % M
        elif formula == "cyclic_interlock":
            # ((N + S) * (W + E)) % M
            self.state[4] = (((N + S) % M) * ((W + E) % M)) % M
        else:
            raise ValueError(f"Unknown formula: {formula}")
        return self.state[4]

def benchmark_fast_diamond(num_batches: int = 500_000, ring_count: int = 50) -> Dict[str, Any]:
    abacus = FastDiamondAbacus(ring_count=ring_count)
    abacus.load_inputs(987654321, 123456789, 555555555, 222222222)

    t0 = time.perf_counter()
    for _ in range(num_batches):
        abacus.compute("balanced_tensor")
    t1 = time.perf_counter()

    elapsed = t1 - t0
    cycles_per_sec = num_batches / max(elapsed, 1e-9)
    ops_per_sec = cycles_per_sec * ring_count * 3

    return {
        "iterations": num_batches,
        "elapsed_seconds": round(elapsed, 4),
        "diamond_cycles_per_sec": round(cycles_per_sec, 2),
        "ring_ops_per_sec": round(ops_per_sec, 2),
        "effective_mops": round(ops_per_sec / 1e6, 2),
        "latency_per_diamond_ns": round((elapsed / num_batches) * 1e9, 2)
    }

if __name__ == "__main__":
    print("[*] Running Fast SIMD Diamond Abacus Benchmark...")
    res = benchmark_fast_diamond(500_000, 50)
    print(res)
