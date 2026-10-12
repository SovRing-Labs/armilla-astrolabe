"""
Sovereign Omniring Diamond Abacus Engine.
Implements the 5-node Diamond Topology (North, West, South, East, Center)
using 50 coprime prime rings (T50 system: primes 3...233, dynamic range ~ 10^93).
Bypasses carry propagation: local ring SIMD modular arithmetic feeding a central
recombination/reduction hub.
"""

import time
import math
from typing import List, Dict, Any, Tuple

# The T50 ring system: 50 pairwise coprime odd primes 3..233
# Total states ~ 10^93, \sum m_i = 5,348
T50_PRIMES = [
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233
]

class OmniringNode:
    """A single T50 modular bead ring register."""
    def __init__(self, name: str, moduli: List[int] = None):
        self.name = name
        self.moduli = moduli or list(T50_PRIMES)
        self.num_rings = len(self.moduli)
        # Residues held strictly in [0, m_i - 1]
        self.residues = [0] * self.num_rings

    def set_integer(self, val: int):
        """Carry-free projection into the RNS ring space."""
        for i, m in enumerate(self.moduli):
            self.residues[i] = val % m

    def set_residues(self, res: List[int]):
        for i, m in enumerate(self.moduli):
            self.residues[i] = res[i] % m

    def add(self, other: 'OmniringNode') -> 'OmniringNode':
        """Carry-free SIMD-style modular addition."""
        out = OmniringNode(f"({self.name}+{other.name})", self.moduli)
        for i, m in enumerate(self.moduli):
            out.residues[i] = (self.residues[i] + other.residues[i]) % m
        return out

    def multiply(self, other: 'OmniringNode') -> 'OmniringNode':
        """Carry-free SIMD-style modular multiplication."""
        out = OmniringNode(f"({self.name}*{other.name})", self.moduli)
        for i, m in enumerate(self.moduli):
            out.residues[i] = (self.residues[i] * other.residues[i]) % m
        return out

    def scalar_mul(self, k: int) -> 'OmniringNode':
        out = OmniringNode(f"({self.name}*{k})", self.moduli)
        for i, m in enumerate(self.moduli):
            out.residues[i] = (self.residues[i] * k) % m
        return out


class DiamondOmniringAbacus:
    r"""
    Diamond Abacus Topology:
                 [ North: T50_N ]
                       /\
                      /  \
      [ West: T50_W ] ---- [ Center: T50_C ] ---- [ East: T50_E ]
                      \  /
                       \/
                 [ South: T50_S ]
    """
    def __init__(self, ring_count: int = 50, custom_primes: List[int] = None):
        if custom_primes:
            self.moduli = custom_primes[:ring_count]
        else:
            self.moduli = T50_PRIMES[:ring_count]

        self.nodes = {
            "N": OmniringNode("N", self.moduli),
            "W": OmniringNode("W", self.moduli),
            "S": OmniringNode("S", self.moduli),
            "E": OmniringNode("E", self.moduli),
            "C": OmniringNode("C", self.moduli)
        }

    def load_inputs(self, n_val: int, w_val: int, s_val: int, e_val: int):
        self.nodes["N"].set_integer(n_val)
        self.nodes["W"].set_integer(w_val)
        self.nodes["S"].set_integer(s_val)
        self.nodes["E"].set_integer(e_val)

    def execute_diamond_cycle(self, formula: str = "balanced_tensor") -> Dict[str, Any]:
        """
        Executes a high-order arithmetic reduction into Center.
        Formulas:
          - 'balanced_tensor': C = (N * S) + (W * E)  (Systolic 2x2 Matrix Determinant / Bilinear Inner Product)
          - 'superposition_sum': C = N + W + S + E    (Simultaneous Multi-Bead Hypervector Bundle)
          - 'cyclic_interlock': C = (N + S) * (W + E) (Dual-T50 Cut Factor Product)
        """
        t0 = time.perf_counter()

        N = self.nodes["N"]
        W = self.nodes["W"]
        S = self.nodes["S"]
        E = self.nodes["E"]

        if formula == "balanced_tensor":
            # (N * S) + (W * E)
            p1 = N.multiply(S)
            p2 = W.multiply(E)
            C = p1.add(p2)
        elif formula == "superposition_sum":
            # N + W + S + E
            s1 = N.add(W)
            s2 = S.add(E)
            C = s1.add(s2)
        elif formula == "cyclic_interlock":
            # (N + S) * (W + E)
            s1 = N.add(S)
            s2 = W.add(E)
            C = s1.multiply(s2)
        else:
            raise ValueError(f"Unknown formula: {formula}")

        self.nodes["C"] = C
        t1 = time.perf_counter()
        elapsed_us = (t1 - t0) * 1e6

        return {
            "formula": formula,
            "elapsed_us": round(elapsed_us, 3),
            "ring_count": len(self.moduli),
            "residues_c": C.residues,
            "sample_rings": [
                {"prime": m, "N": N.residues[i], "W": W.residues[i], "S": S.residues[i], "E": E.residues[i], "C": C.residues[i]}
                for i, m in enumerate(self.moduli[:8])
            ]
        }

    def crt_reconstruct(self, max_rings: int = 6) -> Tuple[int, int]:
        """
        Reconstructs the integer from Center residues using Garner's / Gauss Chinese Remainder Theorem
        up to max_rings (to prevent arbitrarily huge integer overhead during fast evaluation).
        """
        sub_mod = self.moduli[:max_rings]
        sub_res = self.nodes["C"].residues[:max_rings]

        M = 1
        for m in sub_mod:
            M *= m

        total = 0
        for i, m in enumerate(sub_mod):
            Mi = M // m
            # Modular inverse
            inv = pow(Mi, -1, m)
            total = (total + sub_res[i] * Mi * inv) % M

        return total, M


def benchmark_diamond_power(num_iterations: int = 100_000, ring_count: int = 50) -> Dict[str, Any]:
    """
    Measures raw throughput of the carry-free diamond abacus.
    """
    abacus = DiamondOmniringAbacus(ring_count=ring_count)
    abacus.load_inputs(123456789, 987654321, 555555555, 999999999)

    t0 = time.perf_counter()
    for _ in range(num_iterations):
        # Balanced tensor: (N * S) + (W * E) across all rings
        abacus.execute_diamond_cycle("balanced_tensor")
    t1 = time.perf_counter()

    total_s = t1 - t0
    ops_per_sec = num_iterations / max(total_s, 1e-9)
    # Each diamond step executes 2 modular multiplies + 1 modular add per ring = 3 ring ops * ring_count ops
    ring_ops_per_sec = ops_per_sec * ring_count * 3

    return {
        "iterations": num_iterations,
        "total_seconds": round(total_s, 4),
        "diamond_cycles_per_sec": round(ops_per_sec, 2),
        "ring_ops_per_sec": round(ring_ops_per_sec, 2),
        "effective_mops": round(ring_ops_per_sec / 1e6, 2),
        "dynamic_range_states": f"10^{int(sum(math.log10(m) for m in T50_PRIMES[:ring_count]))}"
    }

if __name__ == "__main__":
    print("[*] Benchmarking Diamond Omniring Abacus (50 rings, 5-node diamond)...")
    res = benchmark_diamond_power(50_000, 50)
    print(res)
