import math
"""
Omniring REST API Controller.
Exposes endpoints to create, configure, simulate, and benchmark
arbitrary Omniring diamond and star topology systems on demand.
"""

import json
from diamond_abacus import DiamondOmniringAbacus, T50_PRIMES
from diamond_simd import FastDiamondAbacus

class OmniringService:
    def __init__(self):
        self.active_abacus = FastDiamondAbacus(ring_count=50)

    def configure_abacus(self, ring_count: int = 50, custom_primes: list = None) -> dict:
        clamped_rings = max(3, min(ring_count, 50))
        self.active_abacus = FastDiamondAbacus(ring_count=clamped_rings)
        return {
            "status": "ok",
            "ring_count": clamped_rings,
            "moduli": [int(m) for m in self.active_abacus.moduli],
            "total_states": f"10^{int(sum(math.log10(int(m)) for m in self.active_abacus.moduli))}"
        }

    def compute_cycle(self, n_val: int, w_val: int, s_val: int, e_val: int, formula: str = "balanced_tensor") -> dict:
        self.active_abacus.load_inputs(n_val, w_val, s_val, e_val)
        res_arr = self.active_abacus.compute(formula)

        # Build detailed ring preview
        sample_rings = []
        for i in range(min(12, self.active_abacus.ring_count)):
            p = int(self.active_abacus.moduli[i])
            sample_rings.append({
                "ring_idx": i,
                "prime": p,
                "north": int(self.active_abacus.state[0][i]),
                "west": int(self.active_abacus.state[1][i]),
                "south": int(self.active_abacus.state[2][i]),
                "east": int(self.active_abacus.state[3][i]),
                "center": int(self.active_abacus.state[4][i]),
            })

        return {
            "status": "ok",
            "formula": formula,
            "ring_count": self.active_abacus.ring_count,
            "center_residues": [int(x) for x in res_arr],
            "sample_rings": sample_rings
        }

    def benchmark_throughput(self, iterations: int = 200_000, ring_count: int = 50) -> dict:
        import time
        clamped_rings = max(3, min(ring_count, 50))
        bench_abacus = FastDiamondAbacus(ring_count=clamped_rings)
        bench_abacus.load_inputs(987654321, 123456789, 555555555, 222222222)

        t0 = time.perf_counter()
        for _ in range(iterations):
            bench_abacus.compute("balanced_tensor")
        t1 = time.perf_counter()

        elapsed = t1 - t0
        cps = iterations / max(elapsed, 1e-9)
        ring_ops = cps * clamped_rings * 3
        latency_ns = (elapsed / max(1, iterations)) * 1e9

        return {
            "status": "ok",
            "iterations": iterations,
            "ring_count": clamped_rings,
            "elapsed_seconds": round(elapsed, 4),
            "cycles_per_sec": round(cps, 1),
            "ring_ops_per_sec": round(ring_ops, 1),
            "effective_mops": round(ring_ops / 1e6, 2),
            "latency_ns": round(latency_ns, 2)
        }

# Global singleton
omniring_service = OmniringService()
