"""
Unit and Integration Test Suite for Omniring Diamond Abacus & API.
Verifies algebraic consistency, CRT reconstruction, SIMD vectorization,
and REST service operations.
"""

import pytest
import numpy as np
from diamond_abacus import DiamondOmniringAbacus, T50_PRIMES
from diamond_simd import FastDiamondAbacus
from omniring_api import omniring_service

def test_t50_primes_coprimality():
    """Verify that all 50 primes in T50 are odd, prime, and pairwise coprime."""
    assert len(T50_PRIMES) == 50
    assert T50_PRIMES[0] == 3
    assert T50_PRIMES[-1] == 233
    for p in T50_PRIMES:
        assert p % 2 != 0
        for divisor in range(2, int(p**0.5) + 1):
            assert p % divisor != 0

def test_diamond_tensor_exactness():
    """Verify (N*S + W*E) in DiamondOmniringAbacus matches CRT reconstruction."""
    abacus = DiamondOmniringAbacus(ring_count=8)
    N = 12345
    W = 67890
    S = 13579
    E = 24680

    abacus.load_inputs(N, W, S, E)
    abacus.execute_diamond_cycle("balanced_tensor")

    expected = N * S + W * E
    reconstructed, mod_product = abacus.crt_reconstruct(max_rings=8)
    assert (expected % mod_product) == reconstructed

def test_diamond_simd_vs_pure_python():
    """Ensure FastDiamondAbacus (vectorized SIMD) matches pure Python OmniringNode."""
    abacus_py = DiamondOmniringAbacus(ring_count=50)
    abacus_simd = FastDiamondAbacus(ring_count=50)

    N, W, S, E = 888888, 777777, 666666, 555555
    abacus_py.load_inputs(N, W, S, E)
    abacus_simd.load_inputs(N, W, S, E)

    res_py = abacus_py.execute_diamond_cycle("balanced_tensor")
    res_simd = abacus_simd.compute("balanced_tensor")

    assert np.array_equal(np.array(res_py["residues_c"]), res_simd)

def test_superposition_sum_and_interlock():
    """Test alternative diamond formulas: superposition bundling and cyclic cut interlock."""
    abacus_simd = FastDiamondAbacus(ring_count=50)
    abacus_simd.load_inputs(100, 200, 300, 400)

    res_sum = abacus_simd.compute("superposition_sum")
    # Expected: (100 + 200 + 300 + 400) % m = 1000 % m
    for i, m in enumerate(abacus_simd.moduli):
        assert res_sum[i] == (1000 % m)

    res_cut = abacus_simd.compute("cyclic_interlock")
    # Expected: ((100 + 300) * (200 + 400)) % m = (400 * 600) % m = 240000 % m
    for i, m in enumerate(abacus_simd.moduli):
        assert res_cut[i] == (240000 % m)

def test_omniring_api_service():
    """Verify OmniringService endpoints respond properly."""
    cfg = omniring_service.configure_abacus(15)
    assert cfg["status"] == "ok"
    assert cfg["ring_count"] == 15

    cycle = omniring_service.compute_cycle(500, 400, 300, 200, "balanced_tensor")
    assert cycle["status"] == "ok"
    assert len(cycle["sample_rings"]) == 12

    bench = omniring_service.benchmark_throughput(iterations=10_000, ring_count=15)
    assert bench["status"] == "ok"
    assert bench["cycles_per_sec"] > 10_000

from gelosia_astrolabe import EvolvedGelosiaTracker, AstrolabeSphericalProjector

def test_gelosia_tracker_matmul():
    """Verify Gelosia RNS GEMM matches ground truth for multi-digit matrices."""
    tracker = EvolvedGelosiaTracker(T50_PRIMES[:10])
    A = [[100, 200], [300, 400]]
    B = [[5, 6], [7, 8]]
    # Ground truth:
    # C[0,0] = 100*5 + 200*7 = 1900
    # C[0,1] = 100*6 + 200*8 = 2200
    # C[1,0] = 300*5 + 400*7 = 4300
    # C[1,1] = 300*6 + 400*8 = 5000
    rns_C = tracker.tensor_gelosia_matmul(A, B)

    for ring_idx, m in enumerate(T50_PRIMES[:10]):
        assert rns_C[0, 0, ring_idx] == (1900 % int(m))
        assert rns_C[0, 1, ring_idx] == (2200 % int(m))
        assert rns_C[1, 0, ring_idx] == (4300 % int(m))
        assert rns_C[1, 1, ring_idx] == (5000 % int(m))

def test_astrolabe_projection_and_core():
    """Verify astrolabe stereographic projection angles and Akushsky core magnitude."""
    astrolabe = AstrolabeSphericalProjector(T50_PRIMES[:15])
    test_val = 999999
    res = np.array([test_val % int(m) for m in T50_PRIMES[:15]], dtype=np.int64)

    plate_x, plate_y, sphere = astrolabe.stereographic_astrolabe_projection(res)
    assert len(plate_x) == 15
    assert len(plate_y) == 15
    assert -1.0 <= sphere["sphere_X"] <= 1.0
    assert -1.0 <= sphere["sphere_Y"] <= 1.0
    assert -1.0 <= sphere["sphere_Z"] <= 1.0
    assert 0.0 <= sphere["phase_coherence"] <= 1.0

    core_mag = astrolabe.akushsky_core_magnitude(res)
    assert 0.0 <= core_mag <= 1.0


def test_magnitude_tracks_x_over_m():
    """The magnitude reading must follow X / M (fractional CRT)."""
    import math, random
    astro = AstrolabeSphericalProjector(T50_PRIMES)
    M = math.prod(int(m) for m in T50_PRIMES)
    rng = random.Random(3)
    for _ in range(500):
        x = rng.randrange(M)
        r = np.array([x % int(m) for m in T50_PRIMES], dtype=np.int64)
        assert abs(astro.akushsky_core_magnitude(r) - x / M) < 1e-12


def test_rns_sign_and_relu_centred_range():
    """Sign and ReLU are exact in the centred range, including values next to 0 and M/2."""
    import math, random
    astro = AstrolabeSphericalProjector(T50_PRIMES)
    M = math.prod(int(m) for m in T50_PRIMES)
    rng = random.Random(4)
    vals = list(range(-300, 301)) + [M // 2 - 3 + i for i in range(6)] + [-(M // 2) + i for i in range(3)]
    vals += [rng.randrange(-(M // 2), M // 2) for _ in range(500)]
    for v in vals:
        r = np.array([v % int(m) for m in T50_PRIMES], dtype=np.int64)
        want = 1 if (v % M) < (M + 1) // 2 else -1
        assert astro.rns_sign(r) == want, v
        out = astro.rns_relu(r)
        assert (out == 0).all() if want < 0 else (out == r).all()
