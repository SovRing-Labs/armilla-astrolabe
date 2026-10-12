"""
Evolved Gelosia Tracker & Spherical Astrolabe Coordinate Engine.
Implements:
1. Evolved Gelosia 2D Systolic Lattice:
   Decomposes matrix multiply and tensor contraction into a 2D grid of modular RNS cells.
   Multiplication is strictly decoupled from accumulation (Renaissance Gelosia principle).
   Accumulation flows along anti-diagonals without carry ripple.
2. Astrolabe Coordinate Ring Mapping:
   Maps residue spaces to stereographic projections on the celestial sphere:
   - Rete (star coordinates / dynamic operand vectors)
   - Mater / Tympan (fixed latitude rings / coprime modulus coordinate plates)
   - Alidade / Rule (angular phase alignment and phase-shift multiplication)
3. Zero-Reconstruction Magnitude / Non-Linear Sign Estimator:
   Bypasses CRT conversion tax for activation functions (e.g. ReLU, Signum) using
   the fractional CRT (an approximate X/M) with an exact mixed-radix fallback near the sign boundary.
"""

import time
import math
import numpy as np
from typing import List, Tuple, Dict, Any

# Canonical 50 Coprime Primes (T50)
T50_PRIMES = np.array([
    3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53,
    59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113, 127,
    131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193, 197, 199,
    211, 223, 227, 229, 233
], dtype=np.int64)

class EvolvedGelosiaTracker:
    """
    Evolved Gelosia Lattice Multiplier for high-dimensional tensors.
    Unlike classic 15th-century Gelosia that ripples carries along diagonals,
    this RNS Gelosia operates over an N x K grid where every cell contains
    an independent prime-ring residue.
    """
    def __init__(self, moduli: np.ndarray = None):
        self.moduli = moduli if moduli is not None else T50_PRIMES
        self.num_rings = len(self.moduli)

    def to_rns(self, val: int) -> np.ndarray:
        return np.array([val % int(m) for m in self.moduli], dtype=np.int64)

    def tensor_gelosia_matmul(self, matrix_A: List[List[int]], matrix_B: List[List[int]]) -> np.ndarray:
        """
        Computes Matrix_A (M x K) @ Matrix_B (K x N) completely in RNS Gelosia grid.
        Returns shape (M, N, num_rings).
        Zero carry-ripple between rings, zero scalar bit-ripples.
        """
        M = len(matrix_A)
        K = len(matrix_A[0])
        N = len(matrix_B[0])

        # Convert matrices into RNS tensor planes
        # rns_A: shape (M, K, num_rings)
        rns_A = np.zeros((M, K, self.num_rings), dtype=np.int64)
        for i in range(M):
            for k in range(K):
                rns_A[i, k] = self.to_rns(matrix_A[i][k])

        # rns_B: shape (K, N, num_rings)
        rns_B = np.zeros((K, N, self.num_rings), dtype=np.int64)
        for k in range(K):
            for j in range(N):
                rns_B[k, j] = self.to_rns(matrix_B[k][j])

        # Step 1: Gelosia Lattice Multiplication (Local Products in each lattice box)
        # For each output cell (i, j), we have K diagonal terms: term_k = A[i,k] * B[k,j]
        # In hardware, this is an instant array of 8-bit multipliers
        # Shape: (M, N, K, num_rings)
        lattice_products = (rns_A[:, None, :, :] * rns_B.transpose(1, 0, 2)[None, :, :, :]) % self.moduli

        # Step 2: Gelosia Diagonal Accumulation (Carry-free sum across K)
        # Summing along the K axis modulo each prime
        rns_C = np.sum(lattice_products, axis=2) % self.moduli
        return rns_C


class AstrolabeSphericalProjector:
    """
    Spherical Astrolabe Coordinate Projection.
    Historically, an astrolabe projects the 3D celestial sphere onto a 2D plane (stereographic projection),
    using rotating rings (the Rete) over fixed coordinate rings (the Tympan/Plate).

    In the Omniring:
    - Mater/Tympan: Fixed modular harmonic rings (the prime bases m_i) arranged as nested concentric latitude circles.
    - Rete: Dynamic residue vectors rotated by angular phase θ_i = 2π * (residue_i / m_i).
    - Stereographic Projection: Maps the 50-dimensional toroidal residue coordinates into
      a unit sphere / stereographic plane (X, Y, Z), allowing instantaneous geometric distance
      and phase-lock calculation without computing big-integer CRT values.
    """
    def __init__(self, moduli: np.ndarray = None):
        self.moduli = moduli if moduli is not None else T50_PRIMES
        self.num_rings = len(self.moduli)
        # Precompute harmonic radii for nested Tympan rings
        self.tympan_radii = np.linspace(0.2, 1.0, self.num_rings)

    def project_residues_to_angles(self, residues: np.ndarray) -> np.ndarray:
        """θ_i = 2π * (r_i / m_i) - Phase angle on each astrolabe ring."""
        return (2.0 * math.pi * (residues % self.moduli)) / self.moduli

    def stereographic_astrolabe_projection(self, residues: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        """
        Projects 50 residues onto the Astrolabe 2D plate (x, y) and celestial sphere (X, Y, Z).
        """
        theta = self.project_residues_to_angles(residues)
        
        # 2D Plate Coordinates (Rete points on Tympan plates)
        plate_x = self.tympan_radii * np.cos(theta)
        plate_y = self.tympan_radii * np.sin(theta)

        # 3D Celestial Sphere projection via spherical harmonics
        # Center of mass of the phase vectors
        mean_cos = np.mean(np.cos(theta))
        mean_sin = np.mean(np.sin(theta))
        phase_coherence = float(np.sqrt(mean_cos**2 + mean_sin**2)) # Astrolabe alidade alignment [0..1]
        
        # Spherical coordinate angles
        ra = float(np.arctan2(mean_sin, mean_cos)) # Right ascension
        dec = float(math.asin(max(-1.0, min(1.0, 2.0 * (phase_coherence - 0.5))))) # Declination

        sphere_coord = {
            "right_ascension_rad": ra,
            "declination_rad": dec,
            "phase_coherence": phase_coherence,
            "sphere_X": math.cos(dec) * math.cos(ra),
            "sphere_Y": math.cos(dec) * math.sin(ra),
            "sphere_Z": math.sin(dec)
        }

        return plate_x, plate_y, sphere_coord

    def phase_sum(self, residues: np.ndarray) -> float:
        """
        Plain phase sum: sum(r_i / m_i) mod 1.
        This is a phase reading of the astrolabe plates. It is NOT a magnitude estimate:
        on 4,000 random T50 values its correlation with X/M is 0.008 (experiments/probe_claims.json).
        Kept for the console's plate display only.
        """
        fractions = (residues % self.moduli) / self.moduli
        return float(np.sum(fractions) % 1.0)

    def _crt_weights(self):
        if getattr(self, "_k", None) is None:
            mods = [int(m) for m in self.moduli]
            M = 1
            for m in mods:
                M *= m
            self._M = M
            self._k = np.array([pow(M // m, -1, m) for m in mods], dtype=np.float64)
        return self._k

    def fractional_crt(self, residues: np.ndarray) -> float:
        """
        Fractional CRT: X / M = frac( sum_i r_i * k_i / m_i ),  k_i = (M / m_i)^-1 mod m_i.
        An approximate X / M in float64 without rebuilding X (fractional / approximate CRT;
        cf. Vu 1985, Kawamura et al. 2000). Its absolute error is below about n * 2^-52.
        """
        k = self._crt_weights()
        r = (residues % self.moduli).astype(np.float64)
        return float(np.sum(r * k / self.moduli) % 1.0)

    def _exact_frac_side(self, residues: np.ndarray) -> int:
        """Exact fallback by mixed-radix conversion: returns 0 if X < M/2, else 1."""
        mods = [int(m) for m in self.moduli]
        r = [int(v) % m for v, m in zip(residues, mods)]
        digits = []
        for i, m in enumerate(mods):
            a = r[i]
            for j in range(i):
                a = ((a - digits[j]) * pow(mods[j], -1, m)) % m
            digits.append(a)
        X, w = 0, 1
        for d, m in zip(digits, mods):
            X += d * w
            w *= m
        return 0 if 2 * X < w else 1

    def akushsky_core_magnitude(self, residues: np.ndarray) -> float:
        """
        Magnitude reading in [0, 1): X / M by fractional CRT (see fractional_crt).
        The name is kept for API compatibility; the core-function idea is Akushsky's,
        this estimator is the fractional CRT.
        """
        return self.fractional_crt(residues)

    def rns_sign(self, residues: np.ndarray, guard: float = 1e-9) -> int:
        """
        Sign in the centred range: X in [0, M/2) is >= 0, X in [M/2, M) is negative.
        Fast path: fractional CRT. Inside the guard band around 0, 1/2 and 1 the float
        reading is not trusted and an exact mixed-radix conversion decides.
        Returns +1 (non-negative) or -1.
        """
        f = self.fractional_crt(residues)
        if min(f, abs(f - 0.5), 1.0 - f) < guard:
            return 1 if self._exact_frac_side(residues) == 0 else -1
        return 1 if f < 0.5 else -1

    def rns_relu(self, residues: np.ndarray) -> np.ndarray:
        """ReLU in the centred range: negative values become 0, others pass unchanged."""
        if self.rns_sign(residues) < 0:
            return np.zeros_like(residues)
        return residues
