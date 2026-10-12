# Changelog

## 0.1.1 (2026-10-12)

- Provenance: credit prior art in homomorphic encryption for residue-ring base conversion and AVX-512 IFMA modular
  arithmetic (full-RNS BFV 2016, full-RNS CKKS 2018, OpenFHE 2022, OpenFHE-julia 2024, Intel HEXL 2021), and state what in
  §7 remains new relative to it. No code or results changed.

## 0.1.0 (2026-10-12)

First release.

- T50 ring abacus (50 one-byte prime rings, 311-bit range) with five-register centre operations. 9/9 tests.
- Evolved Gelosia lattice: exact 2,000- and 8,000-bit products (0 mismatches in 160 pairs, Python and AVX2/AVX-512 kernel).
  Not faster than GMP on this CPU: the lattice core is 4.5–23× slower and conversion dominates (measured, disclosed).
- Six ring systems with disjoint primes and a two-level read-out: 6,698 bits, 0 mismatches in 600 operations.
- Sign / ReLU by fractional CRT with an exact fallback: 0 wrong in 26,002 values.
- RNS Montgomery modular multiplication on the six systems (2,047-bit modulus): 0 mismatches in 200, correctness only.

Corrections to the 2026-10-11 draft (not released): the "8,000-bit multiply 23× faster" compared a 311-bit ring product with
a full 16,000-bit product; the "Zero-CRT ReLU" used a sum that does not track size (correlation 0.008) and zeroed half of the
positive values. Both are withdrawn and replaced by the measured results above. The Cores Studio page injector was removed
(not part of this package), and `ArmillaAstrolabe.exe` (a shell script) is now `run.sh`.

Layouts and the JIT centre (same release):
- 6-, 8- and 10-system layouts (middle ring of 3 / 5 / 7 systems + centre + top + bottom, 55 rings each): 5,266 / 7,015 /
  8,761 bits. Guard mode (top + bottom as check systems): one whole faulty system corrected 200/200, two detected 200/200.
- JIT-rendered T50 centre firing the Gelosia lattice once per ring turn: identical to the generic kernel, 26.5 µs per 8,000-bit
  turn on one CPU core (1.14× the generic kernel). Not yet tested: GPU / FPGA turn, residue-resident modexp chain against GMP.
- Five-layer diamond 1-3-7-3-1 (15 systems, 825 rings): 13,111 bits; mirror bases of 6,119 / 6,118 bits; RNS Montgomery at
  4,096 bits exact (100/100, centre check 100/100); guard mode corrects two whole faulty systems (200/200) with four check systems.
- Residue-resident 4,096-bit exponentiation on the diamond: exact (257/257 vs GMP); NumPy, one core: 1,027 ms alone, 308 ms each
  in a batch of 256, GMP 21.5 ms. Exact order-independent gradient sums: 1 result over 8 orders (float32: 5, float64: 8).
- Ternary pull engine (outer rings precompute 27-answer residue tables; centre pulls with AVX-512 vpermb; fractional-CRT
  read-out; check ring on a second core): exact on every output; one core 3.1–4.9× faster than int8 GEMV and 3.2–5.9× faster than
  NumPy float32, on par with a ternary bit-plane kernel.
- Gelosia lattice rendered into the centre: exact f16 x ternary GEMV (activation digits x trit groups, diagonals summed in-ring,
  one Garner carry pass). 0 mismatches and bit-identical under reordering; float32 inexact on >99.9 % of outputs; 0.25-0.66x
  float32 speed at full exactness (42 table sets).
- 5-trit P3 codes in the pull (1.6 bits/weight): exact; 40 % less weight memory; 1.38x faster than 3-trit and 1.51x faster than
  bit-planes on 8,192 x 7,680 (equal at 4,096 x 3,840).
- Forward + backward (dx) propagation with f16 activations/gradients: exact both ways; window mode 1.05-2.0x float32 speed,
  bit-reproducible, error of float32's order.
- Beat GMP: RNS Montgomery modexp with AVX-512 IFMA on primes 2^48 - c, exact on 80/80 exponentiations vs GMP powmod;
  1.14x / 1.10x / 1.12x faster than GMP 6.3 at 2,048 / 3,072 / 4,096 bits on one core (0.86x at 1,024). No security claim.
- Batched (8), windowed, constant-time, multi-core RNS-IFMA modexp: exact 128/128 at 2,048 and 4,096 bits. 4,096 bits: 2.36x GMP
  powmod and 2.96x GMP powmod_sec on one core; 2.64x / 3.36x on four cores. Exponent timing test |t| <= 0.4. No security claim.
- Winograd inner product in the base extension: exact, but slower on this CPU (0.73-0.91x step 3: IFMA costs about an add).
  Step 3 vs GMP powmod: 1.28x / 2.53x / 1.94x at 2,048 / 4,096 / 8,192 bits; GMP ahead at 16,384 (subquadratic).
- Fixed-base comb ("Napier's bones" for exponents): exact 64/64; 4,096 bits 2.6 ms vs GMP powmod 22.5 / powmod_sec 32.1 ms
  (8.6x / 12.3x one core; 9.5x / 15.1x four cores). GMP has no fixed-base API.
- Winograd on AVX2-only build: exact, slower (0.59-0.72x). Batched-token forward pull: exact, 1.7-2.8x float32, 0.52-0.64x
  batched bit-planes. GPU-form pull (OpenCL, local-memory tables): forward + backward exact on PoCL.
- 16,384 bits: exact; GMP powmod ahead (1.23x), rings ahead of powmod_sec (2.5x); fixed-base comb 2.75x powmod, 8.46x powmod_sec.
  Fast-CRT tree exact but ~47x slower than the IFMA extension at this size.
- AVX2-only pull (legacy CPUs, pshufb, 2-trit): exact, on par with omnid's sign_epi8 GEMV, 2.0-2.5x float32, half the weight bytes.
- Exact training step over 8 harnesses: bit-identical across batch splits (float32 differs on 12-13 % of latents); 0.24x float32 speed.
- Three-diamond unit (one diamond per core, core in L1): exact; 1.37x one diamond and 3.14x float32 (3 cores) at 12,288 x 7,680,
  equal to 3-core bit-planes; serial precompute/read-out and Python threading limit the scaling (memory reads 31 GB/s).
