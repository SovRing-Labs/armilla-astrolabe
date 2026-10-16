# Armilla Astrolabe, a digital abacus — Technical Disclosure (2026-10-12): a carry-free ring abacus, the evolved Gelosia, 6/8/10 ring systems, a JIT centre, a ternary pull engine, sign without rebuild

Dated prior art. Every number below was measured on one cloud CPU (Intel Xeon @ 2.10 GHz, 4 vCPU, AVX2 and AVX-512, Python 3,
NumPy 2.5, GMP 6.3 through gmpy2). Harnesses and result files are in `experiments/`; the probe pass rules are fixed in the harness
code (`experiments/probes.py`) and were written before the run, in the same session. Each section carries a FIELD PREDICTS line.
No patents are filed; this text is published so the ideas stay free to use.

Author: Juan Carlos De Santiago, with the Sovereign Raptor fleet (AI-assisted).

Terms are the field's: *ring* = one modulus of a residue number system (RNS); *residue* = a number's remainder on that ring;
*range* M = the product of the moduli; *CRT* = Chinese Remainder Theorem read-out; *limb* = one fixed-size digit of a big number;
*lane* = one ring's slot in a vector register.

## 0. What this is, in one paragraph

A number is kept as its remainders on many small rings at once (here 50 rings, the odd primes 3 to 233). Adding and multiplying
happen on every ring at the same time with no carries between rings, like the bead rods of an abacus that never have to pass a
carry. Three things are disclosed on top of that: (1) an **evolved Gelosia**: the Renaissance lattice method run on the rings, which
multiplies numbers far larger than the ring range exactly; (2) a **six-system layout** (three ring systems in a triangle around a
centre, one above, one below), each system with its own primes, joined by a two-level read-out; (3) **sign and ReLU without
rebuilding the number**, by the fractional CRT with an exact fallback. What was measured, including what did not hold, is below.

## 1. Origin of the design

The author's design notes (written before 2026-10-11) set the idea: a multi-moduli ring counter (rings 3, 5, 7, 13, 17) as an
interlocking-wheel abacus; four to five such systems joined by a larger ring; and "an evolved version of the Gelosia method" to run
vector-symbolic and back-propagation arithmetic. The same notes name the problem this disclosure answers in §5: in RNS, comparison,
ReLU and division need the number's size, which the residues do not show. The six-system shape (§4) is the author's: "4 ring
systems at the middle layer, 3 in a triangle around it and 1 in the center, with one system on top and bottom of the center ring."

## 2. The T50 ring abacus (`engine/diamond_abacus.py`, `engine/diamond_simd.py`, `engine/omniring_api.py`)

- **Description.** Moduli: the first 50 odd primes, 3 … 233. Every residue fits in one byte. Range M = ∏ mᵢ ≈ 1.25 × 10⁹³,
  which is **311 bits** (Σ mᵢ = 5,348). Five registers (N, W, S, E and the centre C) each hold one T50 residue vector; three
  centre operations are built in: C = N·S + W·E, C = N + W + S + E, C = (N + S)·(W + E), each computed ring by ring, mod mᵢ.
- **G1 (exactness).** 9/9 unit tests pass, including exact CRT agreement for the three centre operations
  (`experiments/pytest.2026-10-12.log`).
- **Range by ring count** (`experiments/probe_ring_counts.result.json`): T25 126.5 bits · T50 310.1 bits · T53 333.9 bits (the
  last odd prime below 256 is 251, so **53 rings is the limit for one-byte prime lanes**) · 54 prime powers in one byte
  (256, 243, 125, 49, 121, 169, …) 362.8 bits (the one-byte maximum) · T55 349.9 bits (needs 2-byte lanes) · T64 423.5 bits ·
  T128 998.4 bits.
- **FIELD PREDICTS:** ring-wise operations are exact up to M and wrap above it (Szabo & Tanaka, 1967); a value larger than M
  cannot be recovered.
- **Negative (as measured).** Software speed is not a gain at this size: the vectorised five-register cycle runs at 448,620
  cycles/s, which is 0.04 × the rate of plain Python integers doing the same job (`experiments/test_harness.2026-10-12.log`).

## 3. The evolved Gelosia (`experiments/probes.py` P3, `experiments/gelosia_kernel.c`, `experiments/bench_gelosia_kernel.py`)

The paper Gelosia (Treviso, 1478) fills a grid with digit-by-digit products, adds along the diagonals, and carries once at the end.
Its point is that **all box products are independent and the carry is postponed to one final pass.** The evolved form keeps exactly
that, with ring residues inside the boxes:

1. Cut each operand into limbs of b bits: a = Σ aᵢ 2^{ib}, c = Σ cⱼ 2^{jb} (K limbs each).
2. Put each limb into the rings (its T50 residue vector). Each lattice box (i, j) is one ring-wise product aᵢ·cⱼ mod mᵣ, for all 50
   rings at once. The K² boxes do not depend on each other.
3. Sum each diagonal d = i + j **inside the rings**. In the kernel the products are summed without reduction and each diagonal is
   reduced once per ring (lazy reduction: multiplication kept apart from accumulation, as on paper).
4. Read each diagonal out once by CRT. This is exact if no diagonal can reach M: **K·(2^b − 1)² < M**. The code checks the
   bound and refuses a limb size that could wrap.
5. One carry pass: product = Σ_d D_d · 2^{db}.

- **G2 (exactness, Python reference).** 2,000-bit × 2,000-bit (limb 150, 196 boxes) and 8,000-bit × 8,000-bit (limb 150, 2,916
  boxes; limb 128, 3,969 boxes): 60 random pairs, **0 mismatches** against Python integers; worst diagonal 306 bits under a 311-bit
  range; a 200-bit limb is refused as designed (`experiments/probes.result.json`, P3).
- **G3 (exactness and speed, compiled kernel).** C kernel, lane-major (64 lanes, 50 used), built with `gcc -O3 -mavx2 -mfma`
  (24 `vpmulld` vector multiplies in the inner loop) and with `-march=native` (AVX-512). 100 random pairs, **0 mismatches**
  (`experiments/bench_gelosia_kernel.avx2.result.json`, `.avx512-native.result.json`).

  | Size | Lattice core (AVX2 / AVX-512) | Limbs into rings (Python) | Diagonals out + carry (Python) | GMP multiply | Python int multiply |
  |---|---|---|---|---|---|
  | 2,000-bit | 16.8 / 8.1 µs | 257 µs | 324 µs | 0.74 µs | 4.5 µs |
  | 8,000-bit | 35.4 / 26.3 µs | 868 µs | 1,434 µs | 6.9 µs | 45.0 µs |

  Each value is the mean of 50 pairs; core times are the mean of 200 calls per pair. Conversion, GMP and Python columns are from
  the AVX2 result file.

- **FIELD PREDICTS:** on a general CPU, residue arithmetic loses to tuned binary big-integer libraries (GMP uses Karatsuba and
  Toom-Cook above a few thousand bits); its gains are expected in dedicated parallel hardware, where each ring lane is a small
  independent multiplier, and where data stays in residue form across many operations.
- **Negative (as measured).** On this CPU the lattice core alone is **4.5–23× slower than GMP**, and converting into and out of
  the rings costs 100–1,000× more than the core. The evolved Gelosia is **exact at 8,000 bits but not faster here.**
- **Earlier claims withdrawn.** The bundle dated 2026-10-11 reported an "8,000-bit multiply 23.36× faster" and a "16×16 GEMM of
  2,000-bit elements, exact". The 8,000-bit timing multiplied residues mod M (311 bits) against a full 16,000-bit product. Those
  are different computations: the ring result is a·c mod M, not a·c (`probes.result.json`, P0: equals the true product = false; equals the product mod M = true).
  The GEMM check compared residues of the true product, which always agree, and the result cannot be rebuilt above 311 bits. The
  original logs are kept as `experiments/*.2026-10-11.author.log`; re-runs are in `experiments/*.2026-10-12.log`.

## 4. Six ring systems: triangle, centre, top, bottom (`experiments/probes.py` P2)

- **Description.** Six ring systems, each with its **own** primes (no prime shared). For the probe: the 420 largest primes below
  2¹⁶ are dealt in turn to A, B, C (the triangle), centre, top and bottom, so each system has 70 rings, about 1,116 bits; the six
  together hold **6,698 bits**. Residues are 2-byte lanes. Arithmetic runs ring by ring in all six systems. The read-out has two
  levels: each system rebuilds its own value mod Mₛ, then the centre joins the six values by CRT over (M_A, …, M_bottom).
- **G4 (exactness).** 300 random pairs of 3,300-bit numbers, addition and multiplication: 600 results, **0 mismatches** against
  Python integers, 1.52 s in Python (`experiments/probes.result.json`, P2).
- **FIELD PREDICTS:** disjoint moduli sets multiply their ranges, and CRT applied in two levels equals CRT applied in one.
- **Why the shape matters.** It matches the two bases that RNS modular arithmetic for cryptography needs (§6): the triangle can be
  one base, top and bottom the other, and the centre the converter between them or a check channel. The 2026-10-11 code put all five
  diamond nodes on the **same** 50 primes, so its nodes were registers, not separate ring systems; this section replaces that.
- **Variations also disclosed.** Ring systems of 55, 64, 100 or more rings per system. One-byte lanes up to 53 rings (or 54 prime
  powers) per system; 2-byte lanes above that. Assigning primes by size instead of in turn. A redundant (check) system that detects
  a wrong residue (redundant RNS). The original 3-5-7-13-17 rings as one system, with larger systems around it.

## 5. Sign, ReLU and size without rebuilding the number (`engine/gelosia_astrolabe.py`)

- **Description.** For X in [0, M): X / M = frac( Σᵢ rᵢ·kᵢ / mᵢ ), with kᵢ = (M/mᵢ)⁻¹ mod mᵢ (the fractional CRT). It is computed
  in float64, with no big-integer rebuild, and its error is below about n·2⁻⁵². Numbers are read in the centred range: X < M/2 is ≥ 0
  and X ≥ M/2 is negative. Sign is f < ½. Within 10⁻⁹ of 0, ½ or 1, the float reading is not trusted; an exact mixed-radix
  conversion decides instead. ReLU zeroes a vector whose sign is negative. In the code this is `akushsky_core_magnitude` (name kept
  for compatibility), `rns_sign` and `rns_relu`.
- **G5 (exactness).** 26,002 values: 4,001 next to 0, 2,001 next to M/2 and 20,000 random. **0 wrong signs.** 6,002 values took
  the exact fallback, all of them near a boundary. Mean 12.5 µs per sign in Python, including building the residues
  (`experiments/probes.result.json`, P1). Unit tests: `test_magnitude_tracks_x_over_m` (500 values, error < 10⁻¹²) and
  `test_rns_sign_and_relu_centred_range`.
- **FIELD PREDICTS:** the fractional CRT gives X/M to floating-point accuracy, with an ambiguity zone near the boundaries that
  needs an exact step (Vu, 1985; Kawamura et al., 2000).
- **Earlier claim withdrawn.** The 2026-10-11 code used C(X) = Σ rᵢ/mᵢ mod 1 as an "Akushsky core function" and reported a
  "Zero-CRT ReLU" at 225,617 per second. That sum does not follow the number's size: its correlation with X/M over 4,000 random
  values is 0.008 (`probes.result.json`, P0). On 2,000 positive values below 10⁶ that ReLU zeroed 50.0 % of them (P0). Akushsky's core function uses chosen weights; the unweighted sum is kept only as a display, `phase_sum`.

## 6. RNS Montgomery modular multiplication on the six systems (`experiments/probes.py` P4)

- **Description.** Montgomery multiplication (Montgomery, 1985) done in residues (Posch & Posch, 1995; Bajard, Didier & Kornerup,
  1998). Base A = the triangle (210 rings, 3,349 bits) and base B = top + bottom (140 rings, 2,233 bits). The modulus N is the
  product of two 1,024-bit probable primes (2,047 bits); A > 4N and B > 2N hold.
  - q = −x·y·N⁻¹ is computed ring-wise in base A and extended to base B.
  - r = (x·y + q·N)·A⁻¹ is computed in base B and extended back to A and to the centre (70 rings).
  - The centre's residues must agree with r: this is a check channel.

  Base extension here is the **exact** mixed-radix method (a reference implementation), not the fast approximate one.
- **G6 (exactness).** 200 random modular multiplications: **0 mismatches** against Python, r < 2N every time, and the centre check
  agreed 200/200. A modular exponentiation with a 64-bit exponent matched `pow()`. Mean 30.2 ms per multiplication in Python
  (`experiments/probes.result.json`, P4).
- **FIELD PREDICTS:** RNS Montgomery is exact when the bases exceed the modulus bounds; hardware versions reach high throughput
  by doing base extension approximately (Kawamura et al., 2000) and use a redundant modulus for exact extension
  (Shenoy & Kumaresan, 1989).
- **No security claim.** This probe checks correctness only. It is not constant-time and has not been reviewed for side channels.
  Any security property belongs to the cryptographic algorithm, not to the ring representation.

## 7. Larger layouts (6, 8, 10 systems) and the JIT centre (`experiments/probe_nodes.py`, result `probe_nodes.result.json`)

- **Description.** One family: a middle ring of k ring systems around the centre system, one system on top and one below;
  k = 3 gives the six-system layout of §4, k = 5 gives 8 systems, k = 7 gives 10. Each system here has 55 rings of 16-bit primes,
  no prime shared. Two roles are disclosed for top and bottom: **range mode** (they add range / form Montgomery base B) and
  **guard mode** (they carry the largest primes and act as redundant systems, so a whole wrong system is detected and one is
  corrected by leaving each system out in turn — redundant RNS at the level of whole systems).
- **G7 (range and crypto fit, arithmetic).** 6 systems: 5,266 bits · 8: 7,015 · 10: 8,761. With base A = middle
  ring and base B = top + bottom, the largest Montgomery modulus is about 1,754 bits in every design: **two systems of 55
  rings are the bottleneck**. RSA-2048 needs 66+ rings per system, or base B = top + bottom + centre, or bases split evenly
  (§6 used 70 rings per system and passed at 2,047 bits).
- **G8 (guard mode).** 200 trials per design: clean values read back 200/200; one whole system corrupted → corrected 200/200 in
  all three designs; two systems corrupted → detected 200/200, never mis-corrected. Data range left in guard mode: 6 systems
  3,508 bits, 8 systems 5,257, 10 systems 7,003.
- **G9 (the JIT centre, fired every ring turn).** The centre's Gelosia core is rendered as C for one shape (8,000-bit operands,
  54 limbs of 150 bits, the 50 T50 moduli and their reduction constants baked in), compiled and loaded at run time
  (175.8 ms), then fired once per turn on data already in residues. 2,000 turns: diagonals identical to the generic
  kernel 200/200; 26.54 µs per turn against 30.37 µs generic (1.14×); 5.5 × 10⁹ ring products per second on
  one core (2,916 boxes × 50 rings per turn).
- **The five-layer diamond (`experiments/probe_diamond.py`, result `probe_diamond.result.json`).** Layers top 1 · second 3 ·
  middle 7 (the centre and six around it) · second-mid 3 · bottom 1 = 15 ring systems in one cyclone, every system flowing into the
  centre within two hops. 55 rings per system (825 rings): **13,111 bits**. The mirror symmetry gives two equal bases: upper
  (top + second layer + three of the middle) 6,119 bits and lower 6,118 bits, so a Montgomery modulus up to about 6,116 bits fits,
  and the centre checks every result. **G10:** a 4,096-bit modulus, 100 RNS Montgomery multiplications, 0 mismatches against
  Python, centre check 100/100, a 64-bit-exponent modexp equal to `pow()` (96 ms per multiplication in the exact Python reference;
  no security claim). **G11 guard mode:** with top, bottom and centre as check systems (10,475 data bits) one faulty system is
  corrected 200/200 and two or three are detected 200/200; with top, bottom and one system of each second layer as checks
  (9,598 data bits) **two whole faulty systems are corrected 200/200** and three or four detected 200/200, never mis-corrected.

- **Residue-resident exponentiation on the diamond (`experiments/bench_diamond_modexp.py`, result `bench_diamond_modexp.result.json`).**
  Numbers enter the rings once and leave once; every one of the ~6,150 Montgomery multiplications of a 4,096-bit exponent stays in
  residues. Base extension is one matrix product per step (the float-read correction count of Kawamura et al., 2000), exact in
  float64. **G12:** 257 exponentiations (4,096-bit modulus and exponent), 0 mismatches against GMP `powmod`. Speed on one CPU
  core in NumPy: 1,027 ms one at a time, **308 ms each in a batch of 256**, against GMP 21.5 ms: GMP is 47× faster alone and
  14.4× faster against the batch. Batching closes the gap 3.3× because the extension becomes one large matrix product.
  NOT YET TESTED: a compiled or GPU base extension (the dominant cost), where RNS hardware has reported its gains.
- **Exact, order-independent sums for training (`experiments/probe_exact_reduce.py`, result `probe_exact_reduce.result.json`).**
  1,000,000 gradient values added in 8 orders, as 8 harnesses would: float32 gave 5 different results, float64 8, the T50 rings
  (fixed point, scale 2⁻⁶⁰) **1**, equal to the exact sum. An int8 dot product of length 2²⁰ overflows an int32 accumulator; the
  rings return it exactly (an int64 accumulator also would; rings matter past 64 bits).

- **Beating GMP: RNS Montgomery with AVX-512 IFMA (`experiments/gmp/`, results `bench_rns_modexp.result.json`,
  `bench_ifma_modexp.result.json`).** Why GMP was faster: it multiplies 64-bit pieces in a serial carry chain, while our first
  versions paid for interpreted NumPy, 16-bit rings (385 per base, ~150,000 products per base extension) and integer division.
  Rings have no carries, so every ring can sit in its own vector lane; the remaining question was arithmetic density per
  instruction. Step 1, C with 28-bit rings in 64-bit lanes (`rns_avx512.c`): exact, GMP 1.68× faster at 4,096 bits (was 47×
  in NumPy). Step 2 (`rns_ifma.c`): rings are primes **m = 2⁴⁸ − c (c < 2¹⁶)**, one residue per 64-bit lane; **IFMA**
  (`vpmadd52lo/hi`) forms 8 products of 52-bit inputs per instruction; each base-extension sum adds n products per lane with
  no reduction (low part < n·2⁵², high part < n·2⁴⁴) and is reduced once: S = H·2⁵² + L ≡ H·16c + L (2⁵² = 16·2⁴⁸ ≡ 16c), split
  again by IFMA, then T = Tₕ·2⁴⁸ + Tₗ ≡ Tₕ·c + Tₗ and one or two conditional subtractions. 48-bit rings need 88 rings per base
  at 4,096 bits (28-bit rings: 152-160). **G20 (exactness):** 80 exponentiations (20 each at 1,024 / 2,048 / 3,072 / 4,096 bits,
  random bases and full-length exponents) equal GMP `powmod`. **G21 (speed, one core, 15 interleaved pairs, median per
  Montgomery multiplication inside a full exponentiation):**

  | Modulus bits | Rings per base | RNS-IFMA | GMP 6.3 | RNS speed vs GMP | Whole exponentiation (RNS / GMP) |
  |---|---|---|---|---|---|
| 1024 | 24 | 0.489 µs | 0.42 µs | **0.86×** | 0.745 / 0.639 ms |
| 2048 | 48 | 1.424 µs | 1.618 µs | **1.14×** | 4.322 / 4.91 ms |
| 3072 | 72 | 2.762 µs | 3.043 µs | **1.1×** | 12.766 / 14.063 ms |
| 4096 | 88 | 2.27 µs | 2.542 µs | **1.12×** | 13.946 / 15.613 ms |

  **Faster than GMP from 2,048 bits up**, on one core, exact; GMP is still ahead at 1,024 bits. Not constant-time; no security
  claim. NOT YET TESTED: batching several exponentiations so the extension tables are reused (the 256-batch gave 3.3× in NumPy);
  4-way output blocking of the IFMA extension; the rings as the five-layer diamond's mirror bases; multi-core.

- **Step 3: batched, windowed, constant-time, multi-core (`experiments/gmp/rns_ifma_batch.c`, `bench_batch_ct.py`, result
  `bench_batch_ct.result.json`).** Eight messages per batch share every base-extension table load (16 IFMA accumulators: low /
  high per message), so the table is read 8× less per message; fixed 5-bit windows (x⁰ … x³¹ precomputed, then 5 squarings + 1
  multiply per window); **constant time in the exponent**: every window does the same work and the table entry is chosen by
  scanning all 32 entries with masks (no secret-dependent branch or address); OpenMP over batches. Each message has its own base
  and its own full-length exponent. **G22 (exactness):** 128 of 128 messages equal GMP `powmod` at 2,048 and at 4,096 bits.
  **G23 (speed, ms per exponentiation, interleaved medians; GMP fast `powmod` / constant-time `powmod_sec`; four-core GMP = 4
  processes):**

  | Bits | RNS 1 core | GMP 1 core (fast / sec) | RNS vs GMP | RNS 4 cores | GMP 4 cores (fast / sec) | RNS vs GMP | Welch t (B3) |
  |---|---|---|---|---|---|---|---|
| 2048 | 1.732 | 2.76 / 3.577 | **1.59× / 2.07×** | 0.571 | 0.916 / 1.078 | **1.61× / 1.89×** | 0.37 |
| 4096 | 9.672 | 22.824 / 28.607 | **2.36× / 2.96×** | 2.502 | 6.611 / 8.406 | **2.64× / 3.36×** | 0.4 |

  **G24 (timing, dudect-style):** one batch timed 300 times each with all-zero and with random exponents, interleaved: Welch t =
  0.37 (2,048) and 0.4 (4,096), far below the 4.5 threshold. This is a wall-clock test of the exponent path only; NOT YET TESTED: a
  cycle-level and cache-level side-channel review, and constant-time entry / exit conversions (done in Python here). No security
  claim.
- **Step 4: Winograd's inner product (1968) inside the base extensions (`experiments/gmp/rns_ifma_wino.c`, `bench_wino.py`,
  result `bench_wino.result.json`).** Σ ξᵢTᵢ = Σₖ (ξ₂ₖ + T₂ₖ₊₁)(ξ₂ₖ₊₁ + T₂ₖ) − Σₖ ξ₂ₖξ₂ₖ₊₁ − Σₖ T₂ₖT₂ₖ₊₁: the table half is a constant per
  output, the message half one scalar per message, so each output needs n/2 products instead of n. **G25:** exact at all sizes
  (16-32 messages per size, 2,048–16,384 bits). **Measured: it is slower here** (step 4 / step 3 time ratio 0.91, 0.77, 0.73 at
  2,048 / 4,096 / 8,192 bits): on this CPU an IFMA costs about what an add costs, so trading half the multiplies for added adds and
  broadcasts does not pay. Scope: one CPU with IFMA; NOT YET TESTED where a multiply costs more than an add (byte-ring lookups, FPGA
  multipliers, GPUs without wide integer FMA). ms per exponentiation, one core:

  | Bits | Rings per base | Step 3 | Step 4 (Winograd) | GMP powmod | GMP powmod_sec | Step 4 / step 3 |
  |---|---|---|---|---|---|---|
  | 2048 | 48 | 2.354 | 2.599 | 3.023 | 5.242 | 0.91 |
  | 4096 | 88 | 8.599 | 11.233 | 21.737 | 29.15 | 0.77 |
  | 8192 | 176 | 57.433 | 78.173 | 111.365 | 243.063 | 0.73 |
  | 16384 | 344 | — | 888.036 | 707.176 | 2052.673 | — |

  Step 3 against GMP in this run: 1.28× / 2.53× / 1.94× faster than `powmod` at 2,048 / 4,096 / 8,192 bits. At 16,384 bits GMP
  (subquadratic multiplication) is 1.26× faster than step 4: the base extension is O(n²), so the lead peaks between 4,096 and
  8,192 bits on this design. NOT YET TESTED: a sub-quadratic extension, and step 3 at 16,384 (needs the chunked fold).
- **Fixed-base comb, "Napier's bones" for exponents (`experiments/gmp/rns_ifma_batch.c` `batch_fixedbase_ct`, `bench_comb.py`,
  result `bench_comb.result.json`).** For a fixed base g (signing, key generation, Diffie-Hellman) the rods are precomputed once:
  tab[k][j] = g^(j·32ᵏ) in Montgomery form for every window position k; then gᵉ = Πₖ tab[k][wₖ]: one constant-time table scan and
  one multiplication per window, **no squarings**. Tables 10.1 MB (2,048 bits) and 36.9 MB (4,096 bits). **G26:** 64 of 64 exact at both
  sizes. **G27 (ms per exponentiation, interleaved medians):** 2,048 bits: 0.577 (1 core) / 0.145 (4 cores) against GMP
  powmod 3.007 / 1.092 and powmod_sec 4.503 / 1.278; **4,096 bits: 2.616 / 0.634 ms against powmod 22.482 / 6.035 and powmod_sec
  32.114 / 9.551 — 8.59× / 12.28× faster on one core, 9.52× / 15.06× on four.** Scope: GMP has no fixed-base API,
  so these GMP columns are what a GMP user gets; a fixed-base comb on GMP's own arithmetic is NOT YET TESTED. The comb itself is
  classic (Brickell et al. 1992; Lim & Lee 1994).
- **Winograd on AVX2 only (`experiments/avx2/`, result `bench_ext_avx2.result.json`).** Built with `-mavx2 -mno-avx512f` (no zmm, no
  IFMA in the code). Exact; slower than the plain extension (0.69× / 0.6× / 0.59× at 80 / 152 / 304 rings): AVX2 issues two 32-bit vector
  multiplies per cycle, so the trade does not pay from Haswell on. NOT YET TESTED: pre-2013 CPUs and multiply-poor hardware.
- **16,384 bits (`experiments/gmp/bench_16k.py`, result `bench_16k.result.json`).** Step 3 now folds its sums every 240 terms (the
  high parts stay under 52 bits for any ring count). **G29:** step 3 and step 4 exact (8 of 8). ms per exponentiation, one core:
  step 3 1064.8, step 4 852.6, GMP powmod 692.5, powmod_sec 2130.5: **GMP's fast powmod is ahead at this size (1.23× step 4);
  the constant-time comparison still favours the rings 2.5×.** Here Winograd (step 4) is the faster of the two (the 2·344² table
  no longer fits the caches, so halving the products matters). **Fixed-base comb at 16,384 bits:** exact 8/8, **251.9 ms: 2.75× faster
  than powmod and 8.46× faster than powmod_sec** (table 577.2 MB, built once in 649.5 s of Python).
- **Fast Chinese remaindering (subproduct tree, Borodin & Moenck 1974) as the base extension (same result file, L3).** Exact at
  16,384 and 32,768 bits. On GMP integers from Python: 1280.8 µs (16,384) and 4039.8 µs (32,768) per extension, against about
  27.0 µs for the O(n²) IFMA extension at 16,384 (upper estimate from G29): **the quadratic SIMD extension wins by ~47× here**; the tree only
  pays far beyond these sizes. NOT YET TESTED: a C tree, and sizes above 32,768 bits.
- **FIELD PREDICTS:** redundant RNS detects up to r and corrects up to ⌊r/2⌋ faulty residues when the redundant moduli are the
  largest (Watson & Hastings, 1966; Barsi & Maestrini, 1973); specialising a kernel to constants known at run time removes
  bounds and constant loads but does not change the work.
- **Where it stands.** On this CPU one turn (26.5 µs) is still slower than one GMP 8,000-bit multiply (about 6–7 µs, §3); the
  turn is pure parallel work (2,916 independent boxes, then 107 diagonal sums), the shape that maps onto many small multipliers.
  NOT YET TESTED: the turn on a GPU (the omni-gpu Vulkan path) or FPGA; a chain of turns that stays in residues end to end
  (modular exponentiation against GMP `powmod`, the workload where RNS is expected to pay).

## 8. The ternary pull engine: outer precomputers, a resonant pull at the centre, fired into SIMD (`experiments/tlut/`)

The author's design (2026-10-12): "all the outer rings were precomputers feeding the answer directly into the center ring where
it transferred to the JIT 50t that used a resonator like mechanism to pull the answer instantly and fire it into SIMD", with
ternary weights and the cyclone's wiring and synchronisation. Built and measured as follows.

- **Ternary removes every multiply.** A weight in {−1, 0, +1} adds, subtracts or skips. Three weights have 27 patterns.
- **Outer rings precompute the answers** (`tlut_precompute3`). For each group of 3 inputs and each ring, all 27 possible answers
  (w₀x₀ + w₁x₁ + w₂x₂) mod m are computed once per input vector, two groups to a 64-byte table (answers 0–26 and 32–58). This is
  shared by every weight row that reads the same input (the cyclone's precompute-ahead stream).
- **The centre pulls the answer instead of computing it** (`tlut_gemv3`). Each weight row stores one byte per group: which of the
  27 answers it needs. One AVX-512 VBMI `vpermb` pulls the answers of 32 rows × 2 groups from the table in one instruction;
  `vpmaddubsw` adds each row's two answers; one code load feeds every ring (all rings turn together); table tiles of 96 group pairs
  stay in L1 while every row block pulls from them; sums stay carry-free in their ring and are reduced by vector Barrett once
  per tile.
- **The read-out resonates once per output.** The fractional CRT over the data rings (251, 241, 239) gives y / M in one dot
  product, so value, sign, ReLU and requantisation to the next layer's int8 need no big-integer rebuild.
- **A guard system on its own core.** The check ring 233 is pulled by a second core at the same time; every output must agree
  with it (redundant RNS).
- **G13 (exactness).** Two layer sizes, three weight matrices each: **0 mismatches** against int64 W·x, the guard agreed on every
  output (`experiments/tlut/tlut_cyclone.result.json`, C1).
- **G14 (speed, one core, median).** The outer precompute takes 15.8 µs (4,096 × 4,032) and 43.7 µs (8,192 × 8,064), 1–2 % of the layer.
  Whole layer against: int8 GEMV **3.06× / 4.89× faster**; NumPy float32 GEMV **3.15× / 5.88× faster**; a ternary
  bit-plane GEMV (AVX-512 masked adds, the strongest baseline here) 0.78× / 1.08× when measured interleaved (on par; the
  shared cloud machine varied 0.78–1.43× between runs). Weights: 2.67 bits per weight (codes) against 2 (bit-planes) and 8 (int8).
- **G15 (guard overlap).** With the guard on a second core the wall time is 1.8× / 1.08× the unguarded layer; at the large size
  the check costs 8 %.
- **Path of the kernel** (each step measured, `tlut_bench.py`): v1 one ring per pass 4,670 µs → v2 all rings per code load, two
  groups per table 2,228 µs → v3 L1-resident tiles 2,009 µs (incl. scalar precompute) → v3 with the vectorised outer precompute
  and the check ring on the guard core 885–967 µs (4,096 × 4,032).
- **The Gelosia lattice rendered into the centre: exact f16 × ternary (`experiments/tlut/gelosia_f16.py`, result
  `gelosia_f16.result.json`).** Both factors are cut into digits, as in the paper method: the weights into trits (groups of 3 =
  one pattern code), each f16 activation into its exact fixed-point integer X = x·2²⁴ (f16's smallest step is 2⁻²⁴) cut into
  6 balanced base-256 digits. The outer rings fill the boxes: one 27-answer table set per (digit, ring), 6 × 7 = 42 sets
  (rings 251 … 223, range ≈ 2⁵⁵). The centre pulls every set from one code load. Each diagonal (digit d, weight 256ᵈ) is summed
  **inside each ring**, with no carry, and one exact mixed-radix (Garner) read-out is the single carry pass. The pull is
  **rendered for its shape**: moduli, digit and ring counts, tiling and loop bounds are constants in generated C, compiled and
  loaded at run time (699.3 / 530.2 ms). **G16:** 0 mismatches against the exact integer product on every output, at
  both sizes; bit-identical over 4 reorderings of the inputs. NumPy float32 on the same f16 inputs is not exact on
  4,094 of 4,096 and 8,187 of 8,192 outputs (largest relative error 6.6e-04). Speed, one core: 10.0 ms / 36.4 ms
  against float32 2.5 / 24.0 ms (0.25× / 0.66×): full exactness costs 42 table sets.
  This is the property THEAD's float lookup-table backward pass lacked: it was registered "NOT bit-identical" and shipped OFF.
  NOT YET TESTED: a digit window per token (3 digits, 5 rings = 15 sets, about 2.8× less work); THEAD's 5-trit codes
  (243-entry tables through vpermi2b); the same lattice as the backward pass.
- **5-trit codes in the pull (`experiments/tlut/tlut5_bench.py`, result `tlut5_bench.result.json`).** THEAD's P3 packing
  (code = Σ (wⱼ + 1)·3ʲ, 5 weights per byte, 1.6 bits per weight, the information limit is 1.585). Per group of 5 inputs and
  ring, the outer rings precompute all 243 answers (vectorised, 71.7 / 172.5 µs). The centre pulls 64 rows with two
  `vpermi2b` (the two 128-entry halves) and one blend on the code's top bit. **G17:** exact on every output; weights
  3.1 / 12.6 MB against 5.2 / 21.0 MB (3-trit) and 3.9 / 15.7 MB (bit-planes); interleaved medians,
  one core: 4,096 × 3,840 all three equal (≈ 0.95 ms); **8,192 × 7,680: 5-trit 3.44 ms, 3-trit 4.76 ms, bit-plane
  5.21 ms (1.38× and 1.51×)**: with less to read, the larger layer is faster.
- **Forward and backward propagation, f16 (`experiments/tlut/prop_f16.py`, result `prop_f16.result.json`).** One ternary layer,
  y = W x and dx = Wᵀ dy (the input gradient, through Wᵀ codes: the form of THEAD's float table backward, registered "NOT
  bit-identical"). **G18 (full exactness, 42 table sets):** 0 mismatches forward and backward at 4,032² and 8,064²,
  bit-identical over reorderings; float32 is inexact on 4,029 / 3,988 of 4,032 outputs; speed 0.29–0.74× float32 forward,
  0.26–0.42× backward. **G19 (window mode, 3 digits × 5 rings = 15 table sets, a per-vector 23-bit window):** exact on the
  windowed integers (0 mismatches both ways); the window rounds the finest values once on entry (1,121 of 4,032 and
  2,310 of 8,064 inputs), so the error against the exact f16 product is of float32's order (2.2e-03 vs
  9.8e-04 forward, 2.5e-03 vs 3.6e-03 backward at 8,064) while every run is bit-reproducible; speed
  **1.06× / 2.0× float32 forward and 1.05× / 1.54× backward**. NOT YET TESTED: the weight gradient dW (an outer
  product, dense), a whole model's training step, 5-trit codes inside the f16 lattice.
- **Forward with 8 tokens per code load (`experiments/tlut/tlut_batch_bench.py`, result `tlut_batch_bench.result.json`).** Exact. Per
  token: 852.6 / 2165.8 µs against a batched bit-plane kernel (masks loaded once, byte-pair sums) 439.3 / 1383.8 µs and NumPy float32 GEMM
  1413.6 / 6001.5 µs: **1.66× / 2.77× faster than float32, 0.52× / 0.64× the batched bit-plane.** Batching saves code loads, but every
  token still needs its own pulls, so the pull is compute-bound here.
- **The pull as a GPU kernel (`experiments/gpu/pull_opencl.py`, result `pull_opencl.result.json`).** OpenCL: one work-group = 64
  rows; each tile of 5-trit answer tables is copied into local (shared) memory once per work-group; each work-item pulls its row's
  answers for 3 data rings + the check ring. **G28:** forward and backward exact, check ring agrees, at 1,024 and 4,096 rows. Run
  on PoCL (an OpenCL driver on this CPU), so this proves the GPU form, not GPU speed. NOT YET TESTED: a real GPU.
- **Legacy CPUs: the pull on AVX2 only (`experiments/avx2/pull_avx2.c`, `bench_pull_avx2.py`, result `bench_pull_avx2.result.json`).**
  Built with `-mavx2 -mno-avx512f` (no zmm in the code). 2-trit codes (9 patterns) pulled with `pshufb` (a 16-entry byte table per
  128-bit lane), 3 data rings, 16-bit lazy sums, vector Barrett; codes stored row-block-major and table tiles held in L1 (the first
  layout, group-major, ran 3.5× slower from strided memory access). **G30:** exact; weights at half of int8; one core:
  4,096²: 2049.3 µs vs omnid's production `sign_epi8` GEMV 2110.8 µs (1.03×) and NumPy float32 4194.6 µs (2.05×);
  8,192²: 9152.8 vs 9320.2 (1.02×) and 23195.9 (2.53×).
- **One complete exact training step across 8 harnesses (`experiments/tlut/train_step.py`, result `train_step.result.json`).** Ternary
  layer from an int32 latent; 256 tokens. Forward and backward by the pull (8 tokens per code load), int8 error, weight gradient
  per harness (int8 × int8 in float32, every partial < 2²⁴ so exact), **carry-free all-reduce in 4 byte rings** (C, any order),
  Garner read-out, integer update. **G31:** forward, backward and dW exact (0 mismatches) at 960² and 1,920²; two different
  splits of the batch over the harnesses, reduced in opposite orders, give **bit-identical latents and weights**, while the
  float32 version of the same step differs on 118,026 of 921,600 and 439,972 of 3,686,400 latent entries. Speed, one core:
  108.0 / 467.3 ms against float32 26.6 / 110.0 ms (0.25× / 0.24×); the forward and backward pulls are now the main cost
  (a 256-token batch suits float BLAS; NumPy glue was 6× of the time before the reduce moved to C).
- **The three-diamond unit (`experiments/tlut/three_diamonds.py`, result `three_diamonds.result.json`; author's design 2026-10-12:
  "a three diamond unit setup, where all the answers are then pushed from the core of each diamond that lives in cpu cache").**
  One diamond = one CPU core; its core (a 5-trit table tile, ~18 KB, and its 16-bit accumulators) stays in that core's L1; the
  outer-ring precompute fills the tables once per input; each diamond pushes its finished residues into the shared output. Work
  shared by rows (each diamond owns a third of the weights) or by rings (each diamond owns one data ring). **G32:** exact in every
  configuration (both splits, both code layouts, two sizes). One token, interleaved medians: at 12,288 × 7,680 three diamonds by
  rows take 4080.7 µs against one diamond 5596.8 µs (1.37×), bit-planes on 3 cores 4083.3 µs (1.0×) and NumPy float32 on 3
  cores 12808.5 µs (**3.14× faster**); weights 18.9 MB (5-trit) against 23.6 MB (bit-planes) and 377.5 MB (float32). Splitting by
  rings scales slightly less than by rows. At 6,144 × 7,680 three diamonds give no gain (0.95×). **Where the scaling goes:** the
  machine reads 31 GB/s on 3 threads (`membw.result.txt`), so memory is not the limit; the precompute and read-out still run on
  one core and Python starts the threads on every call. NOT YET TESTED: persistent C threads with the precompute split across
  the diamonds (each diamond fills the tables for a third of the inputs and pushes them to the others).
- **FIELD PREDICTS:** lookup-table ternary and low-bit GEMV (T-MAC, Wei et al., 2024; the bitnet.cpp lookup kernels) beat
  multiply-based kernels on CPUs; table lookups with byte permutes (pshufb / vpermb) are the established SIMD form.
- **What is ours here:** the tables hold **ring residues**, so every entry is an exact byte with no bit-plane splitting and no
  overflow at any depth; several rings turn on one code load; the read-out is the fractional CRT; and a check ring on a separate
  core guards every output.
- **NOT YET TESTED (next tests):** a whole ternary model's forward pass layer to layer in this form; batches of inputs (one table
  set per token); the pull on a GPU (a shared-memory table per workgroup); backward pass (the same pull with transposed codes).

## 9. Interfaces shipped

- `install.sh` checks Python and NumPy, runs the self-test (`engine/test_harness.py`), then starts the local console
  (`gui/standalone_app.py`, bound to 127.0.0.1:8989). `run.sh` calls `install.sh`.
- `engine/omniring_api.py` is a small service object used by the console: it configures 3–50 rings, runs centre cycles and reports
  the range (now Σ log₁₀ mᵢ).

## 10. Not yet tested (next tests)

- The six-system layout as one running engine (the probe proves the arithmetic, not an engine).
- A lattice kernel that keeps data in residues across many operations, which is where RNS is expected to pay. A GPU (Vulkan) or
  FPGA lattice. A vectorised base extension, and fast approximate extension.
- Division and scaling (Adam-style optimiser steps) in residues. Comparison of two numbers (sign of X − Y) is covered by §5;
  division is not.
- Back-propagation fully in residues: the 2026-10-11 "full backprop cycle 8.06 ms" ran with the old ReLU and is withdrawn.
- 2-byte-lane systems of 55+ rings as the abacus default; constant-time RNS Montgomery; RSA-size exponentiation timing.

## 11. Provenance (borrowed / ours)

**Borrowed, with credit**
- Residue number systems and the CRT: Sunzi Suanjing (3rd–5th century); Garner (1959); Szabo & Tanaka (1967).
- Mixed-radix conversion: Garner (1959); Szabo & Tanaka (1967).
- Fractional (approximate) CRT for sign detection: Vu (1985); Kawamura, Koike, Sano & Shimbo (2000).
- The core function in RNS: Akushsky and co-workers (1977); Gonnella (1991).
- Montgomery multiplication: Montgomery (1985).
- Inner product with half the multiplications: Winograd (1968).
- Fast Chinese remaindering by subproduct trees: Borodin & Moenck (1974).
- Fixed-base comb / precomputed exponent tables: Brickell, Gordon, McCurley & Wilson (1992); Lim & Lee (1994); Napier's rods (1617).
- Pseudo-Mersenne moduli (2ᵏ − c) for fast reduction: field-known (Crandall, 1992); AVX-512 IFMA (Intel, 52-bit integer FMA).
- **Residue-ring arithmetic in homomorphic encryption (added in 0.1.1):** full-RNS BFV (Bajard, Eynard, Hasan & Zucca, 2016);
  full-RNS CKKS (Cheon, Han, Kim, Kim & Song, 2018); the OpenFHE library (Al Badawi et al., 2022) and its Julia bindings
  OpenFHE-julia (Schlottke-Lakemper et al., 2024, v0.6.4); **AVX-512 IFMA for modular arithmetic: Intel HEXL (Boemer et al.,
  2021).** These established base conversion between ring sets and IFMA-accelerated modular arithmetic before this work; §7's
  use of IFMA for RNS modular multiplication is an application of that field knowledge, not a first. No code from these
  projects was read (their READMEs only).
- **Residue numbers in high-dimensional vectors, and integer reservoirs (added in 0.1.2):** Residue Hyperdimensional Computing
  (Kymn, Kleyko, Frady, Bybee, Kanerva, Sommer & Olshausen, "Computing with Residue Numbers in High-Dimensional Representation",
  Neural Computation 2025, arXiv 2311.04872) encodes residue numbers over coprime moduli as hypervectors; Integer Echo State
  Networks (Kleyko, Frady, Kheffache & Osipov, arXiv 1706.00280) run reservoir computing in small-integer arithmetic. Both
  predate and are the closest field kin to this work's residue-coded rings and integer reservoir cores; this package's rings
  are exact integer residues in lattice boxes, not hypervectors, and claims here are limited to the measured results above.
  No code from either was read (papers only).
- RNS Montgomery: Posch & Posch (1995); Bajard, Didier & Kornerup (1998); Kawamura et al. (2000).
- Redundant-modulus base extension: Shenoy & Kumaresan (1989).
- Redundant RNS error detection and correction: Watson & Hastings (1966); Barsi & Maestrini (1973).
- Run-time code specialisation (JIT): field-known.
- Lookup-table low-bit GEMV: T-MAC (Wei et al., 2024); byte-permute table lookups (SSSE3 pshufb, AVX-512 VBMI vpermb).
- Lattice (gelosia) multiplication: the Treviso Arithmetic (1478) and Pacioli, *Summa* (1494); older Arabic and Indian sources.
- Lazy (delayed) modular reduction and limb-wise (schoolbook) multiplication: field-known.
- Astrolabe plate and phase pictures: the classical stereographic astrolabe (Ptolemy, *Planisphaerium*).

**Ours**
- The Gelosia lattice run with ring residues in the boxes, under the diagonal bound K·(2^b − 1)² < M, so numbers far beyond the
  ring range multiply exactly.
- The six-system layout (triangle, centre, top, bottom) with disjoint primes and a two-level read-out, mapped onto the two bases
  and the check channel of RNS Montgomery; its 8- and 10-system extensions and the 1-3-7-3-1 five-layer diamond with mirror bases; guard mode at the level of whole systems.
- A JIT-rendered T50 centre that fires the Gelosia lattice once per ring turn.
- The three-diamond unit: one diamond per CPU core, its core resident in that core's cache, answers pushed to a shared output.
- The ternary pull engine: residue answer tables precomputed by the outer rings, pulled by the centre with vpermb across all
  rings per code load, read out by fractional CRT, guarded by a check ring on its own core.
- The five-register centre operations.
- The negative results above.

What remains ours in §7 (relative to the work above): batched constant-time fixed windows on 2⁴⁸ − c rings with a one-pass
lazy fold, the fixed-base comb on RNS-IFMA, the measurements against GMP, the six-system / diamond layouts with mirror bases, and
guard mode at the level of whole ring systems.

All code was written from the method descriptions; no other implementation was read.
