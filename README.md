# Armilla Astrolabe — a digital abacus

<!-- status-badges (REPO-ACCESS-C) -->
[![status: pre-alpha (under development)](https://img.shields.io/badge/status-pre--alpha%20%28under%20development%29-orange)](STATUS.md) **Status: pre-alpha, under development** — what works today, what is untested and what failed: [`STATUS.md`](STATUS.md)

[![defensive publication: DOI pending](https://img.shields.io/badge/defensive%20publication-DOI%20pending-blue)](TECHNICAL-DISCLOSURE.md) **Defensive publication:** [`TECHNICAL-DISCLOSURE.md`](TECHNICAL-DISCLOSURE.md) · DOI pending
<!-- /status-badges -->

A number is held as its remainders on many small prime rings, like beads on rods that never pass a carry. On top of that sit
three old ideas, rebuilt for today's processors:

- **Gelosia (Treviso, 1478):** fill every box first, add along the diagonals, carry once at the end;
- **Napier's bones (1617):** precompute the answers, then read them instead of computing;
- **the astrolabe:** read a value's size from its phase on each ring (fractional CRT), without rebuilding it.

The outer rings precompute answers; the centre pulls them straight into SIMD; a check ring verifies every result.

## What it does today (one CPU, every result exact, every number from a file in `experiments/`)

| Use | Result |
|---|---|
| Ternary AI layer, one token (5-trit codes, 1.6 bits/weight) | 1.51× faster than a ternary bit-plane kernel, 3–6× faster than int8 and float32 |
| Older CPUs (AVX2 only) | matches the int8 `sign_epi8` production kernel, 2.0–2.5× float32, half the weight memory |
| f16 forward and backward (window mode) | 1.05–2.0× float32, bit-reproducible |
| Training across 8 harnesses | bit-identical weights however the batch is split (float32 differed on 12–13 %) |
| Three diamonds (one per CPU core) | 3.1× float32 on 3 cores, exact |
| Constant-time modular exponentiation, 2,048–16,384 bits | 2.0–3.0× faster than GMP `powmod_sec` |
| Modular exponentiation, 2,048–8,192 bits | 1.3–2.5× faster than GMP `powmod` |
| Fixed base (signing, key generation) | 8.6× / 12.3× faster than GMP `powmod` / `powmod_sec` at 4,096 bits |
| Fault checking | a whole faulty ring system corrected, more detected (guard mode) |

Where it is not ahead (measured and disclosed): GMP's fast `powmod` above about 8,000 bits; batched prefill (bit-planes);
a full training step (0.24× float32); Winograd's inner product on IFMA and AVX2 CPUs. GPU speed: not yet tested.
No security claim: the code is not audited and the conversions in and out are not constant-time.

## Run it

```bash
./reproduce.sh                                        # clean clone -> self-test -> results table from the committed result files
./install.sh                                         # self-test, then the console on http://127.0.0.1:8989
PYTHONPATH=engine python3 -m pytest -q engine/tests
python3 experiments/probes.py                        # ring abacus, Gelosia, six systems, sign, RNS Montgomery
python3 experiments/tlut/tlut5_bench.py              # ternary pull engine, 5-trit codes   (needs AVX-512 VBMI)
python3 experiments/avx2/bench_pull_avx2.py          # legacy-CPU pull                       (AVX2)
python3 experiments/tlut/prop_f16.py                 # f16 forward + backward
python3 experiments/tlut/train_step.py               # exact training step across 8 harnesses
pip install gmpy2 && python3 experiments/gmp/bench_batch_ct.py   # vs GMP   (needs AVX-512 IFMA)
python3 experiments/gmp/bench_comb.py                # fixed-base comb
```

Full method, every measurement and what is not yet tested: `TECHNICAL-DISCLOSURE.md`.

Closest field kin, credited in `PROVENANCE.md`: Residue Hyperdimensional Computing (Kymn et al., Neural Computation 2025, arXiv 2311.04872) and Integer Echo State Networks (Kleyko et al., arXiv 1706.00280).

Licence: PolyForm Small Business 1.0.0 or PolyForm Noncommercial 1.0.0 (see `LICENSING.md`). Published as prior art; no patents.
