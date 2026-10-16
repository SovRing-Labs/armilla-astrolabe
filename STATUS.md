> **Status: under development — pre-alpha.** Interfaces, numbers and file layout can change without notice.

# armilla-astrolabe — status

Last checked: 2026-10-16, Linux laptop (Intel i5-8365U, 8 threads), on a clean copy of this repo. 6 of 8 suite tests work today; 7 of 8 can run without special hardware. Each line below names its test id in `suite.json` and the evidence file it rests on.
Quick test `PYTHONPATH=engine python3 -m pytest -q engine/tests`: PASS (`9 passed in 6.58s`, fresh copy, empty HOME).


## Works today

- **C1** Carry-free T50 ring abacus: engine self-test 9/9 — PASS 2026-10-16 (`9 passed in 1.62s`); evidence: `experiments/pytest.2026-10-12.log`.
- **C2** Sign / ReLU decided from the rings without rebuilding the number — PASS 2026-10-16 (`sign_relu": {"rings": 50, "range_bits": 311, "values": 26002, "edge_values_near_0_and_M_over_2": 6002, "wrong"`); evidence: `experiments/probes.result.json`.
- **C3** Bit-identical training step across 8 harnesses — PASS 2026-10-16 (`{"date": "2026-10-12", "tokens": 256, "harnesses": 8, "cpu": "one core, AVX-512 VBMI", "960": {"T1": {"forward`); evidence: `experiments/tlut/train_step.result.json`.
- **C6** Exact modular exponentiation faster than GMP's hardened powmod on one core — PASS 2026-10-16 (`{"date": "2026-10-12", "cpu": "Intel Xeon @ 2.10 GHz, 4 vCPU, AVX-512 IFMA", "gmp": "GMP 6.3.0", "method": "RN`); evidence: `experiments/gmp/bench_batch_ct.result.json`.
- **C7** Fixed-base comb 7.8x at 2,048 bits — PASS 2026-10-16 (`{"date": "2026-10-12", "method": "fixed-base comb, 5-bit windows, constant-time table scan, batches of 8, RNS-`); evidence: `experiments/gmp/bench_comb.result.json`.
- **C9** 5-trit weight codes at 1.6 bits per weight — PASS 2026-10-16 (`{"date": "2026-10-12", "rings": [251, 241, 239], "4096x3840": {"mismatches": {"3trit": 0, "5trit": 0, "bitplan`); evidence: `experiments/tlut/tlut5_bench.result.json`.

## Not yet tested

- **C8** BitNet b1.58 2B4T decoded on the Armilla pull: identical logits hash on three CPUs — external: needs something the test laptop does not have; needs: python3, the public BitNet b1.58 2B4T weights (MIT), a second CPU to compare

## Known failures and honest negatives

- **C4** Honest negative: Gelosia lattice core slower than GMP. This is the negative itself. Recorded and re-checked 2026-10-16 (PASS); evidence: `experiments/bench_gelosia_kernel.avx512-native.result.json`, `experiments/bench_gelosia_kernel.avx2.result.json`
- No DOSSIER.md and no DOI yet: the Zenodo toggle for this repo is off, so no release has minted one.

## Corrections

- **C08** — ABACUS-NL jit per-element time is CPU-dependent: 15.4 µs on the laptop i5-8365U, 9.5 µs on a Sapphire Rapids Xeon (about 1,400× slower than f64); every quoted time names its CPU. `experiments/abacus_nl/README.md` row 2 corrected from "~10 µs/element on CPU" on 2026-10-16.

## How to re-check

- One file, any python3 >= 3.8: download [`sovring-suite.pyz`](https://github.com/SovRing-Labs/SovRing/releases/latest/download/sovring-suite.pyz), then `python3 sovring-suite.pyz --run armilla-astrolabe` (headless) or `python3 sovring-suite.pyz` (local page on 127.0.0.1).
- From this checkout: `bash install.sh` (checks python, prepares a venv, runs the suite headless, then opens the page; `--no-gui` to skip the page).
- Quick test only: `PYTHONPATH=engine python3 -m pytest -q engine/tests`.

## Defensive publication

- `TECHNICAL-DISCLOSURE.md` — on public GitHub since 2026-10-12
- DOI: none yet (the Zenodo integration is off for this repo; the next release after it is switched on mints one).
- Licence: PolyForm Small Business 1.0.0 / PolyForm Noncommercial 1.0.0 (LICENSING.md); PolyForm Free Trial for evaluation
