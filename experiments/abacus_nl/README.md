# abacus_nl — the digital abacus: non-linear maths inside the rings (ABACUS-NL-0, 2026-10-13)

exp, 1/x, ln, softmax, log-sum-exp and cross-entropy computed as ring operations on the 20 T50 outer moduli (131…233),
no floating point anywhere, bit-exact and independent of summation order.

| Step | File | What it proves | Result |
|---|---|---|---|
| 1 | `ring_nl.py` → `ring_nl.result.json` | same algorithm over exact Python ints (oracle) and over residues | 0 mismatches; shuffled order identical; softmax rel err 8.3e-12, LSE 1.1e-11 |
| 2 | `jit_nl.py` → `jit_nl.result.json` | C rendered at run time with constants baked (JIT centre), vs the oracle | 0 mismatches on 4 shapes; ~10 µs/element on CPU |
| 3 | `law_gate.py` → `law_gate.result.json` | math-core law `mathkb.softmax.normalised_shift_invariant` | 500/500; broken constant fails 465/500 |
| 4 | `gpu_nl.py`, `ring_nl.cu` → `gpu_nl.result.json` | ring GEMM (ternary W, int8 A) + softmax/LSE/loss on CUDA vs the CPU kernel | RTX A4500: 0 mismatches ×3 runs ×2 shapes; school step 128×4096 = 29.4 ms |

Method sources (clean-room, from published descriptions): RNS scaling (Szabo & Tanaka 1967), mixed-radix conversion,
Newton reciprocal, atanh series for ln, Ozaki Scheme II for the RNS GEMM framing (arXiv 2504.08009).
Run instructions and gotchas: `~/ai-stack/docs/systems/omniring-abacus/MANUAL.md` § Ring non-linear ops. Plan + status log:
`~/ai-stack/_Pads/worker/claude/ABACUS-NL-0.plan.md`.
