# A5 laptop proof (2026-10-14, i5-8365U, AVX2 — no VBMI, so the 5-trit pull ran as the scalar twin of the same layout)
Build: `gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_fast5.c -o armilla_fast5 -lm`. Prompt `experiments/bitnet/data/promptA.ids` (sonnet-p1).
- `check.txt`: ABN_CHECK — 420 pulls (7 projections x 30 layers x 2 passes), each equal to the direct dot of its 5-trit codes AND of the weights read from the HF bytes: 0 mismatches.
- `hash0-t5-laptop.txt`: run_hash0.sh (A1 protocol) with ABN_T5=1 at 1/4/8 threads: logits 16ba3979...50acae = A1 (i5-8365U and EPYC 4564P, results/fast0/hash0-*.txt on sonnet-p1).
- `hash-L5-laptop.txt`: ABN_FAST=5 plane 3 vs ABN_FAST=5 + ABN_T5=1, 16 tokens, 4 threads: both c782fadc... = A2's L5 hash (ab-L5-L6.txt, EPYC); logits files byte-identical (cmp).
Speeds here are the scalar twin's and mean nothing; the VBMI kernel (vpermi2b) is compile-checked only (2 vpermt2b in the -mavx512vbmi build, 0 FMA). It runs on the VM: `run_a5_vm.sh`.
