#!/bin/bash
# ARMILLA-FAST-0 ladder, ON the pod (/root/corehost/p1/bitcpp): each ABN_FAST level vs bitnet.cpp, prompt A, 64 tokens, interleaved, 3 reps, load logged;
# logits sha256 (16 tokens, 8 threads, ReLU²) of every level must equal the baseline hash. usage: run_ladder_pod.sh OUTDIR THREADS "levels" [src.c]
set -euo pipefail
O=${1:?outdir}; T=${2:-8}; LV=${3:-"0 1 2"}; SRC=${4:-armilla_fast.c}; P=/root/corehost/p1; mkdir -p "$O"; cd $P/bitcpp; N=64; ID=$(cat promptA.ids); M=$P/model.safetensors
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp $SRC -o armilla_fast -lm
F=$O/ladder.txt; echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); threads=$T; $SRC sha256 $(sha256sum $SRC | cut -c1-16); levels: $LV" > $F
for l in $LV; do ABN_FAST=$l ABN_LOGITS=/tmp/l$l.bin ./armilla_fast $M 16 $T $ID >/tmp/l$l.tok 2>/dev/null; echo "level $l logits_sha256=$(sha256sum /tmp/l$l.bin | cut -c1-64) tokens_sha256=$(sha256sum /tmp/l$l.tok | cut -c1-16)" | tee -a $F; done
for i in 1 2 3; do
  echo "ref rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(LD_LIBRARY_PATH=bin ./ref_decode_ppl ggml-model-i2_s.gguf $N $T $ID 2>&1 >/dev/null | grep decode)" | tee -a $F
  for l in $LV; do echo "armilla L$l rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_FAST=$l ./armilla_fast $M $N $T $ID 2>&1 >/dev/null | grep -E '^decode|^time split|^head surv' | tr '\n' ' ' | sed 's/ (8 threads[^)]*)//')" | tee -a $F; done
done
