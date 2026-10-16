#!/bin/bash
# on the pod: interleaved A/B of ABN_FAST levels vs bitnet.cpp, 5 reps, 64 tok, 8 thr. usage: run_ab_pod.sh OUT "levels"
set -euo pipefail
O=$1; LV=$2; P=/root/corehost/p1; cd $P/bitcpp; ID=$(cat promptA.ids); M=$P/model.safetensors
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_fast.c -o armilla_fast -lm; : > $O
for l in $LV; do ABN_FAST=$l ABN_LOGITS=/tmp/ab$l.bin ./armilla_fast $M 16 8 $ID >/dev/null 2>&1; echo "L$l logits16 sha256=$(sha256sum /tmp/ab$l.bin | cut -c1-16)" | tee -a $O; done
for i in 1 2 3 4 5; do
  echo "ref rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(LD_LIBRARY_PATH=bin ./ref_decode_ppl ggml-model-i2_s.gguf 64 8 $ID 2>&1 >/dev/null | grep decode | cut -d'(' -f1)" | tee -a $O
  for l in $LV; do echo "L$l rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_FAST=$l ./armilla_fast $M 64 8 $ID 2>&1 >/dev/null | grep ^decode | cut -d'(' -f1)" | tee -a $O; done
done
