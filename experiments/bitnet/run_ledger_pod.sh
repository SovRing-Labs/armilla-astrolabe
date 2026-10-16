#!/bin/bash
# on the pod: ledger run of one ABN_FAST level (prompt A, 64 tok, 8 thr), hash check with ledger off vs on. usage: run_ledger_pod.sh OUTDIR LEVEL
set -euo pipefail
O=$1; L=$2; P=/root/corehost/p1; mkdir -p $P/bitcpp/$O; cd $P/bitcpp; ID=$(cat promptA.ids)
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_fast.c -o armilla_fast -lm
ABN_FAST=$L ABN_LOGITS=/tmp/lg_off.bin ./armilla_fast $P/model.safetensors 16 8 $ID >/tmp/lg_off.tok 2>/dev/null
ABN_FAST=$L ABN_LEDGER=$O/ledger.jsonl ABN_LOGITS=/tmp/lg_on.bin ./armilla_fast $P/model.safetensors 16 8 $ID >/tmp/lg_on.tok 2>/dev/null
echo "L$L logits off=$(sha256sum /tmp/lg_off.bin | cut -c1-16) on=$(sha256sum /tmp/lg_on.bin | cut -c1-16) load=$(cut -d' ' -f1 /proc/loadavg)" | tee $O/hash.txt
ABN_FAST=$L ABN_LEDGER=$O/ledger.jsonl ./armilla_fast $P/model.safetensors 64 8 $ID 2>$O/run.err >/dev/null; grep ^decode $O/run.err | tee -a $O/hash.txt
