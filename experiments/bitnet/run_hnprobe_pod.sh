#!/bin/bash
# on the pod: dump final hidden vectors (64 decode steps, prompt A) then run head_probe.py. usage: run_hnprobe_pod.sh OUTDIR
set -euo pipefail
O=$1; P=/root/corehost/p1; mkdir -p $P/bitcpp/$O; cd $P/bitcpp; ID=$(cat promptA.ids)
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_fast.c -o armilla_fast -lm
ABN_FAST=4 ABN_DUMPHN=$O/hn.bin ./armilla_fast $P/model.safetensors 64 8 $ID >/dev/null 2>&1
nice -n 19 python3 head_probe.py $P/model.safetensors $O/hn.bin $O/probe.json
