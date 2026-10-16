#!/bin/bash
# ARMILLA-FAST-0 step 4: certified-head argmax == full bf16 argmax at EVERY step. usage: run_headcheck_pod.sh OUTDIR THREADS NTOK  (on the pod)
set -euo pipefail
O=$1; T=$2; N=$3; P=/root/corehost/p1; mkdir -p $O; cd $P/bitcpp; ID=$(cat promptA.ids)
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_fast.c -o armilla_fast -lm
ABN_FAST=4 ABN_HEADCHECK=1 ./armilla_fast $P/model.safetensors $N $T $ID > $O/headcheck.tok 2> $O/headcheck.err
ABN_FAST=2 ./armilla_fast $P/model.safetensors $N $T $ID > $O/base.tok 2>/dev/null
echo "tokens L4(cert head) sha256=$(sha256sum $O/headcheck.tok | cut -c1-16) base L2 sha256=$(sha256sum $O/base.tok | cut -c1-16)" | tee $O/headcheck.txt
grep -E 'headcheck|HEADCHECK|head surv' $O/headcheck.err | tee -a $O/headcheck.txt
