#!/bin/bash
# GCP-CPU-0 A5 (ON a VBMI VM as root, after vm-setup.sh SETUP OK and after that VM's vm-tests.sh runner finished — one heavy job at a time).
# 1 build armilla_fast5 (5-trit vpermi2b pull) with VBMI; 2 ABN_CHECK pass (every pull = direct dot of the HF weights, 2 tokens);
# 3 A1 hash protocol (run_hash0.sh, 16 tokens, 1/4/8 threads) at L0 -> must equal 16ba3979...50acae; 4 L5 hash, T5 vs plane 3 (must be equal);
# 5 five interleaved reps, 64 tokens, NT threads: bitnet.cpp ref, armilla_fast L5 (plane 3, 2 bits/weight), armilla_fast5 L5+T5 (1.6 bits/weight);
# 6 bundle opus/work into the results (brings the code home; never pushed).  usage: run_a5_vm.sh OUTDIR [NT=8] [--hash-only]
set -euo pipefail
O=${1:?OUTDIR}; NT=${2:-8}; HO=${3:-}; P=/root/corehost/p1; B=$P/bitcpp; A=/root/armilla-opus; A1=/root/armilla-sonnet-p1
M=$P/model.safetensors; ID=$(cat $B/promptA.ids); mkdir -p $O; cd $B
grep -qw avx512vbmi /proc/cpuinfo || { echo "no avx512vbmi on this CPU" | tee $O/build.log; exit 3; }
{ echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); opus/work $(git -C $A rev-parse --short HEAD)"
  gcc -O2 -mavx2 -mf16c -mavx512f -mavx512bw -mavx512vbmi -mno-fma -ffp-contract=off -fopenmp $A/experiments/tlut/armilla_fast5.c -o $B/armilla_fast5 -lm
  echo "armilla_fast5 sha256 $(sha256sum $B/armilla_fast5 | cut -c1-64)"; echo "vpermt2b/vpermi2b in binary: $(objdump -d $B/armilla_fast5 | grep -cE 'vperm[ti]2b')"
  [ -x $B/armilla_fast ] || gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp $B/armilla_fast.c -o $B/armilla_fast -lm
  echo "armilla_fast sha256 $(sha256sum $B/armilla_fast | cut -c1-64)"; } > $O/build.log 2>&1; cat $O/build.log
ABN_CHECK=1 ABN_T5=1 $B/armilla_fast5 $M 1 $NT 128000 791 > /dev/null 2> $O/check.err
echo "check: $(grep -c 't5 check' $O/check.err) pulls, $(grep 't5 check' $O/check.err | grep -vc 'direct=0 mismatches_vs_hf_weights=0') with mismatches; $(head -1 $O/check.err)" | tee $O/check.txt
ABN_T5=1 bash $A1/experiments/bitnet/run_hash0.sh $B/armilla_fast5 $M $B/promptA.ids $O/hash0.txt
for v in "fast 5 0" "fast5 5 1"; do set -- $v
  ABN_FAST=$2 ABN_T5=$3 ABN_LOGITS=/tmp/a5_$1.bin $B/armilla_$1 $M 16 $NT $ID > /dev/null 2>&1
  echo "L5 $1 T5=$3 logits16 sha256=$(sha256sum /tmp/a5_$1.bin | cut -c1-64)"; done | tee $O/hash-L5.txt
[ "$HO" = --hash-only ] && exit 0      # VM-A (no bitnet.cpp build): hashes only
: > $O/ab.txt
for i in 1 2 3 4 5; do
  echo "ref rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(LD_LIBRARY_PATH=bin ./ref_decode_ppl ggml-model-i2_s.gguf 64 $NT $ID 2>&1 >/dev/null | grep decode | cut -d'(' -f1)" | tee -a $O/ab.txt
  echo "L5 rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_FAST=5 $B/armilla_fast $M 64 $NT $ID 2>&1 >/dev/null | grep ^decode | cut -d'(' -f1)" | tee -a $O/ab.txt
  echo "L5T5 rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_FAST=5 ABN_T5=1 $B/armilla_fast5 $M 64 $NT $ID 2>&1 >/dev/null | grep ^decode | cut -d'(' -f1)" | tee -a $O/ab.txt
done
git -C $A bundle create $O/opus-a5.bundle opus/work
