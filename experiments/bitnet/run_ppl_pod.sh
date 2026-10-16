#!/bin/bash
# OUTREACH-P1-FOLLOWUP: perplexity of Armilla (ReLU² model-card vs SiLU, exact vs f16 KV) and bitnet.cpp libllama on the same ids.
# Runs ON the pod in /root/corehost/p1 (bitcpp/ holds the gguf, libs, ref_decode_ppl, ppl-ids.txt). Stops at first error. usage: run_ppl_pod.sh OUTDIR THREADS
set -euo pipefail
O=${1:?outdir}; T=${2:-8}; P=/root/corehost/p1; mkdir -p "$O"; cd $P/bitcpp
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_bitnet.c -o armilla_bitnet_ppl -lm
{ echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); threads=$T; load=$(cut -d' ' -f1-3 /proc/loadavg); ids sha256 $(sha256sum ppl-ids.txt | cut -c1-64); armilla_bitnet.c sha256 $(sha256sum armilla_bitnet.c | cut -c1-64)"; } > $O/ppl-env.txt
run() { name=$1; shift; echo "== $name load=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $O/ppl-env.txt; env "$@" ABN_PPL=ppl-ids.txt ./armilla_bitnet_ppl $P/model.safetensors 0 $T 0 > $O/$name.txt 2> $O/$name.err; cat $O/$name.txt | tee -a $O/ppl-env.txt; }
run armilla-silu-f16kv ABN_SILU=1 ABN_F16KV=1
run armilla-relu2-exactkv X=1
run armilla-relu2-f16kv ABN_F16KV=1
run armilla-silu-exactkv ABN_SILU=1
echo "== bitnet.cpp libllama load=$(cut -d' ' -f1 /proc/loadavg)" | tee -a $O/ppl-env.txt
REF_PPL=ppl-ids.txt LD_LIBRARY_PATH=$P/bitcpp/bin ./ref_decode_ppl ggml-model-i2_s.gguf 0 $T 0 > $O/bitnetcpp.txt 2> $O/bitnetcpp.err; cat $O/bitnetcpp.txt | tee -a $O/ppl-env.txt
