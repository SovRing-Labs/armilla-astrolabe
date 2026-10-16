#!/bin/bash
# OUTREACH-P1-FOLLOWUP step 3 + llama-perplexity: runs ON the pod in /root/corehost/p1/bitcpp. Interleaved decode timing (prompt A, 64 tokens),
# bitnet.cpp libllama vs Armilla (SiLU+f16KV = bitnet.cpp graph; ReLU² = model card), 3 reps, load recorded before every run. usage: run_timing_pod.sh OUTDIR THREADS
set -euo pipefail
O=${1:?outdir}; T=${2:-8}; P=/root/corehost/p1; mkdir -p "$O"; cd $P/bitcpp; N=64; ID=$(cat promptA.ids); M=$P/model.safetensors
gcc -O2 -mavx2 -mf16c -mno-fma -ffp-contract=off -fopenmp armilla_bitnet.c -o armilla_bitnet_ppl -lm
F=$O/timing-pod.txt; echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); threads=$T; nproc=$(nproc); armilla_bitnet.c sha256 $(sha256sum armilla_bitnet.c | cut -c1-16); gguf sha256 $(sha256sum ggml-model-i2_s.gguf | cut -c1-16)" > $F
for i in 1 2 3; do
  echo "ref rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(LD_LIBRARY_PATH=bin ./ref_decode_ppl ggml-model-i2_s.gguf $N $T $ID 2>&1 >$O/ref.tok$i | grep decode)" | tee -a $F
  echo "arm-silu-f16kv rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_SILU=1 ABN_F16KV=1 ./armilla_bitnet_ppl $M $N $T $ID 2>&1 >$O/arm-silu.tok$i | grep '^decode')" | tee -a $F
  echo "arm-relu2 rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(./armilla_bitnet_ppl $M $N $T $ID 2>&1 >$O/arm-relu2.tok$i | grep '^decode')" | tee -a $F
done
echo "== llama-perplexity (bitnet.cpp's own tool) on the same text, -c 512" | tee -a $F
LD_LIBRARY_PATH=bin ./bin/llama-perplexity -m ggml-model-i2_s.gguf -f data/pride-and-prejudice-ch1-slice.txt -c 512 -t $T > $O/llama-perplexity.out 2>&1 || true
grep -E "Final estimate|perplexity:|tokens" $O/llama-perplexity.out | tail -3 | tee -a $F
