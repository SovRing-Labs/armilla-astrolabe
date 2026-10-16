#!/bin/bash
# Interleaved decode timing, Armilla vs bitnet.cpp, same prompt/tokens/threads; load average logged before every run (POD-MIGRATE-1013: bench only at load < 4).
set -euo pipefail
O=${1:?outdir}; W=~/projects/bitnet-work; B=~/projects/bitnet; M=$B/hf_model/model.safetensors; G=$B/models/bitnet-gguf/ggml-model-i2_s.gguf
P=$(cat $O/promptA.ids); N=64; T=${THREADS:-4}; : > $O/timing-quiet.txt
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); threads=$T; ref commit $(git -C $B rev-parse --short HEAD) (llama.cpp build 390c30775); armilla commit $(git -C $(dirname $0) rev-parse --short HEAD 2>/dev/null || echo uncommitted)" >> $O/timing-quiet.txt
for i in 1 2 3 4 5; do
  echo "ref rep$i load=$(cut -d' ' -f1 /proc/loadavg) $($W/ref_decode $G $N $T $P 2>&1 >/dev/null | grep decode)" >> $O/timing-quiet.txt
  echo "arm rep$i load=$(cut -d' ' -f1 /proc/loadavg) $(ABN_SILU=1 ABN_F16KV=1 $W/armilla_bitnet $M $N $T $P 2>&1 >/dev/null | grep '^decode')" >> $O/timing-quiet.txt
done
cat $O/timing-quiet.txt
