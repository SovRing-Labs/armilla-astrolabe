#!/bin/bash
# ABACUS-NL-1: re-render ring_nl.cu from gpu_nl.py, build + run on the pod (gpu1, RTX A6000, sm_86), pull outputs, check G1/G2.
# usage: run_pod.sh DIR   (DIR = gpu_nl.py prep output; inputs + CPU twin already there)
set -euo pipefail
D=$(readlink -f "$1"); HERE=$(dirname "$(readlink -f "$0")")
PY=${PY:-python3}
POD=${POD:?set POD=user@host}; POD_PORT=${POD_PORT:-22}; POD_KEY=${POD_KEY:-$HOME/.ssh/id_ed25519}
SSH="ssh -i $POD_KEY -p $POD_PORT $POD"
SCP="scp -i $POD_KEY -P $POD_PORT"
REMOTE=/root/nl1
$PY "$HERE/gpu_nl.py" cu "$D"
$SSH "mkdir -p $REMOTE"
$SCP "$D/ring_nl.cu" $POD:$REMOTE/
for s in small school; do
  $SSH "mkdir -p $REMOTE/$s"
  [ -n "$($SSH "ls $REMOTE/$s/W.bin 2>/dev/null")" ] || $SCP "$D/$s/A.bin" "$D/$s/W.bin" "$D/$s/tgt.bin" $POD:$REMOTE/$s/
done
$SSH "cd $REMOTE && /usr/local/cuda/bin/nvcc -O3 -arch=sm_86 -o ring_nl ring_nl.cu 2>&1 | grep -v Warning | head -20"
: > "$D/gpu.json.new"; echo "{" > "$D/gpu.json.new"; first=1
for s in small school; do
  r=$($PY - "$D/meta.json" "$s" <<'EOF'
import json,sys; m=json.load(open(sys.argv[1]))[sys.argv[2]]; print(m["rows"],m["K"],m["V"],m["cs"])
EOF
)
  out=$($SSH "cd $REMOTE && ./ring_nl $s $r 3" | tail -1)
  echo "$s: $out"
  [ $first = 1 ] || echo "," >> "$D/gpu.json.new"; first=0
  echo "\"$s\": $out" >> "$D/gpu.json.new"
  $SCP "$POD:$REMOTE/$s/out_*.bin" "$D/$s/" >/dev/null
done
echo "}" >> "$D/gpu.json.new"; mv "$D/gpu.json.new" "$D/gpu.json"
$PY "$HERE/gpu_nl.py" check "$D" | grep -E '"(G1_mismatches_per_run|G2_runs_identical|verdict)"' -A4 | head -40
