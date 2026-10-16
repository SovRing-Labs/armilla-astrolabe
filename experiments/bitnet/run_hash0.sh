#!/bin/bash
# ARMILLA-FAST-0 step 0: ReLU² (model card) logits hash, prompt A, 16 tokens, at 1/4/8 threads. usage: run_hash0.sh BIN MODEL IDS OUT
set -euo pipefail
B=$1; M=$2; ID=$(cat $3); O=$4; : > $O
echo "cpu: $(grep -m1 'model name' /proc/cpuinfo | cut -d: -f2 | xargs); bin sha256 $(sha256sum $B | cut -c1-16)" >> $O
for t in 1 4 8; do
  ABN_LOGITS=/tmp/h0_$t.bin $B $M 16 $t $ID > /tmp/h0_$t.tok 2>/tmp/h0_$t.err
  echo "threads=$t logits_sha256=$(sha256sum /tmp/h0_$t.bin | cut -c1-64) tokens_sha256=$(sha256sum /tmp/h0_$t.tok | cut -c1-16) $(grep '^decode' /tmp/h0_$t.err | cut -d'(' -f1)" >> $O
done; cat $O
