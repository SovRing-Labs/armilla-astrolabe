#!/bin/bash
# OUTREACH-P1-DECODE harness: Armilla pull decode vs bitnet.cpp (same token ids, greedy). Stops at first error.
# usage: run_p1.sh OUTDIR   (needs ~/projects/bitnet hf_model + gguf + build, ~/projects/bitnet-work/{armilla_bitnet,ref_decode})
set -euo pipefail
O=${1:?outdir}; mkdir -p "$O"; W=~/projects/bitnet-work; B=~/projects/bitnet
M=$B/hf_model/model.safetensors; G=$B/models/bitnet-gguf/ggml-model-i2_s.gguf; N=64
cd ~; python3 - "$O" <<'PY'
import sys
from tokenizers import Tokenizer
t=Tokenizer.from_file(''+__import__('os').environ.get('BITNET_TOKENIZER','hf_model/tokenizer.json')+'')
P={'A':'User: What is a residue number system, in two sentences?','B':'User: Write a haiku about the sea.','C':'User: List three prime numbers greater than 100 and explain how you know they are prime.'}
for k,v in P.items(): open(f'{sys.argv[1]}/prompt{k}.ids','w').write(' '.join(map(str,t.encode(v+'<|eot_id|>Assistant:').ids)))
PY
for k in A B C; do
  P=$(cat $O/prompt$k.ids)
  REF_LOGITS=$O/ref$k.bin $W/ref_decode $G $N 4 $P > $O/ref$k.tok 2> $O/ref$k.err
  ABN_SILU=1 ABN_F16KV=1 ABN_LOGITS=$O/arm$k.bin $W/armilla_bitnet $M $N 4 $P > $O/arm$k.tok 2> $O/arm$k.err
  ABN_LOGITS=$O/card$k.bin $W/armilla_bitnet $M $N 4 $P > $O/card$k.tok 2> $O/card$k.err      # model-card ReLU^2, exact KV
  cmp -s $O/ref$k.tok $O/arm$k.tok && echo "prompt $k: token stream IDENTICAL to bitnet.cpp ($N tokens)" || echo "prompt $k: token stream DIFFERS from bitnet.cpp"
done | tee $O/token-compare.txt
# thread-count invariance of the Armilla logits hash (prompt A)
for t in 1 2 4 8; do ABN_SILU=1 ABN_F16KV=1 ABN_LOGITS=$O/thr$t.bin $W/armilla_bitnet $M 16 $t $(cat $O/promptA.ids) >/dev/null 2>&1; echo "threads=$t logits_sha256=$(sha256sum $O/thr$t.bin | cut -c1-64)"; done | tee $O/thread-hashes.txt
# interleaved timing, prompt A, 64 tokens, 4 threads: ref, armilla, ref, armilla, ...
for i in 1 2 3; do
  echo "ref   rep$i $($W/ref_decode $G $N 4 $(cat $O/promptA.ids) 2>&1 >/dev/null | grep decode)"
  echo "arm   rep$i $(ABN_SILU=1 ABN_F16KV=1 $W/armilla_bitnet $M $N 4 $(cat $O/promptA.ids) 2>&1 >/dev/null | grep '^decode')"
done | tee $O/timing.txt
echo "loadavg_after $(cat /proc/loadavg)" >> $O/timing.txt
