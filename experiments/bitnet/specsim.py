#!/usr/bin/env python3
"""ARMILLA-2X-0 lever 4 simulator: exact prompt-lookup speculation on a known greedy continuation. Per pass: draft up to K-1 tokens by n-gram lookup (longest n<=N matching the context tail, latest occurrence),
verify in ONE weight pass; accepted = longest prefix equal to the true tokens, +1 (the pass always yields one true token). usage: specsim.py PROMPT.ids GEN.tok"""
import sys
P = [int(x) for x in open(sys.argv[1]).read().split()]; G = [int(x) for x in open(sys.argv[2]).read().split()]
def draft(ctx, K, N=3):
    for n in range(N, 0, -1):
        if len(ctx) < n + 1: continue
        tail = ctx[-n:]
        for i in range(len(ctx) - n - 1, -1, -1):
            if ctx[i:i + n] == tail: return ctx[i + n:i + n + K - 1]
    return []
for K in (1, 2, 3, 4, 6, 8):
    ctx = list(P); i = 0; passes = 0
    while i < len(G):
        d = draft(ctx, K) if K > 1 else []; acc = 0
        while acc < len(d) and i + acc < len(G) and d[acc] == G[i + acc]: acc += 1
        step = min(acc + 1, len(G) - i); ctx += G[i:i + step]; i += step; passes += 1
    print(f"K={K} passes={passes} tokens/pass={len(G)/passes:.2f}")
