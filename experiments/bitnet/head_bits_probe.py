#!/usr/bin/env python3
"""ARMILLA-2X-0 lever 1b: bit-plane cascade. Survivors of the rigorous bound |E.x - Q.x| <= |R||x| (R = row quantisation residual) when pass 1 reads only b bits/weight.
usage: head_bits_probe.py MODEL HN.bin OUT.json"""
import json, sys, numpy as np
D = 2560; V = 128256
f = open(sys.argv[1], "rb"); hl = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(hl)); off = 8 + hl
a, b = hdr["model.embed_tokens.weight"]["data_offsets"]
E = (np.fromfile(sys.argv[1], dtype=np.uint16, count=(b - a) // 2, offset=off + a).astype(np.uint32) << 16).view(np.float32).reshape(V, D)
X = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, D); n = len(X); out = {"steps": n}
true = X @ E.T
for bits in (8, 6, 5, 4, 3):
    L = 2 ** (bits - 1) - 1; sv = np.abs(E).max(1, keepdims=True) / L; Q = np.clip(np.rint(E / sv), -L, L) * sv
    Rn = np.sqrt(((E - Q) ** 2).sum(1)); ap = X @ Q.T; xn = np.sqrt((X * X).sum(1))
    bd = Rn[None, :] * xn[:, None] * 1.0001; lo = (ap - bd).max(1, keepdims=True); s = ((ap + bd) >= lo).sum(1)
    out[f"{bits}bit"] = {"mean_surv": float(s.mean()), "max_surv": int(s.max()), "pass1_MB": V * D * bits / 8 / 1e6, "rescore_MB_at_int8": float(s.mean()) * D / 1e6}
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[3], "w"), indent=1)
