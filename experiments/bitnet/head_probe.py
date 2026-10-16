#!/usr/bin/env python3
"""ARMILLA-2X-0 lever 1 probe: how many head rows survive a rigorous bound, for (a) k-means-style cluster radius bound, (b) PCA-split bound
(E.x = E_P.x_P + E_perp.x_perp, |E_perp.x_perp| <= |E_perp||x_perp|). Real hn vectors from ABN_DUMPHN. usage: head_probe.py MODEL.safetensors HN.bin OUT.json"""
import json, sys, numpy as np
D = 2560; V = 128256
f = open(sys.argv[1], "rb"); hl = int.from_bytes(f.read(8), "little"); hdr = json.loads(f.read(hl)); off = 8 + hl
a, b = hdr["model.embed_tokens.weight"]["data_offsets"]
E = (np.fromfile(sys.argv[1], dtype=np.uint16, count=(b - a) // 2, offset=off + a).astype(np.uint32) << 16).view(np.float32).reshape(V, D)
X = np.fromfile(sys.argv[2], dtype=np.float32).reshape(-1, D); n = len(X)
true = X @ E.T                                    # (n, V) float32 reference
out = {"steps": n}
C = (E.T @ E).astype(np.float64); w, U = np.linalg.eigh(C); U = U[:, ::-1].astype(np.float32); w = w[::-1]
out["energy_top"] = {str(r): float(w[:r].sum() / w.sum()) for r in (64, 128, 256, 512, 1024)}
def surv(approx, bd):
    lo = (approx - bd).max(axis=1, keepdims=True); return ((approx + bd) >= lo).sum(axis=1)
for r in (128, 256, 512, 1024):
    P = U[:, :r]; Ep = E @ P; En = np.sqrt(np.maximum((E * E).sum(1) - (Ep * Ep).sum(1), 0)); Xp = X @ P
    xn = np.sqrt(np.maximum((X * X).sum(1) - (Xp * Xp).sum(1), 0)); ap = Xp @ Ep.T
    s = surv(ap, En[None, :] * xn[:, None] * 1.0001); out[f"pca{r}"] = {"mean_surv": float(s.mean()), "max_surv": int(s.max()), "bytes_stageA_int8": V * r}
for K in (1024, 4096):
    rng = np.random.default_rng(0); cen = E[rng.choice(V, K, replace=False)].copy()
    for it in range(3):
        sc = E @ cen.T - 0.5 * (cen * cen).sum(1)[None, :]; asg = sc.argmax(1)
        for k in range(K):
            m = asg == k
            if m.any(): cen[k] = E[m].mean(0)
    rad = np.zeros(K, np.float32)
    for k in range(K):
        m = asg == k
        if m.any(): rad[k] = np.sqrt(((E[m] - cen[k]) ** 2).sum(1)).max()
    xn = np.sqrt((X * X).sum(1)); ca = X @ cen.T; bdk = rad[None, :] * xn[:, None]
    lower = (true.max(axis=1))  # best case lower bound (oracle): cluster excluded if ub < true max
    keep = (ca + bdk) >= lower[:, None]; cnt = np.array([(np.bincount(asg, minlength=K)[keep[i]]).sum() for i in range(n)])
    out[f"kmeans{K}"] = {"mean_rows_in_kept_clusters(oracle lower bound)": float(cnt.mean()), "mean_radius_over_rownorm": float((rad / np.sqrt((E[np.arange(K)] ** 2).sum(1))).mean())}
print(json.dumps(out, indent=1)); json.dump(out, open(sys.argv[3], "w"), indent=1)
