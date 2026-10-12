#!/usr/bin/env python3
"""
GPU-shaped pull engine in OpenCL (2026-10-12): correctness of the GPU form, run on PoCL (an OpenCL driver on this CPU; NOT a GPU,
so the timing here says nothing about GPU speed). Pass rules fixed before the run.
Mapping: one work-group = 64 rows; the outer-ring answer tables of a tile of groups are copied into __local memory (the GPU's
shared memory) once per work-group; each work-item pulls its row's answers for every ring from local memory; sums stay per ring
(ushort, reduced every tile); the 3 data rings and the check ring are all pulled; the host reads out by fractional CRT.
GPU1 forward exact vs int64 (5-trit codes, THEAD P3 digit order), check ring agrees.     PASS = 0 mismatches
GPU2 backward (W^T codes) exact.                                                       PASS = 0 mismatches
Writes pull_opencl.result.json
"""
import json, os, time
import numpy as np, pyopencl as cl
HERE = os.path.dirname(os.path.abspath(__file__))
KSRC = r"""
#define R 4
#define TILE 16
__kernel void pull5(__global const uchar *codes,   /* [G5][O] */
                    __global const uchar *lut,     /* [G5][R][256] */
                    const int G5, const int O, __constant ushort *mods,
                    __global uchar *out,           /* [R][O] */
                    __local uchar *ltab)           /* TILE * R * 256 */
{
    int row = get_global_id(0), lid = get_local_id(0), lsz = get_local_size(0);
    uint acc[R]; for (int r = 0; r < R; r++) acc[r] = 0;
    for (int q0 = 0; q0 < G5; q0 += TILE) {
        int nq = min(TILE, G5 - q0);
        for (int k = lid; k < nq * R * 256; k += lsz) ltab[k] = lut[(size_t)q0 * R * 256 + k];   /* tables -> shared memory */
        barrier(CLK_LOCAL_MEM_FENCE);
        if (row < O)
            for (int q = 0; q < nq; q++) {
                uchar c = codes[(size_t)(q0 + q) * O + row];
                for (int r = 0; r < R; r++) acc[r] += ltab[(q * R + r) * 256 + c];               /* the pull */
            }
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int r = 0; r < R; r++) acc[r] %= mods[r];
    }
    if (row < O) for (int r = 0; r < R; r++) out[(size_t)r * O + row] = (uchar)acc[r];
}
"""
ctx = cl.create_some_context(interactive=False); qu = cl.CommandQueue(ctx); prg = cl.Program(ctx, KSRC).build()
MODS = np.array([251, 241, 239, 233], np.uint16); M = 251 * 241 * 239
kf = np.array([pow(M // int(m), -1, int(m)) for m in MODS[:3]], np.float64) / MODS[:3].astype(np.float64)
C5 = np.array([[(p // 3 ** j) % 3 - 1 for j in range(5)] for p in range(243)], np.int64)       # pattern -> 5 trits
def tables(x):
    G5 = len(x) // 5; s = C5 @ x.reshape(G5, 5).T.astype(np.int64)                                 # 243 x G5
    lut = np.zeros((G5, 4, 256), np.uint8)
    for r, m in enumerate(MODS): lut[:, r, :243] = (s % int(m)).T
    return lut
def codes5(W):
    O, K = W.shape; w5 = W.reshape(O, K // 5, 5).astype(np.int64) + 1
    return np.ascontiguousarray(sum(w5[:, :, j] * 3 ** j for j in range(5)).T.astype(np.uint8))
def run(W, x):
    O, K = W.shape; G5 = K // 5; cd = codes5(W); lut = tables(x); out = np.zeros((4, O), np.uint8)
    mf = cl.mem_flags
    bc = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=cd); bl = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=lut)
    bm = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=MODS); bo = cl.Buffer(ctx, mf.WRITE_ONLY, out.nbytes)
    gsz = (O + 63) // 64 * 64
    t0 = time.perf_counter()
    prg.pull5(qu, (gsz,), (64,), bc, bl, np.int32(G5), np.int32(O), bm, bo, cl.LocalMemory(16 * 4 * 256)); qu.finish()
    t = time.perf_counter() - t0
    cl.enqueue_copy(qu, out, bo)
    f = (out[:3].T.astype(np.float64) * kf).sum(axis=1) % 1.0
    y = np.rint(f * M).astype(np.int64); y = np.where(2 * y >= M, y - M, y)
    return y, bool(np.all(y % 233 == out[3])), t
rng = np.random.default_rng(5)
res = {"date": "2026-10-12", "device": ctx.devices[0].name, "note": "PoCL CPU device: correctness of the GPU form only"}
for O, K in ((1024, 1020), (4096, 4080)):
    W = rng.choice(np.array([-1, 0, 1], np.int8), size=(O, K))
    x = rng.integers(-127, 128, K, dtype=np.int8); dy = rng.integers(-127, 128, O, dtype=np.int8)
    y, ok, t = run(W, x); WT = np.ascontiguousarray(W.T[:, :O // 5 * 5]); dyc = dy[:O // 5 * 5]
    dx, ok2, t2 = run(WT, dyc)
    res[f"{O}x{K}"] = {"GPU1_forward_mismatches": int(np.sum(y != W.astype(np.int64) @ x.astype(np.int64))), "forward_check_ring_agrees": ok,
                       "GPU2_backward_mismatches": int(np.sum(dx != WT.astype(np.int64) @ dyc.astype(np.int64))), "backward_check_ring_agrees": ok2,
                       "pocl_cpu_ms": [round(t * 1e3, 1), round(t2 * 1e3, 1)]}
    print(O, K, res[f"{O}x{K}"], flush=True)
json.dump(res, open(os.path.join(HERE, "pull_opencl.result.json"), "w"), indent=1)
