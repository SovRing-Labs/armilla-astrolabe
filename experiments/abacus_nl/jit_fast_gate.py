#!/usr/bin/env python3
"""ABACUS-NL-2 step 1 gate: the FAST ring primitives (jit_nl.render_prims(fast=True): fused v_sc, dot-product MRC / alpha base extension) must give the
SAME bits as the NL-0 primitives against the integer twin. J1 as in jit_nl.py (kernel == ring_nl Algo on P, E, LSE, LOSS), on the same four configs,
with the fast primitives in the CPU row kernel.  PASS = 0 mismatches.  Writes jit_fast_gate.result.json."""
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jit_nl as J
k = J.Kernel(J.render_prims(False, True) + J.CPU_ROW)
R = {"gate": "ABACUS-NL-2 step 1: fast primitives == integer twin (J1)", "J1": {f"rows{r}_n{n}_spread{s}": J.compare(k, r, n, sd, s)
     for (r, n, sd, s) in [(64, 512, 0, 8.0), (64, 512, 1, 30.0), (200, 40, 2, 25.0), (16, 4096, 3, 4.0)]}}
R["verdict"] = "PASS" if all(v == 0 for v in R["J1"].values()) else "FAIL"
json.dump(R, open(os.path.join(HERE, "jit_fast_gate.result.json"), "w"), indent=1); print(json.dumps(R))
