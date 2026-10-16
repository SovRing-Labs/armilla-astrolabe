#!/usr/bin/env python3
"""
Armilla — the ring softmax judged by math-core's cemented law (ABACUS-NL-0 step 3, 2026-10-13).

The law is read from the ledger, not restated: ai-stack projects/omniglyph/axioms/mathkb.json,
id "mathkb.softmax.normalised_shift_invariant": stable softmax sums to 1 (1e-9), is shift-invariant (1e-9), and
p_i / p_0 = exp(x_i - x_0) to 1e-9 relative. Its own generator draws the cases (x in [-50, 50], 1-40 entries, c in [-100, 100]).
`sm` in the law's prelude is replaced by the JIT ring kernel (jit_nl.Kernel), so the law runs on the abacus' answers.
Pass rules fixed before the run:
  L1  the ring kernel passes the law on N = 500 generated cases                                  PASS = 500 / 500
  L2  teeth: the same kernel rendered with ln2 off by one part in 10^6 (a wrong exponent split) FAILS the law on some case
Writes law_gate.result.json.
"""
import json, math, os, random, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import jit_nl, ring_nl as RN
from decimal import Decimal

LEDGER = os.path.expanduser("~/ai-stack/projects/omniglyph/axioms/mathkb.json")
LAW_ID = "mathkb.softmax.normalised_shift_invariant"

def load_law():
    d = json.load(open(LEDGER))
    law = next(a for a in d["axioms"] if a["id"] == LAW_ID)
    return law

def as_sm(kernel):
    def sm(x):
        P, E, _, _ = kernel.run(np.array([list(map(float, x))]))
        return [math.ldexp(float(p) / float(RN.S), -int(e)) for p, e in zip(P[0], E[0])]
    return sm

def judge(kernel, law, n, seed):
    ns = {"math": math}; exec(law["prelude"], ns); exec(law["gen"], ns)
    ns["sm"] = as_sm(kernel)                       # the law now judges the abacus
    R = random.Random(seed); fails = []
    for i in range(n):
        x, c = ns["gen"](R, None)
        ns["x"], ns["c"] = x, c
        ok = eval(law["forms"]["python"], ns)
        if not ok: fails.append(i)
    return n - len(fails), fails[:5]

if __name__ == "__main__":
    law = load_law(); N = 500
    good = jit_nl.Kernel()
    passed, f = judge(good, law, N, 7)
    bad_src = good.src.replace(
        "static const u32 C_LN2_2[20] = {" + ", ".join(map(str, jit_nl.res(RN.C_LN2_2))) + "};",
        "static const u32 C_LN2_2[20] = {" + ", ".join(map(str, jit_nl.res(int(Decimal(RN.C_LN2_2) * (1 + Decimal("1e-6")))))) + "};")
    assert bad_src != good.src, "teeth patch did not apply"
    bad = jit_nl.Kernel(bad_src)
    bpassed, bf = judge(bad, law, N, 7)
    R = {"date": "2026-10-13", "law": LAW_ID, "statement": law["statement"], "ledger": LEDGER, "N": N,
         "L1_ring_kernel_pass": passed, "L1_first_fails": f,
         "L2_teeth_broken_ln2_pass": bpassed, "L2_teeth_failed_cases": N - bpassed}
    R["verdict"] = "PASS" if passed == N and bpassed < N else "FAIL"
    json.dump(R, open(os.path.join(HERE, "law_gate.result.json"), "w"), indent=1); print(json.dumps(R, indent=1))
