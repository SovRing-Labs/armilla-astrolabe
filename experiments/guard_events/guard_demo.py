#!/usr/bin/env python3
"""
GUARD-EVENTS-0 (CYCLONE-SHIELD-0 D1.3, 2026-10-14): guard mode as a security / reliability sensor.
Pass rules fixed before the run:
  every trial with 0 faulty systems -> verdict +1, value exact
  every trial with 1 faulty system  -> verdict  0, value exact, the faulty system named, one event emitted
  every trial with 2 faulty systems -> verdict -1 (never a wrong value returned), one event emitted
  silent wrong outputs = 0; baseline without guard (data systems only) shows how many faults would have passed silently.
Jobs (the work a GPU / CPU would really do, done system-wise in residues, no carries):
  A "train": exact ternary GEMV accumulate of one training step (weights in {-1,0,1}, int activations, bias shift to stay ≥ 0)
  B "bignum": sum of 8 products of 1,024-bit integers (the inner work of big-number / crypto jobs)
Faults: one random bit flipped in the residue of a random system, after the arithmetic (silent data corruption model).
Writes guard_demo.result.json + guard_events.jsonl beside this file.
"""
import json, os, random, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "engine"))
from guard_mode import GuardRNS, crt

rng = random.Random(1014)
ev_path = os.path.join(HERE, "guard_events.jsonl")
if os.path.exists(ev_path): os.remove(ev_path)
g = GuardRNS(k=3, rings=55)
events = []
g.on_event(events.append); g.on_event(GuardRNS.jsonl_sink(ev_path))


def flip(res, i):
    bad = list(res); b = rng.randrange(max(1, g.Ms[i].bit_length() - 1))
    bad[i] = (bad[i] ^ (1 << b)) % g.Ms[i]
    if bad[i] == res[i]: bad[i] = (bad[i] + 1) % g.Ms[i]
    return bad


def job_train():
    n = 256; w = [rng.choice((-1, 0, 1)) for _ in range(n)]; x = [rng.randrange(0, 1 << 20) for _ in range(n)]
    shift = n << 20                                                   # keeps the sum ≥ 0 inside the range
    ref = sum(a * b for a, b in zip(w, x)) + shift
    acc = g.encode(shift)
    for a, b in zip(w, x): acc = g.add(acc, [(a * b) % m for m in g.Ms])     # signed term: residue of a·b on EVERY system (10-14 fix)
    return acc, ref % g.MD


def job_bignum():
    ps = [(rng.getrandbits(1024), rng.getrandbits(1024)) for _ in range(8)]
    ref = sum(a * b for a, b in ps)
    acc = g.encode(0)
    for a, b in ps: acc = g.add(acc, g.mul(g.encode(a), g.encode(b)))
    return acc, ref


T = 200
res = {}
t0 = time.time()
for name, job in (("train", job_train), ("bignum", job_bignum)):
    st = {"trials_per_fault_count": T, "f0_clean_exact": 0, "f1_corrected_exact": 0, "f1_named_right": 0, "f2_detected": 0,
          "silent_wrong": 0, "no_guard_silent_wrong": 0, "faults_injected": 0}
    for t in range(T):
        acc, ref = job()
        if job is job_train and ref >= g.MD: raise SystemExit("range")
        v, x, _ = g.decode(acc, name, t)
        st["f0_clean_exact"] += v == 1 and x == ref
        st["silent_wrong"] += v == 1 and x != ref
        i = rng.randrange(len(g.Ms)); bad = flip(acc, i); st["faults_injected"] += 1
        v, x, who = g.decode(bad, name, t)
        st["f1_corrected_exact"] += v == 0 and x == ref; st["f1_named_right"] += who == i
        st["silent_wrong"] += v == 1 or (v == 0 and x != ref)
        if i < g.nd:                                                  # without guard: data systems only, plain CRT
            st["no_guard_silent_wrong"] += crt(bad[:g.nd], g.Ms[:g.nd]) != ref
        a, b = rng.sample(range(len(g.Ms)), 2); bad = flip(flip(acc, a), b); st["faults_injected"] += 2
        v, x, _ = g.decode(bad, name, t)
        st["f2_detected"] += v == -1
        st["silent_wrong"] += v != -1 and x != ref
    st["verdict"] = "PASS" if (st["f0_clean_exact"] == T and st["f1_corrected_exact"] == T and st["f1_named_right"] == T
                               and st["f2_detected"] == T and st["silent_wrong"] == 0) else "FAIL"
    res[name] = st
n_ev = sum(1 for _ in open(ev_path))
out = {"date": "2026-10-14", "layout": "6 systems (k=3), 55 rings each, 2 redundant", "data_range_bits": g.MD.bit_length(),
       "jobs": res, "events_emitted": len(events), "events_in_file": n_ev,
       "events_expected": 2 * 2 * T, "events_by_verdict": {str(k): sum(e["verdict"] == k for e in events) for k in (0, -1)},
       "seconds": round(time.time() - t0, 2)}
out["verdict"] = "PASS" if all(r["verdict"] == "PASS" for r in res.values()) and n_ev == len(events) == 4 * T else "FAIL"
json.dump(out, open(os.path.join(HERE, "guard_demo.result.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
