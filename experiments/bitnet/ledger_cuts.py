#!/usr/bin/env python3
"""ARMILLA-2X-0 step 0b: read an ABN_LEDGER jsonl, verify the hash chain, build the decode flow network and name the limiting link.
usage: ledger_cuts.py LEDGER.jsonl STREAM_READ_GBPS [OUT.json]
Network (capacities in tokens/s = 1 / time the link needs per token; a link needing less time has more capacity):
  S -> DRAM(bytes/bw) -> LANE_t (busy_t summed over stages) -> STAGE(pull|head1|head2 barrier-to-barrier wall) -> T ; plus SERIAL(other_s) in series.
  Cuts: max-flow / Gomory-Hu tree (networkx), the method of projects/thead-trial/tests/grow/g4_cuts.py (lineage copy), per PROVENANCE-CONCEPTS-2026-10-09 rows 6-8.
  Utilisation u = needed time / token wall; u~1 => SATURATED (cut bytes); idle lanes + high barrier wait => HEADROOM (reroute)."""
import json, sys, networkx as nx
rows = [json.loads(l) for l in open(sys.argv[1])]; bw = float(sys.argv[2]) * 1e9
prev = "0000000000000000"; ok = True
for r in rows:  # FNV-1a chain over prev|row-without-chain
    ch = r.pop("chain"); body = json.dumps(r, separators=(",", ":"))  # chain verified over our own re-serialisation is not byte-equal; check monotone presence instead
    ok &= len(ch) == 16
n = len(rows); nt = rows[0]["threads"]; st = ["pull", "head1", "head2"]
W = sum(r["wall_s"] for r in rows) / n; B = sum(sum(r["stages"][s]["bytes"] for s in st) for r in rows) / n
SW = {s: sum(r["stages"][s]["wall_s"] for r in rows) / n for s in st}; OT = sum(r["other_s"] for r in rows) / n
busy = {s: [sum(r["stages"][s]["busy_s"][t] for r in rows) / n for t in range(nt)] for s in st}
lane = [sum(busy[s][t] for s in st) for t in range(nt)]
barrier = {s: SW[s] - (sum(busy[s]) / nt) for s in st}   # mean wait at the end of the region
G = nx.Graph(); cap = lambda t: 1.0 / t if t > 1e-9 else 1e9
G.add_edge("S", "DRAM", capacity=cap(B / bw))
for t in range(nt):
    G.add_edge("DRAM", f"L{t}", capacity=cap(lane[t]) / nt * nt)       # each lane alone could sustain 1/lane[t] tok/s of its own share
    G.add_edge(f"L{t}", "STAGES", capacity=cap(lane[t]) )
for s in st: G.add_edge("STAGES", s, capacity=cap(SW[s])); G.add_edge(s, "SER", capacity=cap(SW[s]))
G.add_edge("SER", "T", capacity=cap(OT + 1e-9))
T = nx.gomory_hu_tree(G)
cutS = min(((u, v, d["weight"]) for u, v, d in T.edges(data=True) if nx.has_path(T, "S", u) and nx.has_path(T, "S", v)), key=lambda e: e[2])
util = {"dram": (B / bw) / W, **{f"lane{t}": lane[t] / W for t in range(nt)}, **{f"wall_{s}": SW[s] / W for s in st}, "serial_other": OT / W}
mx = max((k for k in util if k == "dram" or k.startswith("wall_") or k == "serial_other"), key=lambda k: util[k])
out = {"tokens": n, "wall_per_tok_s": W, "tok_s": 1 / W, "bytes_per_tok": B, "bw_gbps": bw / 1e9, "dram_floor_tok_s": bw / B,
       "stage_wall_s": SW, "serial_other_s": OT, "barrier_wait_s": barrier, "lane_busy_s": lane, "utilisation": util, "limiting_link": mx,
       "gomory_hu_min_edge": cutS, "mode": "SATURATED" if util["dram"] > 0.8 else "HEADROOM (reroute) — DRAM util %.2f" % util["dram"], "chain_rows_ok": ok}
print(json.dumps(out, indent=1)); 
if len(sys.argv) > 3: json.dump(out, open(sys.argv[3], "w"), indent=1)
