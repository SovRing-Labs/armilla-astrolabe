"""
Guard mode with events (ARMILLA M4, CYCLONE-SHIELD-0 D1.3, 2026-10-14).

System-level redundant RNS as in experiments/probe_nodes.py N2: data systems (middle ring + centre) hold the value, two redundant
systems (top + bottom, the largest primes) check it. Arithmetic (+, ·) runs system-wise with no carries. On decode:
  value < data range                         -> verdict +1 (clean)
  one system left out gives a value in range -> verdict  0 (corrected; the faulty system is named)
  otherwise                                  -> verdict -1 (detected, not correctable; no value is returned)
Every decode that is not clean emits an event to every registered sink (default: none; GuardRNS.jsonl_sink(path) appends JSON lines).
A wrong value is never returned silently for up to 2 faulty systems (2 redundant systems).
"""
import json, math, time


def primes_below(n):
    s = bytearray([1]) * n; s[0:2] = b"\x00\x00"
    for i in range(2, int(n ** 0.5) + 1):
        if s[i]: s[i * i::i] = bytearray(len(s[i * i::i]))
    return [i for i in range(n) if s[i]]


def crt(res, mods):
    M = math.prod(mods)
    return sum(r * (M // m) * pow(M // m, -1, m) for r, m in zip(res, mods)) % M


class GuardRNS:
    """k middle systems + centre (data) + top + bottom (redundant); each system = `rings` 16-bit primes, none shared."""

    def __init__(self, k=3, rings=55):
        n = k + 3
        pool = [p for p in primes_below(65536) if p > 2][::-1][:n * rings]
        red = [pool[0:rings], pool[rings:2 * rings]]
        rest = pool[2 * rings:]
        data = [rest[i::k + 1] for i in range(k + 1)]
        self.systems = data + red
        self.Ms = [math.prod(s) for s in self.systems]
        self.nd = k + 1
        self.MD = math.prod(self.Ms[:self.nd])
        self.sinks = []
        self.stats = {"clean": 0, "corrected": 0, "detected": 0}

    # --- sinks -----------------------------------------------------------------------------------------------------------
    def on_event(self, fn):
        self.sinks.append(fn); return fn

    @staticmethod
    def jsonl_sink(path):
        def sink(ev):
            with open(path, "a") as f: f.write(json.dumps(ev) + "\n")
        return sink

    def _emit(self, ev):
        for s in self.sinks: s(ev)

    # --- arithmetic --------------------------------------------------------------------------------------------------------
    def encode(self, x):
        if not 0 <= x < self.MD: raise ValueError("value outside the data range")
        return [x % m for m in self.Ms]

    def add(self, a, b): return [(x + y) % m for x, y, m in zip(a, b, self.Ms)]
    def mul(self, a, b): return [(x * y) % m for x, y, m in zip(a, b, self.Ms)]

    # --- guarded decode ----------------------------------------------------------------------------------------------------
    def decode(self, res, job="", step=None):
        x = crt(res, self.Ms)
        if x < self.MD:
            self.stats["clean"] += 1; return +1, x, None
        for i in range(len(self.Ms)):
            keep = [j for j in range(len(self.Ms)) if j != i]
            y = crt([res[j] for j in keep], [self.Ms[j] for j in keep])
            if y < self.MD:
                self.stats["corrected"] += 1
                self._emit({"t": time.time(), "kind": "armilla.guard", "verdict": 0, "job": job, "step": step, "system": i,
                            "role": "redundant" if i >= self.nd else "data"})
                return 0, y, i
        self.stats["detected"] += 1
        self._emit({"t": time.time(), "kind": "armilla.guard", "verdict": -1, "job": job, "step": step, "system": None})
        return -1, None, None
