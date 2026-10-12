#!/usr/bin/env python3
"""
Standalone Desktop & Web Console for Armilla Astrolabe (carry-free ring abacus).
Runs an integrated local server serving the interactive SVG Diamond Visualizer,
Astrolabe Celestial Map, and Evolved Gelosia Tensor Matrix Calculator.
"""

import os
import sys
import json
import http.server
import socketserver
import webbrowser
import threading

# Add engine path
ENGINE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "engine"))
if ENGINE_DIR not in sys.path:
    sys.path.insert(0, ENGINE_DIR)

from diamond_abacus import DiamondOmniringAbacus, T50_PRIMES
from diamond_simd import FastDiamondAbacus
from gelosia_astrolabe import EvolvedGelosiaTracker, AstrolabeSphericalProjector
from omniring_api import omniring_service

HTML_UI = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Armilla Astrolabe — Carry-Free Ring Abacus</title>
<style>
  :root {
    --bg: #070B12;
    --surface: #0F172A;
    --surface-elevated: #1E293B;
    --border: #334155;
    --border-accent: #00F0FF;
    --text: #F8FAFC;
    --muted: #94A3B8;
    --accent: #00F0FF;
    --accent-glow: rgba(0, 240, 255, 0.25);
    --gold: #F59E0B;
    --ruby: #EF4444;
    --emerald: #10B981;
    --violet: #8B5CF6;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, monospace; }
  body { background: var(--bg); color: var(--text); padding: 24px; min-height: 100vh; }
  header { display: flex; justify-content: space-between; align-items: center; padding-bottom: 20px; border-bottom: 1px solid var(--border); margin-bottom: 24px; }
  .badge { background: rgba(0, 240, 255, 0.1); color: var(--accent); border: 1px solid var(--accent); padding: 4px 10px; border-radius: 9999px; font-size: 11px; font-weight: bold; letter-spacing: 1px; }
  .grid-layout { display: grid; grid-template-columns: 1fr 380px; gap: 24px; }
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: 8px; padding: 20px; }
  .card-title { font-size: 15px; font-weight: 700; color: var(--accent); margin-bottom: 12px; display: flex; justify-content: space-between; align-items: center; }
  button.action-btn { background: #0284C7; color: white; border: none; border-radius: 4px; padding: 6px 14px; font-size: 12px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
  button.action-btn:hover { background: #0369A1; }
  input, select { background: var(--surface-elevated); border: 1px solid var(--border); color: var(--text); padding: 6px 10px; border-radius: 4px; font-size: 12px; font-family: monospace; }
</style>
</head>
<body>
  <header>
    <div>
      <h1 style="font-size: 22px; font-weight: 800; display: flex; align-items: center; gap: 10px;">
        <span>🪐 Armilla Astrolabe</span>
        <span class="badge">RING-POWERED SUPERCOMPUTER</span>
      </h1>
      <p style="font-size: 13px; color: var(--muted); margin-top: 4px;">
        Zero-Carry Residue Number System (T50 Coprime Moduli 3..233) & Spherical Astrolabe Coordinate Engine
      </p>
    </div>
    <div style="display: flex; gap: 10px; align-items: center;">
      <span style="font-size: 12px; color: var(--muted);">Moduli Count:</span>
      <select id="sel-rings" onchange="changeRings()">
        <option value="15">15 Rings (Cartesian K3xK5)</option>
        <option value="30">30 Rings (~10^50 Dynamic Range)</option>
        <option value="50" selected>50 Rings (Full T50 ~10^93 States)</option>
      </select>
      <button class="action-btn" onclick="runBench()">⚡ Run 200k Benchmark</button>
    </div>
  </header>

  <div class="grid-layout">
    <div class="card">
      <div class="card-title">
        <span>💠 5-Node Diamond Topology & Systolic Ring Interlock</span>
        <div style="display: flex; gap: 6px;">
          <button class="action-btn" onclick="execCycle('balanced_tensor')">Tensor (N*S + W*E)</button>
          <button class="action-btn" onclick="execCycle('superposition_sum')">Bundle (N+W+S+E)</button>
          <button class="action-btn" onclick="execCycle('cyclic_interlock')">Cut ((N+S)*(W+E))</button>
        </div>
      </div>

      <div style="display: flex; justify-content: center; align-items: center; height: 340px; background: #030712; border-radius: 6px; border: 1px solid var(--border);">
        <svg id="diamond-svg" width="460" height="320" viewBox="0 0 460 320">
          <line x1="230" y1="45" x2="80" y2="160" stroke="#334155" stroke-width="2" stroke-dasharray="4" />
          <line x1="230" y1="45" x2="380" y2="160" stroke="#334155" stroke-width="2" stroke-dasharray="4" />
          <line x1="80" y1="160" x2="230" y2="275" stroke="#334155" stroke-width="2" stroke-dasharray="4" />
          <line x1="380" y1="160" x2="230" y2="275" stroke="#334155" stroke-width="2" stroke-dasharray="4" />

          <line x1="230" y1="45" x2="230" y2="160" stroke="#00F0FF" stroke-width="2" opacity="0.7" />
          <line x1="80" y1="160" x2="230" y2="160" stroke="#00F0FF" stroke-width="2" opacity="0.7" />
          <line x1="380" y1="160" x2="230" y2="160" stroke="#00F0FF" stroke-width="2" opacity="0.7" />
          <line x1="230" y1="275" x2="230" y2="160" stroke="#00F0FF" stroke-width="2" opacity="0.7" />

          <!-- North -->
          <circle cx="230" cy="45" r="32" fill="#0F172A" stroke="#00F0FF" stroke-width="2" />
          <text x="230" y="42" fill="#00F0FF" font-size="12" font-weight="bold" text-anchor="middle">T50_N</text>
          <text id="lbl-n" x="230" y="58" fill="#F8FAFC" font-size="10" text-anchor="middle">val: ...</text>

          <!-- West -->
          <circle cx="80" cy="160" r="32" fill="#0F172A" stroke="#10B981" stroke-width="2" />
          <text x="80" y="157" fill="#10B981" font-size="12" font-weight="bold" text-anchor="middle">T50_W</text>
          <text id="lbl-w" x="80" y="173" fill="#F8FAFC" font-size="10" text-anchor="middle">val: ...</text>

          <!-- East -->
          <circle cx="380" cy="160" r="32" fill="#0F172A" stroke="#F59E0B" stroke-width="2" />
          <text x="380" y="157" fill="#F59E0B" font-size="12" font-weight="bold" text-anchor="middle">T50_E</text>
          <text id="lbl-e" x="380" y="173" fill="#F8FAFC" font-size="10" text-anchor="middle">val: ...</text>

          <!-- South -->
          <circle cx="230" cy="275" r="32" fill="#0F172A" stroke="#8B5CF6" stroke-width="2" />
          <text x="230" y="272" fill="#8B5CF6" font-size="12" font-weight="bold" text-anchor="middle">T50_S</text>
          <text id="lbl-s" x="230" y="288" fill="#F8FAFC" font-size="10" text-anchor="middle">val: ...</text>

          <!-- Center Hub -->
          <circle cx="230" cy="160" r="44" fill="#1E293B" stroke="#EF4444" stroke-width="3" />
          <text x="230" y="156" fill="#EF4444" font-size="13" font-weight="bold" text-anchor="middle">T50_CENTER</text>
          <text id="lbl-c" x="230" y="172" fill="#FFFFFF" font-size="10" font-weight="bold" text-anchor="middle">ACCUMULATOR</text>
        </svg>
      </div>

      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-top: 16px;">
        <div>
          <label style="font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px;">North (N):</label>
          <input id="in-n" type="text" value="987654321098765" style="width: 100%;">
        </div>
        <div>
          <label style="font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px;">West (W):</label>
          <input id="in-w" type="text" value="123456789012345" style="width: 100%;">
        </div>
        <div>
          <label style="font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px;">South (S):</label>
          <input id="in-s" type="text" value="555555555555555" style="width: 100%;">
        </div>
        <div>
          <label style="font-size: 11px; color: var(--muted); display: block; margin-bottom: 4px;">East (E):</label>
          <input id="in-e" type="text" value="222222222222222" style="width: 100%;">
        </div>
      </div>
    </div>

    <div class="card" style="display: flex; flex-direction: column;">
      <div class="card-title">
        <span>🪐 Astrolabe & RNS Inspector</span>
      </div>
      <div id="status-info" style="font-size: 12px; color: var(--muted); margin-bottom: 12px; line-height: 1.4;">
        Active Rings: 50 | Range: 10^93 States<br>Magnitude: fractional CRT + exact fallback
      </div>

      <div style="flex: 1; overflow-y: auto; max-height: 340px; background: #030712; border: 1px solid var(--border); border-radius: 6px; padding: 10px; font-family: monospace; font-size: 11px;">
        <table style="width: 100%; border-collapse: collapse; text-align: right;">
          <thead>
            <tr style="color: var(--muted); border-bottom: 1px solid var(--border);">
              <th style="text-align: left; padding-bottom: 4px;">Ring</th>
              <th style="color: #00F0FF;">N</th>
              <th style="color: #10B981;">W</th>
              <th style="color: #8B5CF6;">S</th>
              <th style="color: #F59E0B;">E</th>
              <th style="color: #EF4444;">Center</th>
            </tr>
          </thead>
          <tbody id="tbody-residues">
            <tr><td colspan="6" style="text-align: center; color: var(--muted); padding: 30px 0;">Select a cycle formula above</td></tr>
          </tbody>
        </table>
      </div>

      <div id="bench-box" style="display: none; margin-top: 12px; padding: 10px; background: rgba(0, 240, 255, 0.05); border: 1px solid rgba(0, 240, 255, 0.2); border-radius: 6px; font-size: 11px; font-family: monospace;">
        <div style="color: var(--accent); font-weight: bold; margin-bottom: 4px;">⚡ Hardware Telemetry:</div>
        <div id="bench-text">...</div>
      </div>
    </div>
  </div>

<script>
async function execCycle(formula) {
  const n = document.getElementById('in-n').value.trim() || '0';
  const w = document.getElementById('in-w').value.trim() || '0';
  const s = document.getElementById('in-s').value.trim() || '0';
  const e = document.getElementById('in-e').value.trim() || '0';

  document.getElementById('lbl-n').textContent = 'val: ' + (n.length > 7 ? n.slice(0, 6) + '..' : n);
  document.getElementById('lbl-w').textContent = 'val: ' + (w.length > 7 ? w.slice(0, 6) + '..' : w);
  document.getElementById('lbl-s').textContent = 'val: ' + (s.length > 7 ? s.slice(0, 6) + '..' : s);
  document.getElementById('lbl-e').textContent = 'val: ' + (e.length > 7 ? e.slice(0, 6) + '..' : e);

  const res = await fetch('/api/compute', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({n, w, s, e, formula})
  });
  const data = await res.json();
  if (data.status === 'ok') {
    document.getElementById('tbody-residues').innerHTML = data.sample_rings.map(r => `
      <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
        <td style="text-align: left; color: var(--muted); padding: 4px 0;">mod ${r.prime}</td>
        <td style="color: #00F0FF;">${r.north}</td>
        <td style="color: #10B981;">${r.west}</td>
        <td style="color: #8B5CF6;">${r.south}</td>
        <td style="color: #F59E0B;">${r.east}</td>
        <td style="color: #EF4444; font-weight: bold;">${r.center}</td>
      </tr>
    `).join('');
  }
}

async function changeRings() {
  const count = parseInt(document.getElementById('sel-rings').value, 10);
  await fetch('/api/configure', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ring_count: count})
  });
  execCycle('balanced_tensor');
}

async function runBench() {
  const box = document.getElementById('bench-box');
  const txt = document.getElementById('bench-text');
  box.style.display = 'block';
  txt.innerHTML = 'Running 200,000 cycles across all prime rings...';
  const res = await fetch('/api/benchmark', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({iterations: 200000})
  });
  const d = await res.json();
  if (d.status === 'ok') {
    txt.innerHTML = `
      Iterations: <strong>${d.iterations.toLocaleString()}</strong><br>
      Throughput: <span style="color: #10B981; font-weight: bold;">${d.cycles_per_sec.toLocaleString()} diamond cycles/sec</span><br>
      Ring Operations: <span style="color: #00F0FF; font-weight: bold;">${d.ring_ops_per_sec.toLocaleString()} ops/sec (${d.effective_mops} MOPS)</span><br>
      Step Latency: <span style="color: #F59E0B;">${d.latency_ns} ns per cycle</span> (Carry-Free)
    `;
  }
}

window.onload = () => { execCycle('balanced_tensor'); };
</script>
</body>
</html>
"""

class ArmillaHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_UI.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        len_h = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(len_h)
        data = json.loads(body.decode("utf-8")) if body else {}

        if self.path == "/api/configure":
            rc = int(data.get("ring_count", 50))
            res = omniring_service.configure_abacus(rc)
            self.send_json(res)
        elif self.path == "/api/compute":
            n = int(data.get("n", 1))
            w = int(data.get("w", 2))
            s = int(data.get("s", 3))
            e = int(data.get("e", 4))
            formula = str(data.get("formula", "balanced_tensor"))
            res = omniring_service.compute_cycle(n, w, s, e, formula)
            self.send_json(res)
        elif self.path == "/api/benchmark":
            iters = int(data.get("iterations", 200000))
            res = omniring_service.benchmark_throughput(iters)
            self.send_json(res)
        else:
            self.send_json({"error": "Not found"}, 404)

    def send_json(self, obj, code=200):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(obj).encode("utf-8"))

def main():
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8989
    server = socketserver.TCPServer(("127.0.0.1", port), ArmillaHandler)
    print(f"[*] Armilla Astrolabe Console running at http://127.0.0.1:{port}")
    server.serve_forever()

if __name__ == "__main__":
    main()
