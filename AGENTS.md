# AGENTS.md — armilla-astrolabe

For coding assistants (and people) setting this repository up. Plain steps; every command runs from the repo root.

**Status: pre-alpha, under development.** Interfaces, numbers and file layout can change without notice. Read `STATUS.md` first: what works today (each line names its test and evidence file), what is not yet tested, the known failures and the corrections.

## What it is
Carry-free residue-number-system engine on ordinary CPUs: exact integer arithmetic on coprime prime rings, with a ternary pull engine, an exact training step and RNS modular exponentiation.

## Install and check (3 commands)
```bash
git clone --depth 1 https://github.com/SovRing-Labs/armilla-astrolabe && cd armilla-astrolabe
python3 -m venv .venv && . .venv/bin/activate && python -m pip install -q numpy pytest
PYTHONPATH=engine python3 -m pytest -q engine/tests   # expect: 9 passed
```
Needs: python3>=3.10, numpy, pytest.
Whole suite instead: `bash install.sh --no-gui` (python check → venv → every runnable `suite.json` test, headless; without `--no-gui` it then opens the local results page on 127.0.0.1).
Measured: 9.8s on a fresh clone (Intel i5-8365U laptop, 2026-10-16, exit 0).

GUI: `./install.sh   # self-test, then the console on http://127.0.0.1:8989`

## What the tests mean
`suite.json` lists every claim this repository makes with its verify command, the expected output, what passing means, how the number was obtained and how to use it in your own work (8 tests, in `suite.json` at the repository root). `STATUS.md` says what works today, what is not yet tested and the known failures; the umbrella's `STATUS.md` has one row per repository.

## Where the evidence is
- `README.md`
- `TECHNICAL-DISCLOSURE.md` — the method, as dated prior art
- `CITATION.cff`

## Licence
PolyForm Small Business 1.0.0 OR PolyForm Noncommercial 1.0.0 (free trial for evaluation). Details: `LICENSING.md`.

## Guided setup through MCP
The SovRing umbrella ships an MCP server that explains, checks readiness, clones and runs these tests: https://github.com/SovRing-Labs/SovRing (README → "Paste this to your assistant"). Zero-install docs endpoint: https://gitmcp.io/SovRing-Labs/armilla-astrolabe
