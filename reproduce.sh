#!/usr/bin/env bash
# Reproduce Armilla Astrolabe's published results from a clean checkout.
#   ./reproduce.sh            clean clone of the committed tree -> self-test -> print the results table
#   ./reproduce.sh --rerun    also re-run EVERY benchmark that has a script into the clone (needs numpy, gmpy2, gcc; a bench whose
#                             source needs AVX-512 / IFMA / VBMI is skipped and marked NOT YET TESTED when the CPU lacks the flag;
#                             GPU benches are skipped on CPU machines). RERUN_TIMEOUT=seconds per bench (default 1800).
# The table is read from the COMMITTED *.result.json files (not from this run), so it shows the numbers the documents cite;
# --rerun writes new result files inside the temporary clone only, never into your tree.
set -euo pipefail

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RERUN=0; [ "${1:-}" = "--rerun" ] && RERUN=1
WORK="$(mktemp -d "${TMPDIR:-/tmp}/armilla-repro.XXXXXX")"
LOG="$WORK/build.log"
trap 'echo "(clone kept for inspection: $WORK)"' EXIT

echo "[1/4] Clean clone of committed HEAD ($(git -C "$SRC" rev-parse --short HEAD))"
git clone --quiet "$SRC" "$WORK/armilla"
cd "$WORK/armilla"
if [ -n "$(git -C "$SRC" status --porcelain)" ]; then
    echo "NOTE: $SRC has uncommitted changes; they are NOT in this clone, so the table shows committed files only."
fi

echo "[2/4] Dependencies"
command -v python3 >/dev/null || { echo "ERROR: python3 required" >&2; exit 1; }
python3 -c "import numpy" 2>/dev/null || { echo "ERROR: numpy required (pip install numpy)" >&2; exit 1; }
python3 -c "import gmpy2" 2>/dev/null || echo "  NOTE: gmpy2 missing (pip install gmpy2): the GMP-compared benches will FAIL with --rerun"
echo "  cpu flags: $(grep -m1 -o 'avx512ifma\|avx512vbmi\|avx512f' /proc/cpuinfo 2>/dev/null | sort -u | tr '\n' ' ' || true)"

echo "[3/4] Build + self-test (log: $LOG)"
# No compiled step is needed for the engine; the C kernels are built on demand by the benchmarks that use them.
PYTHONPATH=engine python3 engine/test_harness.py >"$LOG" 2>&1 || { tail -20 "$LOG"; echo "SELF-TEST FAILED" >&2; exit 1; }
if python3 -c "import pytest" 2>/dev/null; then
    PYTHONPATH=engine python3 -m pytest -q engine/tests >>"$LOG" 2>&1 || { tail -20 "$LOG"; echo "PYTEST FAILED" >&2; exit 1; }
    echo "  self-test + pytest: PASS"
else
    echo "  self-test: PASS (pytest not installed; engine/tests skipped = NOT YET TESTED here)"
fi

RERUN_TSV="$WORK/rerun.tsv"; : > "$RERUN_TSV"; export RERUN_TSV
if [ "$RERUN" = 1 ]; then
    T="${RERUN_TIMEOUT:-1800}"
    echo "  --rerun: every benchmark with a script (timeout ${T}s each; output in $LOG)"
    st() { printf '%s\t%s\n' "$1" "$2" >> "$RERUN_TSV"; echo "  $1: $2"; }
    for r in $(find experiments -name '*.result.json' | sort); do
        base="${r%.result.json}"; script="$base.py"; args=()
        case "$r" in
            experiments/gpu/*) st "$r" "skipped (GPU bench; CPU-only run)"; continue ;;
            experiments/bench_gelosia_kernel.*.result.json)
                label="${base##*.}"; script=experiments/bench_gelosia_kernel.py; so="$WORK/libgelosia-$label.so"
                if [ "$label" = avx512-native ]; then
                    grep -qw avx512f /proc/cpuinfo || { st "$r" "NOT YET TESTED here (needs avx512f)"; continue; }
                    cf="-O3 -march=native"
                else cf="-O3 -mavx2"; fi
                gcc $cf -shared -fPIC -o "$so" experiments/gelosia_kernel.c >>"$LOG" 2>&1 || { st "$r" "FAILED (gcc $cf)"; continue; }
                args=("$so" "$label") ;;
        esac
        [ -f "$script" ] || { st "$r" "committed (no script)"; continue; }
        # the CPU feature a bench needs = the intrinsics in the C files its script builds (not words in the script)
        csrc=$(grep -o '[A-Za-z0-9_]*\.c"' "$script" | tr -d '"' | sort -u | sed "s|^|$(dirname "$script")/|" || true)
        need=""
        if [ -n "$csrc" ]; then
            grep -qsE 'permutex2var_epi8|permutexvar_epi8|vperm[it]2b' $csrc && need="avx512vbmi"
            [ -z "$need" ] && grep -qsE 'madd52|vpmadd52' $csrc && need="avx512ifma"
            [ -z "$need" ] && grep -qsE '_mm512_|zmm' $csrc && need="avx512f"
        fi
        if [ -n "$need" ] && ! grep -qw "$need" /proc/cpuinfo; then st "$r" "NOT YET TESTED here (needs $need)"; continue; fi
        before="$(sha256sum "$r" | cut -c1-64)"
        if ( cd "$(dirname "$script")" && timeout "$T" python3 "$(basename "$script")" "${args[@]}" ) >>"$LOG" 2>&1; then
            [ "$(sha256sum "$r" | cut -c1-64)" = "$before" ] && st "$r" "re-run here (file unchanged)" || st "$r" "re-run here"
        else st "$r" "FAILED here (rc $?; see $LOG)"; fi
    done
fi

echo "[4/4] Results table (from $( [ "$RERUN" = 1 ] && echo 'this clone, re-run where possible, else' ) committed result files)"
python3 -I - <<'EOF'
import glob, hashlib, json, os

def get(d, *path):
    for p in path:
        if not isinstance(d, dict) or p not in d:
            return None
        d = d[p]
    return d

ran = {}
if os.environ.get('RERUN_TSV') and os.path.exists(os.environ['RERUN_TSV']):
    for line in open(os.environ['RERUN_TSV']):
        k, _, v = line.rstrip('\n').partition('\t'); ran[k] = v
rows = []
for f in sorted(glob.glob('experiments/**/*.result.json', recursive=True)):
    d = json.load(open(f))
    sha = hashlib.sha256(open(f, 'rb').read()).hexdigest()[:12]
    verdict = d.get('verdict') if isinstance(d, dict) else None
    if isinstance(verdict, dict):
        verdict = '; '.join(f'{k}={v}' for k, v in list(verdict.items())[:2])
    rows.append((f, d.get('date', '-') if isinstance(d, dict) else '-', sha, str(verdict)[:90] if verdict else '-', ran.get(f, 'committed')))
w = max(len(r[0]) for r in rows)
print(f"{'result file'.ljust(w)}  {'date':10}  {'sha256[:12]':12}  {'this run':34}  verdict")
for r in rows:
    print(f"{r[0].ljust(w)}  {r[1]:10}  {r[2]:12}  {r[4][:34]:34}  {r[3]}")
n_run = sum(1 for r in rows if r[4].startswith('re-run here')); n_fail = sum(1 for r in rows if r[4].startswith('FAILED'))
print(f"\n{len(rows)} result files: {n_run} re-run here, {n_fail} failed here, {len(rows) - n_run - n_fail} from the committed tree (date = the date written in the file)")

print('\nHeadline numbers (each read from the file named):')
heads = [
    ('experiments/probes.result.json', ('P1_sign_relu',), 'sign/ReLU: values, wrong'),
    ('experiments/gmp/bench_batch_ct.result.json', ('2048',), '2048-bit modexp vs GMP (ms, mismatches)'),
    ('experiments/tlut/tlut5_bench.result.json', ('8192x7680',), '5-trit pull 8192x7680 (us, mismatches)'),
    ('experiments/tlut/train_step.result.json', ('960',), 'exact training step, 8 harnesses, d=960'),
]
for f, path, label in heads:
    d = json.load(open(f)); v = get(d, *path)
    if f.endswith('probes.result.json'):
        out = {k: v[k] for k in ('values', 'wrong') if k in v}
    elif 'bench_batch_ct' in f:
        out = {'ms_per_modexp': v['ms_per_modexp'], 'mismatches': v['mismatches']}
    elif 'tlut5' in f:
        out = {'us': v['us'], 'mismatches': v['mismatches']}
    else:
        out = {'T2': v['T2'], 'T3_ms': v['T3_ms']}
    print(f"- {label}\n    {f}: {json.dumps(out)}")
EOF
echo "Done. Documents: README.md, TECHNICAL-DISCLOSURE.md. Anything not run here is NOT YET TESTED here, not failed."
