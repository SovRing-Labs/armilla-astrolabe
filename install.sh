#!/usr/bin/env bash
# ==============================================================================
# ARMILLA ASTROLABE — installer and console launcher
# Single-command setup and standalone runtime execution
# ==============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$DIR"

echo "=================================================================="
echo "    ARMILLA ASTROLABE — CARRY-FREE RING ABACUS                   "
echo "    T50 Residue Number System & Spherical Astrolabe Engine       "
echo "=================================================================="

echo "[1/4] Checking host runtime dependencies..."
if ! command -v python3 &>/dev/null; then
    echo "ERROR: python3 is required. Please install Python 3.10+." >&2
    exit 1
fi

echo "[2/4] Verifying vector math modules (numpy)..."
python3 -c "import numpy" 2>/dev/null || {
    echo "Installing numpy into environment..."
    pip install numpy || python3 -m pip install numpy
}

echo "[3/4] Running algebraic self-test suite..."
PYTHONPATH="$DIR/engine" python3 "$DIR/engine/test_harness.py"

echo "[4/4] Launching standalone Armilla Astrolabe GUI Console..."
echo "Console starting on port 8989 (or embedded WebView)..."
PYTHONPATH="$DIR/engine" python3 "$DIR/gui/standalone_app.py" "$@"
