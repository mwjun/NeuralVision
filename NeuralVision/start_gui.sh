#!/usr/bin/env bash
# Launches the NeuralVision GUI using the project venv (same folder as this script).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
PY="$SCRIPT_DIR/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  echo "Missing venv Python: $PY" >&2
  echo "Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi
exec "$PY" -m src.gui
