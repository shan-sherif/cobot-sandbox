#!/usr/bin/env bash
# macOS / Linux shortcut for install.py (Windows: py -3.12 install.py). Same options:
#   ./setup.sh          sandbox (the FR3 / UR5e robot models are already in the repo)
#   ./setup.sh --dev    + pytest (to run the tests)
# Picks python3.12 if available, else python3; override with PYTHON=python3.x ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3.12 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "Python 3 not found. macOS: brew install python@3.12 | Ubuntu: sudo apt install python3 python3-venv"
  exit 1
fi
exec "$PY" install.py "$@"
