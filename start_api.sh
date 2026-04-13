#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Ensure runtime directories exist (used by PM2 log output).
mkdir -p local_data/logs

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
UVICORN_RELOAD="${UVICORN_RELOAD:-false}"

# Prefer project virtualenv when present; fallback to python3/python.
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "ERROR: Python interpreter not found (.venv/bin/python, python3, python)." >&2
  exit 127
fi

echo "============================================"
echo " Bad Debt Early-Warning API"
echo " Starting on http://${API_HOST}:${API_PORT}"
echo " Python interpreter: ${PYTHON_BIN}"
echo "============================================"

if [[ "${UVICORN_RELOAD}" == "true" ]]; then
  exec "${PYTHON_BIN}" -m uvicorn api:app --host "${API_HOST}" --port "${API_PORT}" --reload
fi

exec "${PYTHON_BIN}" -m uvicorn api:app --host "${API_HOST}" --port "${API_PORT}"
