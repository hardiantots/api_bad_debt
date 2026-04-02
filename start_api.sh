#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8000}"
UVICORN_RELOAD="${UVICORN_RELOAD:-false}"

echo "============================================"
echo " Bad Debt Early-Warning API"
echo " Starting on http://${API_HOST}:${API_PORT}"
echo "============================================"

if [[ "${UVICORN_RELOAD}" == "true" ]]; then
  exec python -m uvicorn api:app --host "${API_HOST}" --port "${API_PORT}" --reload
fi

exec python -m uvicorn api:app --host "${API_HOST}" --port "${API_PORT}"
