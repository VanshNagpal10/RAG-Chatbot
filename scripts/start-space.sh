#!/usr/bin/env bash
set -euo pipefail

export PORT="${PORT:-7860}"
export BACKEND_PORT="${BACKEND_PORT:-8000}"
export CHROMA_PERSIST_DIR="${CHROMA_PERSIST_DIR:-/home/node/app/chroma_db}"
export HF_HOME="${HF_HOME:-/home/node/.cache/huggingface}"

mkdir -p "$CHROMA_PERSIST_DIR" "$HF_HOME"

# On hosted Spaces there is no local browser profile, so browser-cookie
# extraction cannot work. Put a Netscape-format cookies.txt value in the
# COOKIES_TXT Space secret if YouTube or Instagram asks for login/cookies.
if [[ -n "${COOKIES_TXT:-}" ]]; then
  printf "%s" "$COOKIES_TXT" > /tmp/cookies.txt
  export COOKIES_PATH=/tmp/cookies.txt
fi

uvicorn app.main:app --host 0.0.0.0 --port "$BACKEND_PORT" &
backend_pid=$!

cleanup() {
  kill "$backend_pid" 2>/dev/null || true
}
trap cleanup EXIT

cd frontend
exec npm run start -- -H 0.0.0.0 -p "$PORT"
