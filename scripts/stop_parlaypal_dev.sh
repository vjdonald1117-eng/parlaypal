#!/usr/bin/env bash
# Stop Parlay Pal dev listeners on 8000 (uvicorn) and 5173 (Vite). Safe to run repeatedly.
set -euo pipefail

ports=(8000 5173)

kill_port() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    local pids
    # macOS + Linux: listeners only
    pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
    if [[ -n "${pids:-}" ]]; then
      echo "Stopping listener(s) on port $port: $pids"
      # shellcheck disable=SC2086
      kill $pids 2>/dev/null || true
      sleep 0.2
      pids=$(lsof -nP -iTCP:"$port" -sTCP:LISTEN -t 2>/dev/null || true)
      if [[ -n "${pids:-}" ]]; then
        # shellcheck disable=SC2086
        kill -9 $pids 2>/dev/null || true
      fi
    fi
  elif command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  else
    echo "stop_parlaypal_dev.sh: install lsof or fuser to free ports" >&2
    exit 1
  fi
}

for p in "${ports[@]}"; do
  kill_port "$p"
done

echo "Parlay Pal dev cleanup done (ports ${ports[*]})."
