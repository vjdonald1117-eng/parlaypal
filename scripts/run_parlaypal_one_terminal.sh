#!/usr/bin/env bash
# Single terminal: [api] + [ui] via npx concurrently.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
"$ROOT/scripts/stop_parlaypal_dev.sh"
cd "$ROOT"
export PYTHONUNBUFFERED=1
echo ""
echo "[Parlay Pal] [api] = uvicorn  [ui] = Vite"
echo "App: http://127.0.0.1:5173/   API: http://127.0.0.1:8000/api/health"
echo ""
exec npx --yes concurrently -k -n api,ui -c cyan,magenta \
  "python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload" \
  "sh -c 'sleep 3 && npm run dev --prefix parlay-ui -- --host 127.0.0.1'"
