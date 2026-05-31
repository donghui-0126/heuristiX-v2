#!/bin/sh
# Launch the FastAPI backend in the background, then the Streamlit UI in the
# foreground. Streamlit blocks the container; killing it stops both.
set -e

: "${HX_API_HOST:=0.0.0.0}"
: "${HX_API_PORT:=8000}"
: "${HX_UI_HOST:=0.0.0.0}"
: "${HX_UI_PORT:=8501}"

if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[heuristiX hub] WARNING: OPENAI_API_KEY not set — NL→DSL generation will fail."
fi

cd /app

PYTHONPATH=/app uvicorn hub.api.main:app \
    --host "$HX_API_HOST" --port "$HX_API_PORT" &
API_PID=$!

trap "kill $API_PID 2>/dev/null || true" INT TERM EXIT

export HX_API_URL="http://127.0.0.1:$HX_API_PORT"
exec streamlit run hub/ui/app.py \
    --server.address "$HX_UI_HOST" --server.port "$HX_UI_PORT" \
    --server.headless true --browser.gatherUsageStats false
