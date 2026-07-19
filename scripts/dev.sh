#!/bin/bash
# Dev launcher used by `npm run dev`. Sets up a Python environment (first run
# only), installs dependencies, then starts the app with auto-reload.
set -e
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required (the prediction model runs on Python)."
  echo "Install it from https://www.python.org/downloads/ and run 'npm run dev' again."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "First-time setup: creating Python environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "Installing dependencies (fast if already installed)..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

if [ "$1" = "--setup-only" ]; then
  echo "Setup complete. Run 'npm run dev' to start the app."
  exit 0
fi

echo ""
echo "Starting Match Predictor on http://127.0.0.1:8000  (Ctrl-C to stop)"
( sleep 4; command -v open >/dev/null 2>&1 && open "http://127.0.0.1:8000" || true ) &
exec python -m uvicorn app.api:app --host 127.0.0.1 --port 8000 --reload
