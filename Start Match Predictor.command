#!/bin/bash
# Double-click this file to start the Match Predictor.
# It sets everything up the first time, then just launches the app.

cd "$(dirname "$0")" || exit 1
echo "==============================================="
echo "   Match Predictor - starting up"
echo "==============================================="

# 1. Check Python 3 is installed
if ! command -v python3 >/dev/null 2>&1; then
  echo ""
  echo "Python 3 is not installed on this Mac."
  echo "Please install it from https://www.python.org/downloads/"
  echo "then double-click this file again."
  echo ""
  read -r -p "Press Return to close this window."
  exit 1
fi

# 2. Create a private environment on first run
if [ ! -d ".venv" ]; then
  echo "First-time setup: creating environment..."
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

# 3. Install the required packages (fast if already installed)
echo "Checking dependencies (first run can take a minute)..."
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

# 4. Open the browser once the server is up, then run it
echo ""
echo "Opening http://127.0.0.1:8000 in your browser..."
echo "Leave this window open while you use the app."
echo "To stop the app: close this window or press Control-C."
echo ""
( sleep 4; open "http://127.0.0.1:8000" ) &
python -m uvicorn app.api:app --host 127.0.0.1 --port 8000
