#!/bin/bash
cd "$(dirname "$0")" || exit 1
echo 'Preparing a separate environment for simulation inference. Internet required.'
if python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt; then
  echo 'READY. Double-click 2-Check-Policy.command.'
else
  echo 'Setup failed. Send the error text; do not run the checker until setup succeeds.'
fi
read -r -p 'Press Return to close. ' ignored
