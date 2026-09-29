#!/bin/bash
cd "$(dirname "$0")" || exit 1
if python3 -m venv .venv && .venv/bin/python -m pip install -r requirements.txt && .venv/bin/python -m unittest -v test_bridge.py; then
  echo 'READY for motors-off camera preview. Actual camera detection still needs lab validation.'
else
  echo 'Setup failed. Send the error text.'
fi
read -r -p 'Press Return to close. ' ignored
