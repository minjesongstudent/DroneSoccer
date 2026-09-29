#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Run 1-Setup.command first.'
else
  .venv/bin/python check_policy.py
fi
read -r -p 'Press Return to close. ' ignored
