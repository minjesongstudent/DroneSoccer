#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
if [ ! -x .preview-venv/bin/python ]; then
  echo 'Run 1-Setup.command first.'
else
  .preview-venv/bin/python replay_latest.py
fi
pause_window
