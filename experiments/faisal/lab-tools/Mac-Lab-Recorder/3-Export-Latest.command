#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
if find_python; then
  "$PYTHON_BIN" export_latest.py
fi
pause_window
