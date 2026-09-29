#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
if find_python; then
  echo 'Join the Tello Wi-Fi first. Keep motors OFF and close other drone controllers.'
  echo 'This records for two minutes. Ctrl+C stops early and saves.'
  printf 'Press Return to start, or Ctrl+C to cancel. '
  read -r ignored
  "$PYTHON_BIN" recorder.py --seconds 120
  open sessions
fi
pause_window
