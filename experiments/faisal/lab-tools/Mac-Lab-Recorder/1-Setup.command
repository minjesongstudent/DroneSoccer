#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'Mac lab recorder: offline setup check'
if find_python; then
  "$PYTHON_BIN" --version
  if "$PYTHON_BIN" -m unittest -v test_recorder.py; then
    echo ''
    echo 'READY: At the lab, join the drone Wi-Fi and open 2-Record.command.'
    echo 'Recording is a motors-off ground test. Actual drone connection is not tested by setup.'
    echo 'Optional: install VLC now to view camera.h264 recordings.'
    echo 'https://www.videolan.org/vlc/download-macosx.html'
  else
    echo 'Setup check failed. Send the Terminal error text for troubleshooting.'
  fi
fi
pause_window
