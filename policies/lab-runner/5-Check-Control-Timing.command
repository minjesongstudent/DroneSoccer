#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'Offline policy timing check. No camera connection or drone commands.'
if [ ! -x .preview-venv/bin/python ]; then
  echo 'Run 1-Setup.command first.'
else
  mkdir -p checks || exit 1
  CHECK_FILE="checks/timing-$(date +%Y%m%d-%H%M%S)-$$.json"
  .preview-venv/bin/python check_pc_timing.py
fi
pause_window
