#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
if [ ! -x .preview-venv/bin/python ]; then
  echo 'Run 4-Setup-Camera-Window.command while connected to normal internet first.'
else
  echo 'Join drone Wi-Fi. Motors OFF. Close other drone apps and recorders.'
  printf 'Press Return to start a two-minute recording with camera preview. '
  read -r ignored
  .preview-venv/bin/python live_preview.py --seconds 120
fi
pause_window
