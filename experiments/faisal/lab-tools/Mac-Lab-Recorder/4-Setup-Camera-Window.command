#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'Camera window setup. Normal internet is needed for this first installation.'
if find_python; then
  if "$PYTHON_BIN" -m venv .preview-venv && .preview-venv/bin/python -m pip install -r preview-requirements.txt && .preview-venv/bin/python -m unittest -v test_recorder.py test_live_preview.py; then
    echo 'READY. Join drone Wi-Fi and run 5-Record-With-Camera.command.'
  else
    echo 'Camera setup failed. Copy the error text for troubleshooting.'
  fi
fi
pause_window
