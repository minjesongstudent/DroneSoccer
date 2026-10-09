#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'Camera window setup. Normal internet is needed for this first installation.'
if find_python; then
  if "$PYTHON_BIN" -m venv .preview-venv && .preview-venv/bin/python -m pip install -r preview-requirements.txt && .preview-venv/bin/python -m unittest -v test_recorder.py test_live_preview.py test_calibration.py test_compare_session.py test_flight_control.py test_supervised_flight.py test_pc_policy.py; then
    echo 'CHECKS PASSED. Read START-HERE.txt before a supervised experimental flight.'
  else
    echo 'Camera setup failed. Copy the error text for troubleshooting.'
  fi
fi
pause_window
