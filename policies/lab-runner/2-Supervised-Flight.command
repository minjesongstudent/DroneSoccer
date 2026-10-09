#!/bin/bash
source "$(dirname "$0")/mac-common.sh"
echo 'REAL FLIGHT TEST: sends takeoff, policy RC, and landing commands.'
echo 'Use the enclosed lab cage, guards fitted, and a supervising operator.'
echo 'Click Take off + find ring + run policy once. LAND NOW or Escape requests landing.'
echo 'Search: rise toward 120 cm, turn, lower toward 80 cm if needed; no horizontal search commands.'
echo 'Do not launch another drone controller/recorder at the same time.'
if [ ! -x .preview-venv/bin/python ]; then
  echo 'Run 1-Setup.command on normal internet first.'
else
  printf 'Type FLIGHT to connect and open the controls (no automatic takeoff): '
  read -r reply
  if [ "$reply" = FLIGHT ]; then
    .preview-venv/bin/python supervised_flight.py --physical-flight
  fi
fi
pause_window
