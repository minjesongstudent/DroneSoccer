#!/bin/bash
# Latest progress line from every run's log. Run from the repo root:
#   bash hpc/progress.sh [name]            (default name: sac)
#   watch -n 60 bash hpc/progress.sh       (refresh every minute)
NAME=${1:-sac}

for log in runs/${NAME}_s*.log; do
    [ -e "$log" ] || { echo "no logs in runs/ for '$NAME' yet"; exit 1; }
    line=$(grep '^progress' "$log" | tail -1)
    echo "$(basename "$log" .log)  ${line:-starting...}"
done
