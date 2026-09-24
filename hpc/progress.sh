#!/bin/bash
# Latest progress line from every run's log. Run from the repo root:
#   bash hpc/progress.sh [name]            (default name: sac)
#   watch -n 60 bash hpc/progress.sh       (refresh every minute)
#
# A log nobody has written to for STALE_MIN minutes belongs to an older run,
# so mark it. The glob cannot tell one job's logs from another's.
NAME=${1:-sac}
STALE_MIN=10

now=$(date +%s)
found=0
for log in runs/${NAME}_s*.log; do
    [ -e "$log" ] || continue
    found=1
    line=$(grep '^progress' "$log" | tail -1)
    age=$(( (now - $(stat -c %Y "$log")) / 60 ))
    if [ "$age" -ge "$STALE_MIN" ]; then
        tag="  [stale, ${age}m old]"
    else
        tag=""
    fi
    echo "$(basename "$log" .log)  ${line:-starting...}$tag"
done
[ "$found" = 1 ] || { echo "no logs in runs/ for '$NAME' yet"; exit 1; }
