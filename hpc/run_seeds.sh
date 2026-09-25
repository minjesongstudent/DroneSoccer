#!/bin/bash
# Train one SAC run per seed, all in parallel, one CPU core each.
# Works inside any scheduler's job, or by hand in an interactive session.
#   bash hpc/run_seeds.sh <name> <n_seeds> [steps] [train options...]
# Train options go to every seed, e.g. --randomise or --level 0.75.
# Results: runs/<name>_s<seed>/model.zip, logs in runs/<name>_s<seed>.log
# Exits non-zero if any seed fails, so the job shows as failed.
set -euo pipefail

NAME=${1:?usage: run_seeds.sh <name> <n_seeds> [steps]}
SEEDS=${2:?usage: run_seeds.sh <name> <n_seeds> [steps]}
STEPS=${3:-500000}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

# Each run gets one core; without this PyTorch grabs every core in every run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p runs
pids=()
for seed in $(seq 0 $((SEEDS - 1))); do
    # The +"..." form expands an empty array safely under set -u on old bash.
    .venv/bin/python -m src.train --steps "$STEPS" --name "${NAME}_s${seed}" --seed "$seed" \
        ${EXTRA[@]+"${EXTRA[@]}"} > "runs/${NAME}_s${seed}.log" 2>&1 &
    pids+=($!)
done

# A bare `wait` returns 0 even when a run fails, so wait on each one.
failed=0
for seed in $(seq 0 $((SEEDS - 1))); do
    if ! wait "${pids[$seed]}"; then
        echo "seed $seed FAILED, see runs/${NAME}_s${seed}.log:"
        tail -5 "runs/${NAME}_s${seed}.log"
        failed=$((failed + 1))
    fi
done

echo "pass rates on held-out episodes:"
for seed in $(seq 0 $((SEEDS - 1))); do
    echo "seed $seed  $(grep '^policy' "runs/${NAME}_s${seed}.log" || echo FAILED)"
done
grep -h '^baseline' runs/${NAME}_s*.log | head -1 || true

if [ "$failed" -gt 0 ]; then
    echo "$failed of $SEEDS seeds failed"
    exit 1
fi
