#!/bin/bash
# Train one SAC run per seed, all in parallel, one CPU core each.
# Works inside any scheduler's job, or by hand in an interactive session.
#   bash hpc/run_seeds.sh <name> <n_seeds> [steps]
# Results: runs/<name>_s<seed>/model.zip, logs in runs/<name>_s<seed>.log
set -euo pipefail

NAME=${1:?usage: run_seeds.sh <name> <n_seeds> [steps]}
SEEDS=${2:?usage: run_seeds.sh <name> <n_seeds> [steps]}
STEPS=${3:-200000}

# Each run gets one core; without this PyTorch grabs every core in every run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

mkdir -p runs
for seed in $(seq 0 $((SEEDS - 1))); do
    .venv/bin/python -m src.train --steps "$STEPS" --name "${NAME}_s${seed}" --seed "$seed" \
        > "runs/${NAME}_s${seed}.log" 2>&1 &
done
wait

echo "pass rates on held-out episodes:"
for seed in $(seq 0 $((SEEDS - 1))); do
    echo "seed $seed  $(grep '^policy' "runs/${NAME}_s${seed}.log")"
done
grep -h '^baseline' "runs/${NAME}_s0.log"
