#!/bin/bash
# Train one SAC run per seed, all in parallel, one CPU core each.
# Works inside any scheduler's job, or by hand in an interactive session.
#   bash hpc/run_seeds.sh <name> <n_seeds> [steps] [train options...]
# Train options go to every seed, e.g. --randomise or --level 0.75.
# Results: runs/<name>_s<seed>/model.zip, logs in runs/<name>_s<seed>.log
# Once every seed has finished, only the best seed keeps its model.zip, which is
# also copied to trainedModels/<name>.zip. Every seed's checkpoints and resume
# snapshot are deleted; logs, eval.json and plots stay.
# Either way it ends by writing trainingReports/<name>.txt (see src/report.py).
# Exits non-zero if any seed fails, so the job shows as failed.
set -euo pipefail

NAME=${1:?usage: run_seeds.sh <name> <n_seeds> [steps]}
SEEDS=${2:?usage: run_seeds.sh <name> <n_seeds> [steps]}
STEPS=${3:-500000}
shift $(( $# < 3 ? $# : 3 ))
EXTRA=("$@")

# Each run gets one core; without this PyTorch grabs every core in every run.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

# A resumed run appends to its logs, so each still reads as one run.
RESUME=false
for arg in ${EXTRA[@]+"${EXTRA[@]}"}; do [ "$arg" = --resume ] && RESUME=true; done

mkdir -p runs
pids=()
for seed in $(seq 0 $((SEEDS - 1))); do
    log="runs/${NAME}_s${seed}.log"
    $RESUME || : > "$log"
    # The +"..." form expands an empty array safely under set -u on old bash.
    .venv/bin/python -m src.train --steps "$STEPS" --name "${NAME}_s${seed}" --seed "$seed" \
        ${EXTRA[@]+"${EXTRA[@]}"} >> "$log" 2>&1 &
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

# Settings, results and flags in one file; never fails the job.
report() { .venv/bin/python -m src.report "$NAME" --seeds "$SEEDS" || true; }

if [ "$failed" -gt 0 ]; then
    # Keep every snapshot: ./ds resume needs them all to carry the run on.
    echo "$failed of $SEEDS seeds failed, so nothing was deleted"
    report
    exit 1
fi

# Best by the curriculum's bar (full minute, enough passes), then passes/min.
best=$(.venv/bin/python - "$NAME" "$SEEDS" <<'PY'
import json, sys
name, seeds = sys.argv[1], int(sys.argv[2])
scores = {s: json.load(open(f"runs/{name}_s{s}/eval.json")) for s in range(seeds)}
print(max(scores, key=lambda s: (scores[s]["ready"], scores[s]["passes_per_min"])))
PY
)
echo "best seed $best: keeping runs/${NAME}_s${best}/model.zip, deleting the rest"
mkdir -p trainedModels
cp "runs/${NAME}_s${best}/model.zip" "trainedModels/${NAME}.zip"
echo "copied to trainedModels/${NAME}.zip"
for seed in $(seq 0 $((SEEDS - 1))); do
    rm -rf "runs/${NAME}_s${seed}/checkpoints" "runs/${NAME}_s${seed}/resume"
    [ "$seed" = "$best" ] || rm -f "runs/${NAME}_s${seed}/model.zip"
done
report
