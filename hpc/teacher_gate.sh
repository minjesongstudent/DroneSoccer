#!/bin/bash
# Before a student trains on a --teacher, test that teacher and report it, whatever
# the result: the student trains anyway. Run by job.pbs, after the teacher's job.
#   bash hpc/teacher_gate.sh <teacher name> <cores>
# Report: trainingReports/<teacher>-check.txt. Never fails, except when no teacher
# model exists at all, which the student could not train without.
set -uo pipefail

NAME=${1:?usage: teacher_gate.sh <teacher name> [cores]}
CORES=${2:-1}
MODEL="trainedModels/${NAME}.zip"
REPORT="trainingReports/${NAME}-check.txt"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p trainingReports trainedModels

{
echo "teacher_check for ${NAME}, $(date)"
# A teacher job that failed a seed exits before copying its best seed: pick from
# the seeds that finished, else the newest checkpoint of any seed.
if [ ! -f "$MODEL" ]; then
    echo "no ${MODEL}: the teacher job did not finish cleanly"
    seeds=$(ls -d runs/${NAME}_s* 2>/dev/null | grep -c '_s[0-9]*$')
    best=$(.venv/bin/python -m src.teacher_check --pick "$NAME" "$seeds" --flights 100 2>/dev/null)
    if [ -n "$best" ] && [ -f "runs/${NAME}_s${best}/model.zip" ]; then
        cp "runs/${NAME}_s${best}/model.zip" "$MODEL"
        echo "using seed ${best}'s model.zip (best worst corner)"
    else
        newest=$(ls -t runs/${NAME}_s*/final.zip runs/${NAME}_s*/checkpoints/model_*_steps.zip 2>/dev/null | head -1)
        if [ -z "$newest" ]; then
            echo "no model or checkpoint of ${NAME} anywhere: the student cannot train"
            exit 1
        fi
        cp "$newest" "$MODEL"
        echo "using the newest checkpoint, ${newest}"
    fi
fi
.venv/bin/python -m src.teacher_check "$MODEL" --workers "$CORES"
echo "teacher_check exit $? (0: every gate passed). The student trains either way."
} 2>&1 | tee "$REPORT"
# The pipeline's status is tee's or the exit 1 above.
[ -f "$MODEL" ]
