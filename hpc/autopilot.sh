#!/bin/bash
# Autopilot: when a training run finishes, have Claude design the next MDP and submit it.
#
# Cron runs this every 20 minutes on the login node (./ds autopilot install). Most
# calls exit at once: a job is still queued or running, or the last run was
# already handled. When the last run has finished:
#   1. a run cut off by its walltime is resumed once, not replaced;
#   2. Claude runs headless in a sandbox, runs/autopilot/work/, and proposes a new
#      contract.py and a proposal.md (see hpc/autopilot_prompt.md);
#   3. hpc/autopilot_guard.py verifies it (repo untouched, MDP-only change, smoke rollout);
#   4. only then is src/contract.py replaced and ./ds submit run by this script.
#
# What stops Claude changing anything but the MDP, all enforced here, not asked for:
#   - tools: Read, Grep, Glob, Edit, Write only. No Bash, no MCP.
#   - it works in a sandbox copy, never the repo; the context copies are read-only.
#   - permission rules (dontAsk) allow editing only work/contract.py and work/proposal.md.
#   - a PreToolUse hook (hpc/autopilot_hook.py) rejects every other write, and any
#     contract.py edit outside R_* / GAMMA / reward(), before it is written.
#   - the post-run guard re-checks all of that and checksums every code file in the repo.
#
# Stop:   touch runs/autopilot/STOP     (./ds autopilot stop)
# Failed: runs/autopilot/FAILED says why; delete it to carry on (./ds autopilot resume).
# Log:    runs/autopilot/autopilot.log; Claude's transcript: runs/autopilot/claude-<run>.jsonl
# Test:   DRY_RUN=1 AUTOPILOT_LAST=<finished run> bash hpc/autopilot.sh
#         runs Claude and the guard on that run and submits nothing. It keeps its
#         state in runs/autopilot-dry/, so it never touches the real autopilot's.
set -euo pipefail
cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.."
# cron starts with a bare PATH.
export PATH="/opt/pbs/bin:$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

DRY_RUN=${DRY_RUN:-}
AP=runs/autopilot${DRY_RUN:+-dry}
W=$AP/work
PY=.venv/bin/python
MAX_RUNS=${MAX_RUNS:-5}          # experiments the autopilot may launch before it stops
SEEDS=4 STEPS=5000000            # fixed here, not chosen by Claude
TRAIN_ARGS=(--randomise)
CLAUDE_TIMEOUT=${CLAUDE_TIMEOUT:-3600}
MAX_ATTEMPTS=3                   # Claude runs per finished run before giving up

mkdir -p "$AP"
touch "$AP/handled" "$AP/launched" "$AP/resumed" "$AP/autopilot.log"
log()  { printf '%s %s\n' "$(date '+%F %T')" "$*" | tee -a "$AP/autopilot.log"; }
fail() { log "FAILED: $*"; printf '%s %s\n' "$(date '+%F %T')" "$*" > "$AP/FAILED"; exit 1; }

exec 9>"$AP/lock"
flock -n 9 || exit 0             # the previous call is still working
[ -e "$AP/STOP" ] && exit 0
[ -e "$AP/FAILED" ] && exit 0

if [ -n "$DRY_RUN" ] && [ -n "${AUTOPILOT_LAST:-}" ]; then
  last=$AUTOPILOT_LAST
else
  # Wait while any of our training jobs is queued, running or exiting.
  jobs=$(qstat -u "$USER" -w 2>/dev/null) || exit 0   # no answer from PBS: try next time
  grep -q ' hoop-' <<<"$jobs" && exit 0
  last=$(cat runs/.last-name 2>/dev/null) || exit 0
fi
grep -qx -- "$last" "$AP/handled" && exit 0
compgen -G "runs/${last}_s*.log" >/dev/null || fail "no logs for the last run, $last"

# A seed without a final evaluation was cut off. Resume it once rather than judge it.
unfinished=$(for f in runs/"${last}"_s*.log; do grep -q '^policy' "$f" || echo "$f"; done)
if [ -n "$unfinished" ]; then
  if grep -qx -- "$last" "$AP/resumed"; then
    fail "$last was resumed once and still did not finish: $(echo $unfinished)"
  fi
  [ -n "$DRY_RUN" ] && { log "DRY RUN: would resume $last"; exit 0; }
  log "$last did not finish ($(echo $unfinished)); resuming it"
  ./ds resume "$last" >>"$AP/autopilot.log" 2>&1 || fail "./ds resume $last failed"
  echo "$last" >> "$AP/resumed"
  exit 0
fi

if [ "$(wc -l < "$AP/launched")" -ge "$MAX_RUNS" ]; then
  log "launched $MAX_RUNS runs, the limit. Stopping; raise MAX_RUNS and delete $AP/STOP to go on."
  touch "$AP/STOP"
  exit 0
fi

attempts_file="$AP/attempts-$last"
attempts=$(cat "$attempts_file" 2>/dev/null || echo 0)
[ "$attempts" -ge "$MAX_ATTEMPTS" ] && fail "Claude failed $MAX_ATTEMPTS times on $last"
echo $((attempts + 1)) > "$attempts_file"
log "run $last finished; asking Claude for the next MDP (attempt $((attempts + 1)))"
"$PY" -m src.report "$last" >/dev/null 2>&1 || true   # make sure the report is final

# --- the sandbox ---------------------------------------------------------------
[ -d "$W" ] && chmod -R u+w "$W" && rm -rf "$W"
mkdir -p "$W"/context/{trainingReports,last_run,contracts,src}
chmod u+w "$AP/contract.before.py" 2>/dev/null || true
cp src/contract.py "$AP/contract.before.py"
cp src/contract.py "$W/contract.py"
cp trainingReports/*.txt "$W/context/trainingReports/"
cp runs/"${last}"_s*.log "$W/context/last_run/"
cp runs/*.contract.py "$W/context/contracts/" 2>/dev/null || true
cp "$AP/autopilot.log" "$W/context/"
cp src/{sim,train,baseline,params,evaluate}.py "$W/context/src/"
cp plan.md "$W/context/"
chmod -R a-w "$W/context" "$AP/contract.before.py"

"$PY" hpc/autopilot_guard.py snapshot "$AP/snapshot.json"

# --- Claude ------------------------------------------------------------------------
hook="$PWD/$PY $PWD/hpc/autopilot_hook.py"
settings='{"hooks":{"PreToolUse":[{"matcher":"Edit|MultiEdit|Write|NotebookEdit|Bash","hooks":[{"type":"command","command":"'"$hook"'"}]}]}}'
prompt=$(cat hpc/autopilot_prompt.md)
if [ -n "$DRY_RUN" ]; then
  prompt+=$'\n\nTEST RUN: the "last run" here is an older experiment, so its MDP may differ from '
  prompt+='contract.py. Treat it as history and propose the next change to the current contract.py anyway.'
fi
transcript="$PWD/$AP/claude-$last.jsonl"
if ! (cd "$W" && AUTOPILOT_WORK="$PWD" timeout "$CLAUDE_TIMEOUT" claude -p "$prompt" \
        --tools Read,Grep,Glob,Edit,Write --strict-mcp-config \
        --permission-mode dontAsk \
        --allowedTools Read Grep Glob "Edit(./contract.py)" "Edit(./proposal.md)" \
        --settings "$settings" \
        --output-format stream-json --verbose) > "$transcript" 2>&1; then
  log "Claude exited with an error (see $transcript); will retry next time"
  exit 1
fi

# --- check, then apply ---------------------------------------------------------------
[ -s "$W/proposal.md" ] || fail "Claude wrote no proposal.md (see $transcript)"
name=$(head -1 "$W/proposal.md" | tr -d '[:space:]')
if [ "$name" = ABORT ]; then
  fail "Claude aborted after $last and wants a human: $(tail -n +2 "$W/proposal.md" | tr '\n' ' ' | cut -c1-400)"
fi
[[ "$name" =~ ^[a-z0-9][a-z0-9-]{2,40}$ ]] || fail "bad run name '$name' in proposal.md"
if [ -e "trainingReports/$name.txt" ] || compgen -G "runs/${name}_s*.log" >/dev/null; then
  fail "run name '$name' is already used"
fi

if ! "$PY" hpc/autopilot_guard.py verify "$AP/snapshot.json" "$W" > "$AP/guard.txt" 2>&1; then
  fail "guard rejected the proposal for $name: $(tr '\n' ' ' < "$AP/guard.txt" | cut -c1-600)"
fi
log "guard passed for $name"

if [ -n "$DRY_RUN" ]; then
  log "DRY RUN: would install the new contract.py and run: ./ds submit $SEEDS $STEPS $name ${TRAIN_ARGS[*]}"
  diff -u src/contract.py "$W/contract.py" | tee -a "$AP/autopilot.log" || true
  exit 0
fi

cp "$W/contract.py" src/contract.py
cp src/contract.py "runs/$name.contract.py"
{
  echo "PROPOSED BY THE AUTOPILOT after run $last (Claude; MDP-only change, verified by hpc/autopilot_guard.py)"
  tail -n +2 "$W/proposal.md"
  echo
  echo "contract.py diff against $last:"
  diff -u "$AP/contract.before.py" src/contract.py | tail -n +3 || true
} > "runs/$name.notes"

out=$(./ds submit "$SEEDS" "$STEPS" "$name" "${TRAIN_ARGS[@]}" 2>&1) || {
  cp "$AP/contract.before.py" src/contract.py
  fail "./ds submit failed, contract.py restored: $out"
}
echo "$last" >> "$AP/handled"
echo "$name" >> "$AP/launched"
log "submitted $name: $(echo "$out" | tr '\n' ' ')"
