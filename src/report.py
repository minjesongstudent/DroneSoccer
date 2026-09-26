"""One plain-text report per experiment: settings, results and warning flags.

Run: python -m src.report <name> [--seeds N]      e.g. verticalBar_negative50CrashPenalty_noTimePenalty
Read: cat trainingReports/<name>.txt

Reads only the seed logs, runs/<name>_s<seed>.log, plus runs/<name>.args and
each seed's eval.json. run_seeds.sh calls it when a job ends; run it by hand on a
running job for a partial report. Each experiment writes only its own file, and
writes it whole (tmp file, then rename), so jobs running at once never clash.
Imports nothing heavy, so it is quick on the login node.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
REPORTS = ROOT / "trainingReports"
MODELS = ROOT / "trainedModels"

STALE_MIN = 10          # a log untouched this long with no result belongs to a dead run
LEARNED = 0.1           # passes/min a seed needs to count as having learned something
HOVER_TIMEOUT = 0.5     # this share of timeouts with under LEARNED passes/min is hovering
CRASH_HEAVY = 0.5       # crash share that earns a flag
COLLAPSE = 0.5          # final below this share of the seed's peak is a collapse
TAIL = 0.3              # "the end of training": this share of the progress lines
CLIMBING = 0.1          # rising by this share over the tail: more steps may help
STUCK_TAIL = 0.4        # curriculum level unchanged over this share of steps is stuck
SMOOTH = 10             # progress lines averaged when judging a curve

PAIR = re.compile(r"([a-z_/]+)\s+(-?\d+(?:\.\d+)?)(%?)")
FIRST_PASS = re.compile(r"first pass\s+(never|[\d.]+s)")
PROGRESS = re.compile(r"^progress\s+(\d+)/(\d+)")
LEVEL = re.compile(r"^level\s+([\d.]+) at step (\d+)")

# Result columns, in this order when present. The first two are the headline
# score: passes/min for the laps mission, passed for the older single-pass runs.
COLUMNS = [("passes/min", "pass/min"), ("passed", "passed"), ("crashed", "crashed"),
           ("wall", "wall"), ("hit", "hit"), ("wrong_way", "wrong"),
           ("missed", "missed"), ("timeout", "timeout")]


def pairs(text: str) -> dict[str, float]:
    """`key 1.5  other 20%` as {key: 1.5, other: 0.2}. Tolerant of format changes."""
    return {k: float(v) / (100 if pct else 1) for k, v, pct in PAIR.findall(text)}


def parse_result(line: str) -> dict:
    """A `policy ...` or `baseline ...` line from evaluate.report, either format."""
    body = line.split(None, 1)[1] if " " in line else ""
    first = FIRST_PASS.search(body)
    result = pairs(FIRST_PASS.sub("", body))
    result["first pass"] = first.group(1) if first else None
    return result


def parse_log(path: Path) -> dict:
    lines = path.read_text(errors="replace").splitlines()
    seed = {"settings": [], "progress": [], "levels": [], "policy": None, "baseline": None,
            "error": None, "age_min": (time.time() - path.stat().st_mtime) / 60}
    for line in lines:
        if line.startswith("settings "):
            seed["settings"].append(line[len("settings "):])
        elif m := PROGRESS.match(line):
            seed["progress"].append((int(m.group(1)), int(m.group(2)), pairs(line[m.end():])))
        elif m := LEVEL.match(line):
            seed["levels"].append((int(m.group(2)), float(m.group(1))))
        elif line.startswith("policy "):
            seed["policy"] = parse_result(line)
        elif line.startswith("baseline "):
            seed["baseline"] = parse_result(line)
        elif re.match(r"^[\w.]*(Error|Exception|Interrupt)\b", line):
            seed["error"] = line    # the last line of a traceback says what went wrong
    if seed["policy"]:
        seed["status"] = "done"
    elif seed["error"] or seed["age_min"] >= STALE_MIN:
        seed["status"] = "failed"
    else:
        seed["status"] = "running"
    return seed


def score(result: dict | None) -> float | None:
    """The headline number: passes/min, or the pass share for older runs."""
    if not result:
        return None
    return result.get("passes/min", result.get("passed"))


def curve(seed: dict) -> list[tuple[int, float]]:
    """The training score over steps, from the progress lines."""
    points = []
    for step, _, values in seed["progress"]:
        value = values.get("passes/min", values.get("pass_rate"))
        if value is not None:
            points.append((step, value))
    return points


def smoothed(values: list[float]) -> list[float]:
    return [sum(values[max(0, i - SMOOTH + 1):i + 1]) / len(values[max(0, i - SMOOTH + 1):i + 1])
            for i in range(len(values))]


def final_level(seed: dict) -> float | None:
    if seed["progress"] and "level" in seed["progress"][-1][2]:
        return seed["progress"][-1][2]["level"]
    return seed["levels"][-1][1] if seed["levels"] else None


def best_seed(seeds: dict[int, dict], name: str) -> int | None:
    """run_seeds.sh's pick: eval.json's (ready, passes_per_min). Else the best log score."""
    evals = {}
    for s in seeds:
        path = RUNS / f"{name}_s{s}" / "eval.json"
        if path.exists():
            e = json.loads(path.read_text())
            evals[s] = (e.get("ready", 0.0), e.get("passes_per_min", 0.0))
    if evals:
        return max(evals, key=evals.get)
    scored = {s: score(d["policy"]) for s, d in seeds.items() if score(d["policy"]) is not None}
    return max(scored, key=scored.get) if scored else None


def flags(seeds: dict[int, dict], expected: int | None, best: int | None) -> list[str]:
    out = []
    if expected is not None:
        missing = sorted(set(range(expected)) - set(seeds))
        if missing:
            out.append(f"!! seed {_list(missing)}: no log (never started?)")
    for status, mark in (("failed", "!!"), ("running", "..")):
        found = [s for s, d in seeds.items() if d["status"] == status]
        if found:
            detail = ""
            if status == "failed":
                errors = {d["error"] for d in seeds.values() if d["error"]}
                detail = f": {errors.pop()}" if len(errors) == 1 else " (see logs)"
            out.append(f"{mark} seed {_list(found)}: {status}{detail}")

    base = next((score(d["baseline"]) for d in seeds.values() if d["baseline"]), None)
    if best is not None and base is not None and score(seeds[best]["policy"]) is not None:
        top = score(seeds[best]["policy"])
        if top < base:
            out.append(f"!! best seed below baseline ({top:.2f} vs {base:.2f})")

    done = {s: d for s, d in seeds.items() if d["policy"]}
    laps = {s: d for s, d in done.items() if "passes/min" in d["policy"]}
    hovering = [s for s, d in laps.items()
                if d["policy"].get("timeout", 0) >= HOVER_TIMEOUT and d["policy"]["passes/min"] < LEARNED]
    if hovering:
        out.append(f"!  seed {_list(hovering)}: hovering - mostly timeouts, under {LEARNED} passes/min; "
                   "it learned to avoid the hoop")
    crashy = [s for s, d in done.items() if d["policy"].get("crashed", 0) >= CRASH_HEAVY]
    if crashy:
        out.append(f"!  seed {_list(crashy)}: crash-heavy - {CRASH_HEAVY:.0%}+ of episodes crash")
    if len(laps) > 1:
        learned = [s for s, d in laps.items() if d["policy"]["passes/min"] >= LEARNED]
        if len(learned) == 1:
            out.append(f"!  only seed {learned[0]} learned anything: the result may be luck")

    collapsed, climbing, stuck = [], [], []
    for s, d in seeds.items():
        points = curve(d)
        if len(points) >= 2 * SMOOTH:
            values = smoothed([v for _, v in points])
            peak = max(values)
            peak_step = points[values.index(peak)][0]
            if peak >= LEARNED and values[-1] < COLLAPSE * peak:
                collapsed.append(f"{s} (peak {peak:.2f} at {peak_step:,}, ended {values[-1]:.2f})")
            start = values[int(len(values) * (1 - TAIL))]
            if values[-1] > start * (1 + CLIMBING) and values[-1] - start > 0.05:
                climbing.append(s)
        level = final_level(d)
        if d["progress"] and level is not None and level < 1.0:
            last_step, total = d["progress"][-1][0], d["progress"][-1][1]
            last_change = d["levels"][-1][0] if d["levels"] else 0
            if last_step - last_change >= STUCK_TAIL * total or \
                    (d["status"] != "running" and last_step - last_change >= STUCK_TAIL * last_step):
                stuck.append(f"{s} (never levelled up)" if not d["levels"]
                             else f"{s} (at {level:.2f} since step {last_change:,})")
    if collapsed:
        out.append(f"!  collapse - seed {', '.join(collapsed)}")
    if stuck:
        out.append(f"!  curriculum stuck - seed {', '.join(stuck)}")
    if climbing:
        out.append(f"   seed {_list(climbing)}: still improving at the end - more steps may help")

    starts = {d["settings"][0] for d in seeds.values() if d["settings"]}
    if len(starts) > 1:
        out.append("!  seeds started with different settings (sim.py edited mid-launch?)")
    if any(len(set(_constants(x) for x in d["settings"])) > 1 for d in seeds.values()):
        out.append("!  settings changed on --resume (sim.py edited between runs)")
    return out


def _list(seeds) -> str:
    return ",".join(map(str, seeds))


def _constants(line: str) -> str:
    """A settings line without the per-launch options, to compare across resumes."""
    return " ".join(p for p in line.split() if p.split("=")[0].isupper() or p.startswith("git="))


def _fmt(key: str, value) -> str:
    if value is None:
        return "-"
    if key == "passes/min":
        return f"{value:.2f}"
    return f"{value:.1%}"


def settings_table(line: str | None, args: list[str]) -> list[str]:
    rows = [f"args      {' '.join(args) if args else '(none)'}"]
    if not line:
        return rows + ["          settings not logged (run predates the settings line)"]
    items = [p.split("=", 1) for p in line.split()]
    rows.append("run       " + "  ".join(f"{k} {v}" for k, v in items if not k.isupper() and k not in ("git", "seed")))
    constants = [f"{k:<20}{v}" for k, v in items if k.isupper()]
    width = 34
    for i in range(0, len(constants), 3):
        rows.append("          " + "".join(c.ljust(width) for c in constants[i:i + 3]).rstrip())
    return rows


def results_table(seeds: dict[int, dict], best: int | None) -> list[str]:
    rows = [(str(s) + ("*" if s == best else ""), d["status"], d["policy"], d) for s, d in sorted(seeds.items())]
    base = next((d["baseline"] for d in seeds.values() if d["baseline"]), None)
    present = [(k, h) for k, h in COLUMNS
               if any(r[2] and k in r[2] for r in rows) or (base and k in base)]
    header = f"{'seed':<6}{'status':<9}" + "".join(f"{h:>9}" for _, h in present) \
        + f"{'1st pass':>10}{'level':>7}{'steps':>11}{'time':>7}"
    lines = [header]
    for label, status, result, d in rows:
        result = result or {}
        if not d["policy"] and d["progress"]:   # running: show the training numbers
            result = {"passes/min": d["progress"][-1][2].get("passes/min"),
                      "crashed": d["progress"][-1][2].get("crashed")}
        level = final_level(d)
        steps = f"{d['progress'][-1][0] // 1000}k" if d["progress"] else "-"
        elapsed = d["progress"][-1][2].get("elapsed") if d["progress"] else None
        lines.append(f"{label:<6}{status:<9}" + "".join(f"{_fmt(k, result.get(k)):>9}" for k, _ in present)
                     + f"{result.get('first pass') or '-':>10}"
                     + f"{'-' if level is None else f'{level:.2f}':>7}{steps:>11}"
                     + f"{'-' if elapsed is None else f'{elapsed:.0f}m':>7}")
    if base:
        lines.append(f"{'base':<6}{'':<9}" + "".join(f"{_fmt(k, base.get(k)):>9}" for k, _ in present)
                     + f"{base.get('first pass') or '-':>10}")
    return lines


def build(name: str, expected: int | None = None, job: str = "-") -> str:
    seeds = {}
    for path in RUNS.glob(f"{name}_s*.log"):
        m = re.fullmatch(rf"{re.escape(name)}_s(\d+)\.log", path.name)
        if m:
            seeds[int(m.group(1))] = parse_log(path)
    seeds = dict(sorted(seeds.items()))
    args_file = RUNS / f"{name}.args"
    args = args_file.read_text().split() if args_file.exists() else []
    first = next((d for _, d in sorted(seeds.items()) if d["settings"]), None)
    line = first["settings"][0] if first else None
    git = dict(p.split("=", 1) for p in line.split()).get("git", "?") if line else "?"
    total = max((d["progress"][-1][1] for d in seeds.values() if d["progress"]), default=None)
    counts = f"seeds {len(seeds)}" + (f"/{expected}" if expected else "")
    steps = f"   steps {total:,}" if total else ""
    out = [name, f"job {job}   git {git}   {counts}{steps}",
           f"written {datetime.now():%Y-%m-%d %H:%M}", ""]
    out += ["SETTINGS"] + settings_table(line, args) + [""]
    if not seeds:
        out += ["RESULTS", f"no logs at runs/{name}_s*.log", ""]
        best = None
    else:
        best = best_seed(seeds, name)
        out += ["RESULTS   (* best seed; score is on held-out episodes at level 1)"]
        out += results_table(seeds, best)
        model = MODELS / f"{name}.zip"
        if best is not None:
            out.append(f"best      seed {best}" + (f" -> {model.relative_to(ROOT)}" if model.exists() else ""))
        out.append("")
    found = flags(seeds, expected, best) if seeds else ["!! no seed logs found"]
    out += ["FLAGS"] + [f"  {f}" for f in found or ["none"]]
    return "\n".join(out) + "\n"


def write(name: str, expected: int | None = None) -> Path:
    REPORTS.mkdir(exist_ok=True)
    path = REPORTS / f"{name}.txt"
    old = path.read_text() if path.exists() else ""
    # Outside the job there is no PBS_JOBID, and ./ds report passes no --seeds:
    # keep what the job itself wrote rather than blanking it.
    job = os.environ.get("PBS_JOBID", "").split(".")[0]
    if not job:
        m = re.search(r"^job (\S+)", old, re.M)
        job = m.group(1) if m else "-"
    if expected is None and (m := re.search(r"seeds \d+/(\d+)", old)):
        expected = int(m.group(1))
    new = build(name, expected, job)
    # Only rewrite on a real change, so viewing a report leaves git clean.
    if _without_date(new) != _without_date(old):
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(new)
        os.replace(tmp, path)
    return path


def _without_date(report: str) -> str:
    return re.sub(r"^written .*$", "", report, flags=re.M)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("name")
    parser.add_argument("--seeds", type=int, help="how many seeds were launched, to spot missing logs")
    args = parser.parse_args()
    print(write(args.name, args.seeds).read_text(), end="")
