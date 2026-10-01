"""The wrapper's own checks around an autopilot Claude run. Claude cannot skip them.

    python hpc/autopilot_guard.py snapshot <out.json>
        Checksum every code file in the repo, before Claude runs.
    python hpc/autopilot_guard.py verify <snapshot.json> <work dir>
        After Claude: no code file in the repo changed, <work>/contract.py passes
        mdp_guard against src/contract.py and actually differs from it, and a
        short rollout with it in a scratch copy of src/ gives finite rewards.
        Prints each failure and exits 1; exits 0 only when all pass.
Run from the repo root. Only after verify passes does autopilot.sh copy the
new contract.py into src/.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
from mdp_guard import check  # noqa: E402

SKIP = ("runs/", "trainingReports/", "trainedModels/", "oldvenv/", ".venv/", ".git/")

SMOKE = """
import numpy as np
from src import train   # settings() reads every constant, so it must still import
from src.sim import HoopEnv
env = HoopEnv(level=1.0, randomise=True)
env.reset(seed=0)
rng = np.random.default_rng(0)
for _ in range(2000):
    _, r, term, trunc, _ = env.step(rng.uniform(-1, 1, 4))
    assert np.isfinite(r) and isinstance(r, float), r
    if term or trunc:
        env.reset()
print(train.settings()[:60])
"""


def code_files() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard"], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout.split("\n")
    files = {f for f in out if f and not f.startswith(SKIP) and ".o" not in Path(f).suffix}
    files |= {str(p.relative_to(ROOT)) for d in ("src", "hpc", "tests", ".claude")
              if (ROOT / d).exists() for p in (ROOT / d).rglob("*")
              if p.is_file() and "__pycache__" not in p.parts}
    return sorted(f for f in files if (ROOT / f).is_file())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(out: Path):
    out.write_text(json.dumps({f: digest(ROOT / f) for f in code_files()}, indent=0))


def verify(snap: Path, work: Path) -> list[str]:
    errors = []
    before = json.loads(snap.read_text())
    now = {f: digest(ROOT / f) for f in code_files()}
    for f in sorted(set(before) | set(now)):
        if before.get(f) != now.get(f):
            errors.append(f"repo file changed during the Claude run: {f}")

    original = (ROOT / "src/contract.py").read_text()
    proposed = (work / "contract.py").read_text()
    errors += check(original, proposed)
    if proposed == original:
        errors.append("contract.py is unchanged: nothing new to run")
    if not (work / "proposal.md").exists():
        errors.append("no proposal.md")
    if errors:
        return errors

    with tempfile.TemporaryDirectory() as tmp:
        shutil.copytree(ROOT / "src", Path(tmp) / "src",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copy(work / "contract.py", Path(tmp) / "src/contract.py")
        run = subprocess.run([str(ROOT / ".venv/bin/python"), "-c", SMOKE], cwd=tmp,
                             capture_output=True, text=True, timeout=600)
        if run.returncode:
            errors.append("smoke rollout failed:\n" + run.stderr[-2000:])
    return errors


if __name__ == "__main__":
    if sys.argv[1] == "snapshot":
        snapshot(Path(sys.argv[2]))
    elif sys.argv[1] == "verify":
        problems = verify(Path(sys.argv[2]), Path(sys.argv[3]))
        for p in problems:
            print(p)
        sys.exit(1 if problems else 0)
    else:
        sys.exit(f"unknown command {sys.argv[1]}")
