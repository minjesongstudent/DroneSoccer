"""The autopilot may change the MDP in src/contract.py and nothing else."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "hpc"))
from mdp_guard import check  # noqa: E402

ORIGINAL = (ROOT / "src/contract.py").read_text()
REWARD_BODY = "            - R_WALL * on_wall)"


def edited(old, new):
    assert old in ORIGINAL, old
    return ORIGINAL.replace(old, new, 1)


@pytest.mark.parametrize("new", [
    edited("R_PASS = 40.0", "R_PASS = 55.5"),
    edited("GAMMA = 0.995", "GAMMA = 0.99"),
    edited("R_WALL = 1.0", "R_WALL = 1.0\nR_SMOOTH = 0.1   # a new term"),
    edited(REWARD_BODY, "            - R_WALL * on_wall\n"
                        "            - R_SMOOTH * float(np.square(action[0]) + abs(action[3])))").replace(
        "R_WALL = 1.0", "R_WALL = 1.0\nR_SMOOTH = 0.1"),
    edited(REWARD_BODY, "            - R_WALL * on_wall - 0.02 * (1.0 - obs[6]))"),
    edited("# Reward per step", "# A new comment\n# Reward per step"),
])
def test_mdp_changes_are_allowed(new):
    assert check(ORIGINAL, new) == []


def test_sum_is_not_on_the_numpy_whitelist():
    new = edited(REWARD_BODY, "            - R_WALL * on_wall - float(np.sum(action)))")
    assert check(ORIGINAL, new)


@pytest.mark.parametrize("new", [
    edited('CRASHES = ("hit", "wrong_way", "net")', 'CRASHES = ("wrong_way",)'),
    edited("LAPS_MAX_STEPS = 600", "LAPS_MAX_STEPS = 1200"),
    edited("OBS_LAGS = (24, 20, 16, 12, 8, 4, 2, 1, 0)", "OBS_LAGS = (8, 4, 2, 1, 0)"),
    edited("    ratio = minor / major", "    ratio = minor / major * 1.1"),
    edited("import cv2", "import cv2\nimport os"),
    edited("R_PASS = 40.0", "R_PASS = '40'"),
    edited("R_PASS = 40.0", "R_PASS = float('inf')"),
    edited("GAMMA = 0.995", "GAMMA = 1.5"),
    edited("R_PASS = 40.0", "R_PASS = __import__('os').system('true')"),
    edited("def reward(outcome: str,", "def reward(outcome: str, extra=None,"),
    edited(REWARD_BODY, "            - R_WALL * on_wall)\n    import os"),
    edited(REWARD_BODY, "            - R_WALL * on_wall + open('/tmp/x', 'w').write('x'))"),
    edited(REWARD_BODY, "            - R_WALL * on_wall + np.save('/tmp/x', obs))"),
    edited(REWARD_BODY, "            - R_WALL * on_wall + len(obs.__class__.__name__))"),
    edited(REWARD_BODY, "            - R_WALL * on_wall + LAPS_MAX_STEPS)"),
    edited(REWARD_BODY, "            - R_WALL * on_wall)\n\n\nFIELDS2 = 1"),
    "this is not python (",
])
def test_everything_else_is_rejected(new):
    assert check(ORIGINAL, new)


def test_removing_reward_is_rejected():
    start = ORIGINAL.index("def reward(")
    end = ORIGINAL.index("CRASHES = (")
    assert check(ORIGINAL, ORIGINAL[:start] + ORIGINAL[end:])


# --- the hook, as Claude Code calls it ---------------------------------------------

@pytest.fixture
def sandbox(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    (tmp_path / "contract.before.py").write_text(ORIGINAL)
    (work / "contract.py").write_text(ORIGINAL)
    return work


def hook(work, tool, **tool_input):
    call = {"tool_name": tool, "tool_input": tool_input, "cwd": str(work)}
    return subprocess.run([sys.executable, str(ROOT / "hpc/autopilot_hook.py")],
                          input=json.dumps(call), capture_output=True, text=True,
                          env={"AUTOPILOT_WORK": str(work), "PATH": "/usr/bin:/bin"})


def test_hook_allows_mdp_edits_and_the_proposal(sandbox):
    assert hook(sandbox, "Edit", file_path=str(sandbox / "contract.py"),
                old_string="R_PASS = 40.0", new_string="R_PASS = 50.0").returncode == 0
    assert hook(sandbox, "Write", file_path="proposal.md", content="x").returncode == 0


@pytest.mark.parametrize("tool, args", [
    ("Write", {"file_path": str(ROOT / "src/sim.py"), "content": ""}),
    ("Write", {"file_path": "../../../src/contract.py", "content": ORIGINAL}),
    ("Write", {"file_path": "context/src/sim.py", "content": ""}),
    ("Edit", {"file_path": "contract.py", "old_string": "LAPS_MAX_STEPS = 600",
              "new_string": "LAPS_MAX_STEPS = 900"}),
    ("MultiEdit", {"file_path": "contract.py", "edits": [
        {"old_string": "R_PASS = 40.0", "new_string": "R_PASS = 50.0"},
        {"old_string": "import cv2", "new_string": "import cv2, os"}]}),
    ("Write", {"file_path": "contract.py", "content": ORIGINAL.replace("MIN_RATIO = 0.2", "MIN_RATIO = 0.1")}),
    ("NotebookEdit", {"notebook_path": "x.ipynb"}),
    ("Bash", {"command": "echo hi > ../../../src/sim.py"}),
])
def test_hook_blocks_everything_else(sandbox, tool, args):
    result = hook(sandbox, tool, **args)
    assert result.returncode == 2 and "BLOCKED" in result.stderr


def test_hook_blocks_a_symlink_out_of_the_sandbox(sandbox):
    (sandbox / "link.py").symlink_to(ROOT / "src/sim.py")
    assert hook(sandbox, "Write", file_path="link.py", content="").returncode == 2


def test_hook_refuses_without_a_sandbox(sandbox):
    call = {"tool_name": "Write", "tool_input": {"file_path": "proposal.md"}, "cwd": str(sandbox)}
    result = subprocess.run([sys.executable, str(ROOT / "hpc/autopilot_hook.py")],
                            input=json.dumps(call), capture_output=True, text=True, env={})
    assert result.returncode == 2
