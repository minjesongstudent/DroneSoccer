"""labcheck builds the policy's input from the Tello as the sim does. labcheck needs
djitellopy and cv2, so this runs where the drone is flown (skipped on the HPC)."""
import csv
from collections import deque
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("djitellopy")
from src import contract, labcheck   # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
FLIGHT = ROOT / "runs" / "lab" / "20261012-185424-policy" / "state.csv"


def _history(rng, n=contract.HISTORY_STEPS):
    frames = deque([rng.uniform(-1, 1, contract.OBS_DIM) for _ in range(n)], maxlen=n)
    heights = deque([rng.uniform(0, 2) for _ in range(n)], maxlen=n)
    motion = deque([contract.motion_reading(rng.uniform(0, 1), rng.uniform(-3, 3)) for _ in range(n)], maxlen=n)
    actions = deque([rng.uniform(-1, 1, contract.ACTION_DIM) for _ in range(n)], maxlen=n)
    return frames, heights, motion, actions


def test_the_430_input_is_the_new_one_without_its_motion():
    """ppo-dagger and teacher3-12s-height still get the input they were trained on."""
    frames, heights, motion, actions = _history(np.random.default_rng(0))
    new = contract.policy_input(frames, heights, motion, actions, True)
    old = labcheck.slots_v1_policy_input(frames, heights, motion, actions, True)
    assert len(old) == labcheck.SLOTS_V1_DIM == 430
    f, h, a, flag = contract.split_input(new)
    slots = old[:-1].reshape(contract.N_SLOTS, -1)
    assert np.allclose(slots[:, :contract.OBS_DIM], f)
    assert np.allclose(slots[:, -contract.ACTION_DIM:], a)
    assert old[-1] == flag


@pytest.mark.skipif(not FLIGHT.exists(), reason="lab logs not checked out")
def test_motion_from_a_real_flight_reads_its_speed_and_turn():
    rows = [r for r in csv.DictReader(open(FLIGHT)) if r["phase"] == "policy"]
    start = float(rows[0]["yaw"])
    for r in rows[::10]:
        state = {k: float(r[k]) for k in ("vgx", "vgy", "yaw")}
        speed, s, c = labcheck.motion_from_state(state, start)
        assert speed == pytest.approx(np.hypot(state["vgx"], state["vgy"]) / 10)
        # Telemetry yaw grows clockwise, the turn counter-clockwise as in the sim.
        turned = (start - state["yaw"] + 180) % 360 - 180
        assert np.degrees(np.arctan2(s, c)) == pytest.approx(turned, abs=1e-3)


@pytest.mark.skipif(not FLIGHT.exists(), reason="lab logs not checked out")
def test_a_right_turn_reads_negative_as_in_the_sim():
    """rc yaw > 0 turns the Tello clockwise; the sim's yaw_cw > 0 lowers sin_turned
    (test_sim), so on the drone it must too."""
    rows = [r for r in csv.DictReader(open(FLIGHT)) if r["phase"] == "policy"]
    start = float(rows[0]["yaw"])
    signs = []
    for i in range(len(rows) - 5):
        if all(float(r["cmd_yaw"]) >= 20 for r in rows[i:i + 5]):
            before = labcheck.motion_from_state({"yaw": float(rows[i]["yaw"])}, start)
            after = labcheck.motion_from_state({"yaw": float(rows[i + 5]["yaw"])}, start)
            turned = np.arctan2(after[1], after[2]) - np.arctan2(before[1], before[2])
            signs.append(np.sign((turned + np.pi) % (2 * np.pi) - np.pi))
    assert signs and np.mean(np.array(signs) < 0) > 0.7
