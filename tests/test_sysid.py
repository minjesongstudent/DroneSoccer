"""src.sysid recovers the sim's own dynamics from a flight it flew, and reads real logs."""
import csv
import dataclasses
from pathlib import Path

import numpy as np
import pytest

from src import sysid
from src.params import Cage, Drone
from src.sim import HoopEnv

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "runs" / "lab"


def _write_flight(folder: Path, drone: Drone, seed: int = 0):
    """A sim flight logged as labcheck's state.csv: rc in force and the Tello's readings."""
    env = HoopEnv(drone=drone, cage=Cage(length=40.0, width=40.0))   # room to fly without the net
    env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    folder.mkdir()
    with open(folder / "state.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t", "phase", "cmd_lr", "cmd_fb", "cmd_ud", "cmd_yaw", "vgx", "vgy", "tof", "pitch",
                    "roll", "yaw"])
        command = np.zeros(4)
        for k in range(600):
            if k % 15 == 0:   # steps held 1.5 s, as labcheck steps and a policy's run in
                command = np.round(rng.uniform(-1, 1, 4) * rng.choice([0.3, 0.6, 1.0]), 2)
                command[3] = 0.0
            env.step(command.astype(np.float32))
            env.pos[:2] = 5.0   # far from the net and the hoop (~20 m out), which would stop it
            rc = np.round(command * np.array(sysid.RC_FULL)).astype(int)
            vx, vy = np.floor(np.abs(env.vel[:2]) * 10) * np.sign(env.vel[:2])
            w.writerow([f"{0.1 * (k + 1):.3f}", "policy", *rc, int(vx), int(vy), int(env.pos[2] * 100), 0, 0, 0])
            env.pos[2] = float(np.clip(env.pos[2], 0.6, 1.9))


def test_recovers_the_sims_own_dynamics(tmp_path):
    true = dataclasses.replace(Drone(), delay_steps=2, tau=0.5, max_speed=0.7, up_exponent=1.5, max_up_speed=0.4)
    _write_flight(tmp_path / "f", true)
    fit = sysid.fit([sysid.load(tmp_path / "f")])
    assert fit.max_speed == pytest.approx(0.7, abs=0.08)
    assert fit.tau == pytest.approx(0.5, abs=0.15)
    assert fit.up_exponent == pytest.approx(1.5, abs=0.25)


def test_a_flight_is_cut_at_its_crash(tmp_path):
    folder = tmp_path / "c"
    folder.mkdir()
    rows = [["t", "phase", "cmd_lr", "cmd_fb", "cmd_ud", "cmd_yaw", "vgx", "vgy", "tof", "pitch", "roll", "yaw"]]
    rows += [[0.1 * k, "policy", 0, 50, 0, 0, 5, 0, 150, 0 if k < 30 else 40, 0, 0] for k in range(50)]
    with open(folder / "state.csv", "w", newline="") as fh:
        csv.writer(fh).writerows(rows)
    assert len(sysid.load(folder).t) == 30


@pytest.mark.skipif(not (LAB / "20261012-185424-policy").exists(), reason="lab logs not checked out")
def test_the_2026_10_12_flights_were_faster_and_weaker_in_descent_than_the_old_sim():
    """The flights that showed the dynamics gap: top speed over the old sim's 0.44 and an
    up/down response well past linear."""
    flights = [sysid.load(p) for p in sorted(LAB.glob("20261012-*-policy"))]
    fit = sysid.fit([f for f in flights if f is not None])
    assert fit.max_speed > 0.44
    assert fit.up_exponent > 1.3
