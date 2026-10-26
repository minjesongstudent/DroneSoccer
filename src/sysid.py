from __future__ import annotations

import csv
import dataclasses
import sys
from pathlib import Path

import numpy as np

from src.params import Drone, Randomise
from src.sim import velocity_command

RC_FULL = (50, 50, 50, 70)
PHASES = ("policy", "baseline", "steps")
UPSET_DEG = 20.0
YAW_JUMP_DEG = 6.0
DELAYS = (0, 1, 2, 3, 4)
TAUS = np.round(np.arange(0.25, 1.01, 0.05), 2)
SPEEDS = np.round(np.arange(0.3, 1.01, 0.025), 3)
UP_SPEEDS = np.round(np.arange(0.25, 0.61, 0.025), 3)
UP_EXPONENTS = np.round(np.arange(0.8, 2.51, 0.05), 2)
MIN_MOVING = 20


@dataclasses.dataclass
class Flight:
    name: str
    t: np.ndarray
    rc: np.ndarray
    speed: np.ndarray
    tof: np.ndarray
    level: np.ndarray


def load(folder: Path) -> Flight | None:
    path = Path(folder) / "state.csv"
    if not path.exists():
        return None
    rows = [r for r in csv.DictReader(open(path)) if r["phase"] in PHASES]
    if len(rows) < 10:
        return None
    f = lambda k: np.array([float(r[k]) for r in rows])
    t, pitch, roll, yaw = f("t"), f("pitch"), f("roll"), f("yaw")
    jump = np.r_[0.0, np.abs((np.diff(yaw) + 180) % 360 - 180)]
    upset = np.flatnonzero((np.abs(pitch) > UPSET_DEG) | (np.abs(roll) > UPSET_DEG) | (jump > YAW_JUMP_DEG))
    end = upset[0] if len(upset) else len(rows)
    rc = np.stack([f("cmd_lr"), f("cmd_fb"), f("cmd_ud"), f("cmd_yaw")], 1)
    sl = slice(0, end)
    return Flight(Path(folder).name, t[sl], rc[sl], np.hypot(f("vgx"), f("vgy"))[sl] / 10, f("tof")[sl],
                  ((np.abs(pitch) < 4) & (np.abs(roll) < 4))[sl])


def _horizontal_error(flight: Flight) -> np.ndarray:
    cmd = flight.rc[:, [0, 1]] / np.array(RC_FULL[:2])
    across = np.maximum(1.0, np.hypot(cmd[:, 0], cmd[:, 1]))
    unit = np.hypot(cmd[:, 0], cmd[:, 1]) / across
    dt = np.r_[np.diff(flight.t), 0.1].clip(0.02, 0.5)
    err = np.zeros((len(DELAYS), len(TAUS), len(SPEEDS)))
    tau, top = TAUS[:, None], SPEEDS[None, :]
    for i, delay in enumerate(DELAYS):
        v = np.full((len(TAUS), len(SPEEDS)), flight.speed[0])
        sq = np.zeros_like(v)
        for k in range(len(flight.t) - 1):
            target = top * unit[max(0, k - delay)]
            v = v + (target - v) * (1.0 - np.exp(-dt[k] / tau))
            sq += (np.floor(v * 10 + 1e-9) / 10 - flight.speed[k + 1]) ** 2
        err[i] = sq / max(1, len(flight.t) - 1)
    return err


def _vertical_windows(flight: Flight, window: float = 1.0, lag: float = 0.3):
    out = []
    for i in range(len(flight.t)):
        a, b = np.searchsorted(flight.t, [flight.t[i] - window, flight.t[i]])
        a2, b2 = np.searchsorted(flight.t, [flight.t[i] - window + lag, flight.t[i] + lag])
        if a < 1 or b - a < 6 or b2 >= len(flight.t) or b2 - a2 < 6:
            continue
        ud, seg = flight.rc[a:b, 2], flight.tof[a2:b2]
        if ud.max() - ud.min() > 8 or not flight.level[a2:b2].all():
            continue
        if (np.abs(np.diff(seg)) > 15).any() or not ((seg > 20) & (seg < 300)).all():
            continue
        out.append((ud.mean() / RC_FULL[2], np.polyfit(flight.t[a2:b2], seg, 1)[0] / 100))
    return out


def _vertical_error(windows) -> np.ndarray | None:
    if len(windows) < 10:
        return None
    c, rate = np.array(windows).T
    drone = Drone()
    err = np.zeros((len(UP_SPEEDS), len(UP_EXPONENTS)))
    for i, up in enumerate(UP_SPEEDS):
        for j, k in enumerate(UP_EXPONENTS):
            d = dataclasses.replace(drone, max_up_speed=float(up), up_exponent=float(k))
            pred = np.array([velocity_command(d, 0.0, 0.0, x, 0.0, 0.0)[0][2] for x in c])
            err[i, j] = np.median(np.abs(pred - rate))
    return err


@dataclasses.dataclass
class Fit:
    delay_steps: int | None = None
    tau: float | None = None
    max_speed: float | None = None
    speed_rms: float | None = None
    max_up_speed: float | None = None
    up_exponent: float | None = None
    windows: int = 0


def fit(flights: list[Flight]) -> Fit:
    out = Fit()
    moving = [f for f in flights if (np.abs(f.rc[:, :2]).max(1) > 5).sum() >= MIN_MOVING]
    if moving:
        lengths = np.array([len(f.t) for f in moving], float)
        err = sum(_horizontal_error(f) * n for f, n in zip(moving, lengths)) / lengths.sum()
        i, j, k = np.unravel_index(np.argmin(err), err.shape)
        out.delay_steps, out.tau, out.max_speed = DELAYS[i], float(TAUS[j]), float(SPEEDS[k])
        out.speed_rms = float(np.sqrt(err[i, j, k]))
    windows = [w for f in flights for w in _vertical_windows(f)]
    out.windows = len(windows)
    verr = _vertical_error(windows)
    if verr is not None:
        i, j = np.unravel_index(np.argmin(verr), verr.shape)
        out.max_up_speed, out.up_exponent = float(UP_SPEEDS[i]), float(UP_EXPONENTS[j])
    return out


def ranges() -> dict:
    r, d, err = Randomise(), Drone(), Randomise().error
    up = err.get("drone.max_up_speed", 0.0)
    return {"delay_steps": r.delay_steps, "tau": r.tau, "max_speed": r.max_speed,
            "max_up_speed": (d.max_up_speed - up, d.max_up_speed + up), "up_exponent": r.up_exponent}


def report(fits: dict[str, Fit]) -> str:
    keys = ("delay_steps", "tau", "max_speed", "max_up_speed", "up_exponent")
    lines = [f"{'flight':28s}" + "".join(f"{k:>14s}" for k in keys) + f"{'speed rms':>11s}{'windows':>9s}"]
    for name, f in fits.items():
        cells = []
        for k in keys:
            v = getattr(f, k)
            lo, hi = ranges()[k]
            flag = "" if v is None or lo - 1e-9 <= v <= hi + 1e-9 else " OUTSIDE"
            cells.append(f"{'-' if v is None else f'{v:g}':>{14 - len(flag)}s}{flag}")
        rms = "-" if f.speed_rms is None else f"{f.speed_rms:.3f}"
        lines.append(f"{name:28s}" + "".join(cells) + f"{rms:>11s}{f.windows:>9d}")
    lines.append(f"{'trained on (Randomise)':28s}" + "".join(f"{f'{lo:g}-{hi:g}':>14s}" for lo, hi in
                                                            (ranges()[k] for k in keys)))
    return "\n".join(lines)


def summary_lines(folder: Path) -> list[str]:
    flight = load(folder)
    if flight is None:
        return []
    f = fit([flight])
    if f.max_speed is None and f.up_exponent is None:
        return []
    out = []
    for k in ("delay_steps", "tau", "max_speed", "max_up_speed", "up_exponent"):
        v = getattr(f, k)
        if v is not None:
            lo, hi = ranges()[k]
            out.append(f"{k} {v:g}" + ("" if lo - 1e-9 <= v <= hi + 1e-9 else " (OUTSIDE the sim's range)"))
    return ["  sysid        " + ", ".join(out)]


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return
    flights = [f for f in (load(Path(a)) for a in argv) if f is not None]
    fits = {f.name: fit([f]) for f in flights}
    if len(flights) > 1:
        fits["all flights"] = fit(flights)
    print(report(fits))


if __name__ == "__main__":
    main(sys.argv[1:])
