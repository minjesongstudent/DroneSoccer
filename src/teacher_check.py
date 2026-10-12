"""Is a privileged teacher (train.py --privileged-actor) good enough to teach a student?

    python -m src.teacher_check trainedModels/teacher4.zip [--flights 200]
    python -m src.teacher_check --pick <name> <seeds>     (run_seeds.sh: prints the best seed)

A teacher is judged at its worst, not on average. It flies:
  - each corner of the ranges the lab flights spread over: top speed 0.55 or 0.8,
    up/down exponent 1.0 or 1.6, hoop stretch short or tall, everything else drawn
    as in training;
  - the full random mix, with the dynamics part of its true state hidden (set to the
    middle of the ranges). A teacher that leans on numbers the camera never shows
    teaches a clone that averages them away (teacher3's run-in trap);
  - the full mix with action noise 0.1, as DAgger's mixed rounds fly it.
Every corner must clear GATES. --pick ranks seeds by their worst corner.

Whether a student can copy it is the student's own DAgger rounds: a short run with
--teacher prints each round's clone loss and the clone's flights (see the end of main).
"""
from __future__ import annotations

import argparse
import dataclasses
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from src import actor as actor_module
from src.contract import CRASHES
from src.params import Drone, Hoop, Randomise
from src.sim import HoopEnv

SEED = 2_000_000          # past evaluate's and train's selection seeds
SPEEDS = (0.55, 0.8)
EXPONENTS = (1.0, 1.6)
STRETCH = ("short", "tall")
GATES = {"passes_per_ep": 3.0, "crashed": 0.05, "never_passed": 0.03, "centred": 0.8, "pass_height": 0.03}
NOISE = 0.1
NOISE_CRASHED = 0.08
HIDDEN_DROP = 0.2        # the share of passes it may lose with the dynamics hidden
# The dynamics block of HoopEnv.privileged(): after to_hoop pos (3), vel (3), facing and
# yaw rate (3), pos/cage (3), cage (3), the six layout numbers.
DYN_START = 21
DYN_HIDDEN = {0: "delay_steps", 1: "tau", 2: "max_speed", 3: "max_up_speed", 4: "up_exponent",
              7: "telemetry_lag", 8: "yaw_drift"}


def _hidden_values() -> dict[int, float]:
    r, d = Randomise(), Drone()
    mid = lambda lo_hi: (lo_hi[0] + lo_hi[1]) / 2
    return {DYN_START + i: (d.max_up_speed if name == "max_up_speed" else mid(getattr(r, name)))
            for i, name in DYN_HIDDEN.items()}


class _Hidden:
    """The teacher with its dynamics numbers replaced by the ranges' middles."""

    def __init__(self, policy):
        self.policy, self.values = policy, _hidden_values()

    def reset(self):
        self.policy.reset()

    def bind(self, env):
        real = env.privileged
        def privileged():
            p = real()
            for i, v in self.values.items():
                p[i] = v
            return p
        self.policy.bind(type("Env", (), {"privileged": staticmethod(privileged)})())

    def __call__(self, obs):
        return self.policy(obs)


def corner_env(speed: float | None = None, exponent: float | None = None, stretch: str | None = None) -> HoopEnv:
    """Training's randomised sim with the named values pinned; None leaves one drawn."""
    r = Randomise()
    r = dataclasses.replace(r, max_speed=(speed, speed) if speed else r.max_speed,
                            up_exponent=(exponent, exponent) if exponent else r.up_exponent)
    hoop = Hoop()
    if stretch:
        sign = 1 if stretch == "tall" else -1
        hoop = dataclasses.replace(hoop, outer_height=hoop.outer_height * (1 + sign * hoop.stretch_tolerance),
                                   stretch_tolerance=0.0)
    return HoopEnv(level=1.0, randomise=True, ranges=r, hoop=hoop)


def fly(policy, env: HoopEnv, flights: int, noise: float = 0.0, seed: int = SEED) -> dict:
    """passes_per_ep, crashed, never_passed, median centred, median |pass height| and the
    median speed through the hoop over `flights` flights."""
    rng = np.random.default_rng(seed)
    passes, crashed, centred, height, speed = [], [], [], [], []
    for i in range(flights):
        obs, _ = env.reset(seed=seed + i)
        policy.reset()
        policy.bind(env)
        done, before = False, 0
        while not done:
            action = np.asarray(policy(obs), np.float32)
            if noise:
                action = np.clip(action + rng.normal(0.0, noise, action.shape), -1, 1).astype(np.float32)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if info["passes"] > before:
                before = info["passes"]
                centred.append(info["centred"])
                height.append(abs(info["drone_pos"][2] - info["hoop_centre"][2]))
                speed.append(float(np.linalg.norm(env.vel)))
        passes.append(info["passes"])
        crashed.append(info["outcome"] in CRASHES)
    med = lambda x: float(np.median(x)) if x else float("nan")
    return {"passes_per_ep": float(np.mean(passes)), "crashed": float(np.mean(crashed)),
            "never_passed": float(np.mean(np.array(passes) == 0)), "centred": med(centred),
            "pass_height": med(height), "pass_speed": med(speed)}


def failures(score: dict, crashed_limit: float = GATES["crashed"]) -> list[str]:
    out = []
    if score["passes_per_ep"] < GATES["passes_per_ep"]:
        out.append("passes")
    if score["crashed"] > crashed_limit:
        out.append("crashed")
    if score["never_passed"] > GATES["never_passed"]:
        out.append("never_passed")
    if not score["centred"] >= GATES["centred"]:
        out.append("centred")
    if not score["pass_height"] <= GATES["pass_height"]:
        out.append("height")
    return out


def corners() -> list[tuple]:
    return list(itertools.product(SPEEDS, EXPONENTS, STRETCH))


def _corner_job(args):
    path, corner, flights = args
    return fly(actor_module.load_policy(path), corner_env(*corner), flights)


def check(path: str, flights: int, workers: int = 1, full: bool = True) -> dict:
    """Every corner, and with full also the mix, the hidden-dynamics and noisy flights."""
    jobs = [(path, c, flights) for c in corners()]
    with Pool(workers) as pool:
        scores = dict(zip(corners(), pool.map(_corner_job, jobs)))
    out = {"corners": scores, "worst": min(s["passes_per_ep"] for s in scores.values())}
    if full:
        policy = actor_module.load_policy(path)
        out["mix"] = fly(policy, corner_env(), flights)
        out["hidden"] = fly(_Hidden(policy), corner_env(), flights)
        out["noisy"] = fly(policy, corner_env(), flights, noise=NOISE)
    return out


def _line(name: str, s: dict, fails: list[str]) -> str:
    return (f"{name:28s} passes/ep {s['passes_per_ep']:5.2f}  crashed {s['crashed']:4.0%}  "
            f"never {s['never_passed']:4.0%}  centred {s['centred']:.2f}  |height| {100 * s['pass_height']:4.1f} cm  "
            f"speed {s['pass_speed']:.2f} m/s  {'ok' if not fails else 'FAIL ' + ', '.join(fails)}")


def report(result: dict) -> tuple[str, bool]:
    lines, ok = [], True
    for (speed, exponent, stretch), s in result["corners"].items():
        fails = failures(s)
        ok &= not fails
        lines.append(_line(f"speed {speed} exp {exponent} {stretch}", s, fails))
    if "mix" in result:
        mix, hidden, noisy = result["mix"], result["hidden"], result["noisy"]
        fails = failures(mix)
        lines.append(_line("full mix", mix, fails))
        drop = 1 - hidden["passes_per_ep"] / mix["passes_per_ep"] if mix["passes_per_ep"] else 0.0
        hidden_fails = ["leans on hidden dynamics"] if drop > HIDDEN_DROP else []
        lines.append(_line("mix, dynamics hidden", hidden, hidden_fails) + f"  (passes {-drop:+.0%})")
        noisy_fails = [f for f in failures(noisy, NOISE_CRASHED) if f in ("crashed", "passes")]
        lines.append(_line(f"mix, noise {NOISE}", noisy, noisy_fails))
        ok &= not (fails or hidden_fails or noisy_fails)
    lines.append(f"worst corner passes/ep {result['worst']:.2f}: {'PASSES every gate' if ok else 'NOT ready'}")
    return "\n".join(lines), ok


def pick(name: str, seeds: int, flights: int) -> int:
    """The seed whose worst corner is best; its gates are printed to stderr."""
    worst = {}
    for seed in range(seeds):
        path = Path("runs") / f"{name}_s{seed}" / "model.zip"
        if not path.exists():
            continue
        result = check(str(path), flights, workers=len(corners()), full=False)
        worst[seed] = result["worst"]
        print(f"seed {seed}\n{report(result)[0]}", file=sys.stderr, flush=True)
    return max(worst, key=worst.get)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", nargs="?")
    parser.add_argument("--pick", nargs=2, metavar=("NAME", "SEEDS"))
    parser.add_argument("--flights", type=int, default=200)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    if args.pick:
        print(pick(args.pick[0], int(args.pick[1]), args.flights))
        return
    if not actor_module.is_privileged(args.model):
        sys.exit(f"{args.model} is not a privileged actor: nothing to check here")
    text, ok = report(check(args.model, args.flights, args.workers))
    print(text)
    print(f"\nCan a student copy it? Its DAgger rounds say:\n  python -m src.train --algo ppo --teacher "
          f"{args.model} --randomise --level 1 --steps 1500000 --name clone-check\n"
          "and read each 'dagger round' line: the student's flights and the clone loss.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
