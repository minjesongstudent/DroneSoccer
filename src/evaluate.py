"""Score a policy or the baseline over fixed evaluation episodes.

Evaluation seeds start at EVAL_SEED so they never overlap training episodes,
and every policy sees the same layouts, so the numbers compare directly.

    hover   pass rate, and how the other episodes ended
    laps    passes per minute, how many runs crashed, and the time to the first pass

Run: python -m src.evaluate --baseline
     python -m src.evaluate --model runs/sac/model.zip --task laps
     python -m src.evaluate --model ... --task laps --delay 3     (one point of the delay sweep)
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Callable

import numpy as np

from src import baseline
from src.params import Drone
from src.sim import HoopEnv

EVAL_SEED = 1_000_000
OUTCOMES = ("passed", "hit", "missed", "left_cage", "timeout")
LAP_ENDINGS = ("hit", "wrong_way", "left_cage", "timeout")
CRASHES = ("hit", "wrong_way", "left_cage")
DEFAULT_EPISODES = {"hover": 500, "laps": 200}   # a laps episode is four times longer

Policy = Callable[[np.ndarray], np.ndarray]


def run_episodes(policy: Policy, episodes: int, env: HoopEnv | None = None) -> list[dict]:
    """One dict per episode: how it ended, passes, steps, and the step of the first pass."""
    env = env or HoopEnv()
    results = []
    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        if hasattr(policy, "reset"):
            policy.reset()
        first_pass, done = None, False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
            if first_pass is None and info["passes"]:
                first_pass = info["steps"]
        results.append({"outcome": info["outcome"], "passes": info["passes"],
                        "steps": info["steps"], "first_pass": first_pass})
    return results


def load_policy(model_path: str) -> Policy:
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def baseline_for(task: str) -> Policy:
    return baseline.Lapper() if task == "laps" else baseline.act


def report(results: list[dict], task: str = "hover", dt: float = Drone().dt) -> str:
    outcomes = Counter(r["outcome"] for r in results)
    total = len(results)
    if task == "hover":
        return "  ".join(f"{name} {outcomes[name] / total:6.1%}" for name in OUTCOMES)

    minutes = sum(r["steps"] for r in results) * dt / 60
    per_min = sum(r["passes"] for r in results) / minutes
    crashed = sum(outcomes[name] for name in CRASHES) / total
    firsts = [r["first_pass"] * dt for r in results if r["first_pass"] is not None]
    first = f"{np.mean(firsts):5.1f}s" if firsts else "  never"
    endings = "  ".join(f"{name} {outcomes[name] / total:.0%}" for name in LAP_ENDINGS)
    return (f"passes/min {per_min:5.2f}  crashed {crashed:6.1%}  first pass {first}"
            f"  ({endings})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--model", help="path to a saved SAC model.zip")
    parser.add_argument("--task", choices=("hover", "laps"), default="hover")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--level", type=float, default=1.0, help="laps curriculum level, 0-1")
    parser.add_argument("--randomise", action="store_true", help="draw dynamics per episode")
    parser.add_argument("--delay", type=int, help="fixed command delay in steps, for the sweep")
    args = parser.parse_args()

    drone = Drone(delay_steps=args.delay) if args.delay is not None else None
    env = HoopEnv(task=args.task, level=args.level, randomise=args.randomise, drone=drone)
    policy = baseline_for(args.task) if args.baseline else load_policy(args.model)
    episodes = args.episodes or DEFAULT_EPISODES[args.task]
    print(report(run_episodes(policy, episodes, env), args.task))
