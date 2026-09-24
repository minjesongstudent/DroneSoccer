"""Pass rate of a policy or the baseline over fixed evaluation episodes.

Evaluation seeds start at EVAL_SEED so they never overlap training episodes,
and every policy sees the same layouts, so pass rates compare directly.

Run: python -m src.evaluate --baseline
     python -m src.evaluate --model runs/sac/model.zip
"""
from __future__ import annotations

import argparse
from collections import Counter
from typing import Callable

import numpy as np

from src import baseline
from src.sim import HoopEnv

EVAL_SEED = 1_000_000
OUTCOMES = ("passed", "hit", "missed", "left_cage", "timeout")

Policy = Callable[[np.ndarray], np.ndarray]


def run_episodes(policy: Policy, episodes: int, env: HoopEnv | None = None) -> Counter:
    env = env or HoopEnv()
    outcomes = Counter()
    for i in range(episodes):
        obs, _ = env.reset(seed=EVAL_SEED + i)
        done = False
        while not done:
            obs, _, terminated, truncated, info = env.step(policy(obs))
            done = terminated or truncated
        outcomes[info["outcome"]] += 1
    return outcomes


def load_policy(model_path: str) -> Policy:
    from stable_baselines3 import SAC
    model = SAC.load(model_path, device="cpu")
    return lambda obs: model.predict(obs, deterministic=True)[0]


def report(outcomes: Counter) -> str:
    total = sum(outcomes.values())
    return "  ".join(f"{name} {outcomes[name] / total:6.1%}" for name in OUTCOMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", action="store_true")
    group.add_argument("--model", help="path to a saved SAC model.zip")
    parser.add_argument("--episodes", type=int, default=500)
    args = parser.parse_args()

    policy = baseline.act if args.baseline else load_policy(args.model)
    print(report(run_episodes(policy, args.episodes)))
