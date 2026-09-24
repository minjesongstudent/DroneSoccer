"""Train SAC on the hoop simulator.

Run: python -m src.train --steps 500000 --name sac
Watch: tensorboard --logdir runs   (rollout/pass_rate is the number that matters)

Saves runs/<name>/model.zip, then evaluates it against the baseline on the
same episodes.
"""
from __future__ import annotations

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src import baseline
from src.evaluate import load_policy, report, run_episodes
from src.params import ROOT
from src.sim import HoopEnv

RUNS = ROOT / "runs"


class PassRate(BaseCallback):
    """Logs the pass rate over the last 100 training episodes, and prints a
    one-line progress report every `every` steps."""

    def __init__(self, total: int, every: int = 10_000):
        super().__init__()
        self.recent = deque(maxlen=100)
        self.total, self.every = total, every

    def _on_training_start(self) -> None:
        self.start = time.time()

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.recent.append(info["outcome"] == "passed")
                self.logger.record("rollout/pass_rate", float(np.mean(self.recent)))
        if self.num_timesteps % self.every == 0:
            elapsed = time.time() - self.start
            eta = elapsed / self.num_timesteps * (self.total - self.num_timesteps)
            rate = f"{np.mean(self.recent):.0%}" if self.recent else "-"
            print(f"progress {self.num_timesteps:>7}/{self.total} "
                  f"({self.num_timesteps / self.total:.0%})  pass_rate {rate}  "
                  f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m", flush=True)
        return True


def train(steps: int, name: str, seed: int) -> Path:
    out = RUNS / name
    env = Monitor(HoopEnv())
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                gamma=0.98,              # episodes are at most 150 steps
                learning_starts=5_000,   # fill the buffer with random flying first
                tensorboard_log=str(RUNS))
    model.learn(total_timesteps=steps, callback=PassRate(steps), tb_log_name=name)
    model.save(out / "model")
    return out / "model.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--name", default="sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-episodes", type=int, default=500)
    args = parser.parse_args()

    path = train(args.steps, args.name, args.seed)
    print(f"saved {path}")
    print("policy   ", report(run_episodes(load_policy(str(path)), args.eval_episodes)))
    print("baseline ", report(run_episodes(baseline.act, args.eval_episodes)))
