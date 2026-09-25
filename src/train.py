"""Train SAC on the hoop simulator.

Run: python -m src.train --steps 3000000 --name laps-v1              (the laps mission)
     python -m src.train --steps 500000 --name sac --task hover      (the original task)
Watch: tensorboard --logdir runs
    laps:  rollout/passes_per_min and curriculum/level are the numbers that matter
    hover: rollout/pass_rate

Saves runs/<name>/model.zip, then evaluates it against the baseline on the
same episodes, then saves the training curves to runs/<name>/plots/.
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

from src.evaluate import CRASHES, DEFAULT_EPISODES, baseline_for, load_policy, report, run_episodes
from src.params import ROOT, Drone
from src.plots import save_plots
from src.sim import HoopEnv

RUNS = ROOT / "runs"

# Laps curriculum: step the start level up by LEVEL_STEP once LEVEL_UP of the
# last LEVEL_WINDOW episodes lasted the full minute with at least LEVEL_PASSES passes.
LEVEL_STEP = 0.25
LEVEL_WINDOW = 200
LEVEL_UP = 0.7
LEVEL_PASSES = 2


class Progress(BaseCallback):
    """Logs how the last training episodes went, raises the laps curriculum level,
    and prints a one-line progress report every `every` steps."""

    def __init__(self, total: int, task: str, every: int = 10_000):
        super().__init__()
        self.total, self.task, self.every = total, task, every
        self.recent = deque(maxlen=100 if task == "hover" else LEVEL_WINDOW)
        self.level = 0.0

    def _on_training_start(self) -> None:
        self.start = time.time()
        if self.task == "laps":
            self.level = self.training_env.get_attr("level")[0]

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.recent.append(info)
                self._record()
        if self.num_timesteps % self.every == 0:
            elapsed = time.time() - self.start
            eta = elapsed / self.num_timesteps * (self.total - self.num_timesteps)
            print(f"progress {self.num_timesteps:>7}/{self.total} "
                  f"({self.num_timesteps / self.total:.0%})  {self._summary()}  "
                  f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m", flush=True)
        return True

    def _passes_per_min(self) -> float:
        minutes = sum(i["steps"] for i in self.recent) * Drone().dt / 60
        return sum(i["passes"] for i in self.recent) / minutes

    def _record(self):
        if self.task == "hover":
            self.logger.record("rollout/pass_rate",
                               float(np.mean([i["outcome"] == "passed" for i in self.recent])))
            return
        self.logger.record("rollout/passes_per_min", self._passes_per_min())
        self.logger.record("rollout/crash_rate",
                           float(np.mean([i["outcome"] in CRASHES for i in self.recent])))
        self.logger.record("curriculum/level", self.level)
        if len(self.recent) == LEVEL_WINDOW and self._ready() >= LEVEL_UP and self.level < 1.0:
            self.level = min(1.0, self.level + LEVEL_STEP)
            self.training_env.env_method("set_level", self.level)
            self.recent.clear()   # judge the new level on its own episodes
            print(f"level    {self.level:.2f} at step {self.num_timesteps}", flush=True)

    def _ready(self) -> float:
        """Share of recent episodes that lasted the full minute with LEVEL_PASSES+ passes."""
        return float(np.mean([i["outcome"] == "timeout" and i["passes"] >= LEVEL_PASSES
                              for i in self.recent]))

    def _summary(self) -> str:
        if not self.recent:
            return "-"
        if self.task == "hover":
            return f"pass_rate {np.mean([i['outcome'] == 'passed' for i in self.recent]):.0%}"
        # ready: how close the level-up is. It counts over up to LEVEL_WINDOW
        # episodes, and the level only rises once the window is full.
        ready = f"ready {self._ready():.0%}/{LEVEL_UP:.0%} of {len(self.recent)}/{LEVEL_WINDOW} eps"
        return f"passes/min {self._passes_per_min():.2f}  level {self.level:.2f}  {ready}"


def train(steps: int, name: str, seed: int, task: str, level: float, randomise: bool) -> Path:
    out = RUNS / name
    env = Monitor(HoopEnv(task=task, level=level, randomise=randomise))
    model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                # Hover episodes are at most 150 steps. A lap takes ~100, so the
                # next pass must still be worth chasing from the far side of the hoop.
                gamma=0.995 if task == "laps" else 0.98,
                learning_starts=5_000,   # fill the buffer with random flying first
                tensorboard_log=str(RUNS))
    model.learn(total_timesteps=steps, callback=Progress(steps, task), tb_log_name=name)
    model.save(out / "model")
    return out / "model.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--name", default="sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--task", choices=("hover", "laps"), default="laps")
    parser.add_argument("--level", type=float, default=0.0,
                        help="laps: starting curriculum level; evaluation is always at 1")
    parser.add_argument("--randomise", action="store_true", help="draw dynamics per episode")
    parser.add_argument("--eval-episodes", type=int)
    args = parser.parse_args()

    path = train(args.steps, args.name, args.seed, args.task, args.level, args.randomise)
    print(f"saved {path}")
    episodes = args.eval_episodes or DEFAULT_EPISODES[args.task]

    def evaluate(policy):
        env = HoopEnv(task=args.task, randomise=args.randomise)
        return report(run_episodes(policy, episodes, env), args.task)

    print("policy   ", evaluate(load_policy(str(path))))
    print("baseline ", evaluate(baseline_for(args.task)))
    # Last, and never fatal: the model and scores matter more than the pictures.
    try:
        print(f"plots    {save_plots(args.name)}")
    except Exception as e:
        print(f"plots    FAILED: {e!r}. Retry: python -m src.plots {args.name}")
