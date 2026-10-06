"""Train SAC on the hoop simulator.

Run: python -m src.train --steps 3000000 --name laps-v1
Watch: tensorboard --logdir runs
    rollout/passes_per_ep and curriculum/level are the numbers that matter

Saves a checkpoint to runs/<name>/checkpoints/ every CHECKPOINT_EVERY steps and
runs/<name>/model.zip at the end, then evaluates it against the baseline on the
same episodes, then saves the training curves to runs/<name>/plots/. The
policy's scores also go to runs/<name>/eval.json, which run_seeds.sh reads to
keep only the best seed's models.
See what a checkpoint does: python -m src.replay --model runs/<name>/checkpoints/<file>.zip

At each checkpoint it also overwrites runs/<name>/resume/ with everything needed
to carry on: the model with its optimisers, the replay buffer and the curriculum
level. Continue a run that was cut off with the same command plus --resume;
--steps is still the total, not the steps left.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv

from src import baseline, contract, sim
from src.contract import CRASHES
from src.evaluate import DEFAULT_EPISODES, load_policy, report, run_episodes
from src.params import ROOT, Drone
from src.plots import save_plots
from src.sim import HoopEnv

RUNS = ROOT / "runs"
CHECKPOINT_EVERY = 250_000   # steps; ~3 MB each, without the replay buffer
BUFFER_SIZE = 1_000_000      # SB3's default; ~700 MB in each resume snapshot
# --init flies the loaded policy this long, unchanged, before the first update. With
# SB3's 5k random steps the first 1k updates on so small a buffer wiped out the
# lap (2.04 passes/ep to 0.28); 200k of its own flying kept it near 2.
INIT_WARMUP = 200_000

# Curriculum: step the start level up by LEVEL_STEP once LEVEL_UP of the last
# LEVEL_WINDOW episodes made at least LEVEL_PASSES passes, however they ended.
# The level only changes the start, so its bar is the first pass. A second pass
# needs the lap, which is the same at every level: at 2 passes, leaving level 0
# meant lapping as well as the best model yet (seenflag-s1-level1 s1: 2+ passes in
# 71% of level-0 episodes), and 7 of 8 seeds from scratch never left it. Level 0
# also never needs a line-up, which the lap's second pass does. Lasting the minute
# with MISSION_PASSES+ passes is the mission's bar, which scores() ranks seeds by.
LEVEL_STEP = 0.25
LEVEL_WINDOW = 200
LEVEL_UP = 0.7
LEVEL_PASSES = 1
MISSION_PASSES = 2


class Progress(BaseCallback):
    """Logs how the last training episodes went, raises the curriculum level,
    and prints a one-line progress report every `every` steps."""

    def __init__(self, total: int, every: int = 10_000):
        super().__init__()
        self.total, self.every = total, every
        self.recent = deque(maxlen=LEVEL_WINDOW)
        self.level = 0.0

    def _on_training_start(self) -> None:
        self.start = time.time()
        self.start_steps = self.num_timesteps   # non-zero after --resume
        self.printed = self.num_timesteps // self.every
        self.level = self.training_env.get_attr("level")[0]

    def _on_step(self) -> bool:
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if done:
                self.recent.append(info)
                self._record()
        # N parallel envs raise the count by N, so an exact multiple may never
        # come round. Fire on the step that crosses one instead.
        if self.num_timesteps // self.every > self.printed:
            self.printed = self.num_timesteps // self.every
            elapsed = time.time() - self.start
            done = max(1, self.num_timesteps - self.start_steps)
            eta = elapsed / done * (self.total - self.num_timesteps)
            print(f"progress {self.num_timesteps:>7}/{self.total} "
                  f"({self.num_timesteps / self.total:.0%})  {self._summary()}  "
                  f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m", flush=True)
        return True

    def _passes_per_ep(self) -> float:
        return float(np.mean([i["passes"] for i in self.recent]))

    def _crash_rate(self) -> float:
        return float(np.mean([i["outcome"] in CRASHES for i in self.recent]))

    def _record(self):
        self.logger.record("rollout/passes_per_ep", self._passes_per_ep())
        self.logger.record("rollout/crash_rate", self._crash_rate())
        self.logger.record("curriculum/level", self.level)
        if len(self.recent) == LEVEL_WINDOW and self._ready() >= LEVEL_UP and self.level < 1.0:
            self.level = min(1.0, self.level + LEVEL_STEP)
            self.training_env.env_method("set_level", self.level)
            self.recent.clear()   # judge the new level on its own episodes
            print(f"level    {self.level:.2f} at step {self.num_timesteps}", flush=True)

    def _ready(self) -> float:
        """Share of recent episodes with LEVEL_PASSES+ passes, however they ended."""
        return float(np.mean([i["passes"] >= LEVEL_PASSES for i in self.recent]))

    def _summary(self) -> str:
        if not self.recent:
            return "-"
        # ready: how close the level-up is. It counts over up to LEVEL_WINDOW
        # episodes, and the level only rises once the window is full.
        ready = f"ready {self._ready():.0%}/{LEVEL_UP:.0%} of {len(self.recent)}/{LEVEL_WINDOW} eps"
        # Episodes that end before the minute are crashes.
        seconds = np.mean([i["steps"] for i in self.recent]) * Drone().dt
        return (f"passes/ep {self._passes_per_ep():.2f}  crashed {self._crash_rate():.0%}  "
                f"ep {seconds:.0f}s  level {self.level:.2f}  {ready}")


def scores(results: list[dict]) -> dict:
    """What run_seeds.sh ranks seeds by. ready is the mission's bar: the share of
    episodes that last the full minute with MISSION_PASSES+ passes. The curriculum's
    bar is looser: one pass, however the episode ended. Passes are per episode, not
    per minute flown, which flattered seeds that pass once and crash within seconds."""
    return {"ready": float(np.mean([r["outcome"] == "timeout" and r["passes"] >= MISSION_PASSES
                                    for r in results])),
            "passes_per_ep": float(np.mean([r["passes"] for r in results])),
            "crashed": float(np.mean([r["outcome"] in CRASHES for r in results]))}


class Snapshot(BaseCallback):
    """Every CHECKPOINT_EVERY steps, overwrites `folder` with what --resume needs.
    Only one copy is kept: the replay buffer alone is ~700 MB."""

    def __init__(self, folder: Path, total: int):
        super().__init__()
        self.folder, self.total = folder, total

    def _on_training_start(self) -> None:
        self.saved = self.num_timesteps // CHECKPOINT_EVERY

    def _on_step(self) -> bool:
        if self.num_timesteps // CHECKPOINT_EVERY > self.saved:
            self.saved = self.num_timesteps // CHECKPOINT_EVERY
            self.save()
        return True

    def save(self):
        self.folder.mkdir(parents=True, exist_ok=True)
        state = {"steps": self.num_timesteps, "total": self.total,
                 "level": float(self.training_env.get_attr("level")[0])}
        # Write every file in full before replacing any, so a job killed mid-save
        # still leaves the previous snapshot to resume from.
        files = {"model.zip": self.model.save, "replay_buffer.pkl": self.model.save_replay_buffer,
                 "state.json": lambda f: f.write(json.dumps(state).encode())}
        for file, write in files.items():
            with open(self.folder / f"{file}.tmp", "wb") as f:
                write(f)
        for file in files:
            os.replace(self.folder / f"{file}.tmp", self.folder / file)
        print(f"snapshot at step {self.num_timesteps}", flush=True)


def settings(**args) -> str:
    """One `settings` line for the log: the run's options, every constant in
    contract.py and sim.py, the constants here, and the git commit.
    src.report reads it back, so a report shows what the run was trained with
    even after contract.py or sim.py is edited for the next experiment."""
    constants = {k: v for scope in (vars(contract), vars(sim), globals()) for k, v in scope.items()
                 if k.isupper() and isinstance(v, (int, float, str, tuple))}
    try:  # compute nodes have no git
        git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except FileNotFoundError:
        git = "unknown"
    pairs = [*args.items(), *constants.items(), ("git", git)]
    return "settings " + " ".join(
        f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in pairs)


def make_env(level: float, randomise: bool, envs: int):
    """One environment, or `envs` of them in their own processes.

    Stepping is numpy and the policy runs on CPU, so copies scale nearly
    linearly. train_freq and gradient_steps rise with them, which keeps the
    ratio of steps flown to gradients taken the same as with one.
    """
    def one():
        return Monitor(HoopEnv(level=level, randomise=randomise))
    return one() if envs == 1 else SubprocVecEnv([one] * envs)


class PolicyWarmup(SAC):
    """SAC that fills its buffer with its own policy's actions before learning
    starts, not uniform random ones, so a loaded policy keeps flying as it did."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def grown(init: str, env, seed: int, envs: int) -> SAC:
    """A fresh model for `env` with a trained model's networks, optimiser state and
    entropy coefficient, and an empty replay buffer (fine-tuning). Unlike --resume,
    this needs only the model.zip.

    If the saved policy input was shorter, the inputs added after it (the seen-hoop
    flag) get zero weights in the actor and both critics, so the new model acts and
    values exactly as the old one did until training changes them.
    """
    old = SAC.load(init, device="cpu")
    model = PolicyWarmup("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                         gamma=old.gamma, learning_rate=old.learning_rate,
                         batch_size=old.batch_size, tau=old.tau,
                         train_freq=envs, gradient_steps=envs,
                         learning_starts=INIT_WARMUP, buffer_size=BUFFER_SIZE,
                         tensorboard_log=str(RUNS))
    n_old = old.observation_space.shape[0]
    added = model.observation_space.shape[0] - n_old
    if added < 0:
        raise ValueError(f"{init} takes {n_old} inputs, more than the {n_old + added} the env gives")

    def widen(old_t, new_t):
        # First layers read [obs, (action)]: insert zero columns where the old obs ended.
        if old_t.shape == new_t.shape:
            return old_t.clone()
        zeros = old_t.new_zeros(old_t.shape[0], added, *old_t.shape[2:])
        return torch.cat([old_t[:, :n_old], zeros, old_t[:, n_old:]], dim=1)

    new_state = model.policy.state_dict()
    model.policy.load_state_dict({k: widen(v, new_state[k]) for k, v in old.policy.state_dict().items()})
    for name in ("actor", "critic"):
        old_opt, new_opt = getattr(old, name).optimizer, getattr(model, name).optimizer
        state = old_opt.state_dict()
        params = [p for g in new_opt.param_groups for p in g["params"]]
        for i, slot in state["state"].items():
            for key in ("exp_avg", "exp_avg_sq"):
                slot[key] = widen(slot[key], params[i])
        new_opt.load_state_dict(state)
    with torch.no_grad():
        model.log_ent_coef.copy_(old.log_ent_coef)
    model.ent_coef_optimizer.load_state_dict(old.ent_coef_optimizer.state_dict())
    return model


def train(steps: int, name: str, seed: int, level: float, randomise: bool,
          resume: bool = False, envs: int = 1, init: str | None = None) -> Path:
    print(settings(steps=steps, seed=seed, level=level, randomise=randomise, resume=resume,
                   envs=envs, init=init or "none"), flush=True)
    out = RUNS / name
    snapshot = out / "resume"
    if resume:
        level = json.loads((snapshot / "state.json").read_text())["level"]
    env = make_env(level, randomise, envs)
    if resume:
        model = SAC.load(snapshot / "model.zip", env=env, device="cpu")
        model.load_replay_buffer(snapshot / "replay_buffer.pkl")
        # A fresh seed, so the resumed run does not replay its first episode layouts.
        model.set_random_seed(seed + model.num_timesteps)
        print(f"resumed  from step {model.num_timesteps} at level {level:.2f}", flush=True)
    elif init:
        model = grown(init, env, seed, envs)
        print(f"init     from {init} at level {level:.2f}, flying it for {INIT_WARMUP} steps "
              f"before the first update", flush=True)
    else:
        model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                    gamma=contract.GAMMA,
                    train_freq=envs, gradient_steps=envs,
                    learning_starts=5_000,   # fill the buffer with random flying first
                    buffer_size=BUFFER_SIZE,
                    tensorboard_log=str(RUNS))
    checkpoints = CheckpointCallback(CHECKPOINT_EVERY // envs, str(out / "checkpoints"),
                                     name_prefix="model")
    # With reset_num_timesteps=False, learn() counts total_timesteps on from the
    # loaded step count and appends to the same tensorboard run.
    model.learn(total_timesteps=steps - model.num_timesteps,
                callback=[Progress(steps), checkpoints, Snapshot(snapshot, steps)],
                tb_log_name=name, reset_num_timesteps=not resume)
    model.save(out / "model")
    return out / "model.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500_000)
    parser.add_argument("--name", default="sac")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--level", type=float, default=0.0,
                        help="starting curriculum level; evaluation is always at 1")
    parser.add_argument("--randomise", action="store_true", help="draw dynamics per episode")
    parser.add_argument("--eval-episodes", type=int)
    parser.add_argument("--envs", type=int, default=1,
                        help="parallel environments; needs one core each")
    parser.add_argument("--resume", action="store_true",
                        help="carry on from runs/<name>/resume/ up to --steps in total")
    parser.add_argument("--init", help="start from this trained model.zip, with an empty "
                        "replay buffer (fine-tuning)")
    args = parser.parse_args()
    if args.init and args.resume:
        parser.error("--init and --resume do not mix")

    path = train(args.steps, args.name, args.seed, args.level, args.randomise,
                 args.resume, args.envs, args.init)
    print(f"saved {path}")
    episodes = args.eval_episodes or DEFAULT_EPISODES

    def evaluate(policy):
        return run_episodes(policy, episodes, HoopEnv(randomise=args.randomise))

    results = evaluate(load_policy(str(path)))
    print("policy   ", report(results))
    print("baseline ", report(evaluate(baseline.Lapper())))
    (path.parent / "eval.json").write_text(json.dumps(scores(results)))
    # Last, and never fatal: the model and scores matter more than the pictures.
    try:
        print(f"plots    {save_plots(args.name)}")
    except Exception as e:
        print(f"plots    FAILED: {e!r}. Retry: python -m src.plots {args.name}")
