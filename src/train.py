"""Train SAC on the hoop simulator.

Run: python -m src.train --steps 3000000 --name laps-v1
Watch: tensorboard --logdir runs
    rollout/passes_per_ep and curriculum/level are the numbers that matter

Saves a checkpoint to runs/<name>/checkpoints/ every CHECKPOINT_EVERY steps and
runs/<name>/final.zip at the end. Then it scores every checkpoint and the final
model on SELECT_EPISODES episodes of their own (checkpoints.json) and copies the
best to runs/<name>/model.zip: seeds have peaked at 1.3M steps and collapsed by 5M.
That one is evaluated against the baseline on the held-out episodes, then the
training curves go to runs/<name>/plots/. The
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
import shutil
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
from src.demo_buffer import DemoReplayBuffer
from src.teacher import Teacher

RUNS = ROOT / "runs"
CHECKPOINT_EVERY = 250_000   # steps; ~3 MB each, without the replay buffer
BUFFER_SIZE = 1_000_000      # SB3's default; ~700 MB in each resume snapshot
# --init flies the loaded policy this long, unchanged, before the first update. With
# SB3's 5k random steps the first 1k updates on so small a buffer wiped out the
# lap (2.04 passes/ep to 0.28); 200k of its own flying kept it near 2.
INIT_WARMUP = 200_000
# --demos flies src.teacher before the first update and keeps its flights for the whole
# run, so the critic sees the whole mission from the start: the blind climb until the
# hoop is in view, passes, and the way round. From scratch, the first passes came only
# after 150-240k steps at level 0 (firstpass-curriculum), and the new detector makes
# them rarer. They fly the real start (level 1), whatever the curriculum level. Noise
# on the teacher's actions widens what they cover (the first teacher: 0.1 cost little,
# 0.2 crashed 27% of flights). They go in a buffer of their own (demo_buffer.py) that
# supplies DEMO_SHARE of every batch: in the ordinary buffer they were gone by 1M steps,
# before the curriculum reached the start they show.
DEMO_LEVEL = 1.0
DEMO_NOISE = 0.1
DEMO_SHARE = 0.25       # RLPD uses 0.5
DEMO_SEED = 2_000_000   # layouts apart from training's and evaluate.EVAL_SEED's

# Run-in starts (sim.RUN_IN_*): this share of training episodes starts lined up
# 0.6-1.0 m in front of the hoop, whatever the curriculum level, so the blind last
# half metre of a pass gets practised. teacher2-level1 stopped there on the drone.
# They do not count towards the curriculum (Progress), and evaluation never has them.
RUN_IN_SHARE = 0.1

# Picking the model: every checkpoint and the final model fly SELECT_EPISODES episodes
# from SELECT_SEED, apart from evaluation's, so the reported score is not the one it
# was picked on. Ranked as run_seeds.sh ranks seeds (scores()).
SELECT_EPISODES = 50
SELECT_SEED = 3_000_000

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


class Episodes:
    """The last LEVEL_WINDOW training episodes: they raise the curriculum level and make
    the progress line. Progress (SAC) and src.ppo's Tracker keep one each, so both
    algorithms climb the curriculum and print their progress the same way, which
    src.report and hpc/progress.sh read. Run-in starts are kept apart."""

    def __init__(self, total: int, level: float = 0.0, every: int = 10_000):
        self.total, self.level, self.every = total, level, every
        self.recent = deque(maxlen=LEVEL_WINDOW)
        self.run_ins = deque(maxlen=LEVEL_WINDOW)   # run-in starts, kept apart
        self.start_clock(0)

    def start_clock(self, steps: int):
        """The progress line's elapsed and eta count from here: non-zero after --resume."""
        self.start, self.start_steps = time.time(), steps
        self.printed = steps // self.every

    def add(self, info: dict, reward: float):
        """reward: the episode's summed (unscaled) reward."""
        if info.get("run_in"):
            self.run_ins.append(info["passes"] > 0)
        else:
            self.recent.append({**info, "reward": reward})

    def level_up(self) -> bool:
        """Step the level up if the full window met the bar. The caller tells its envs."""
        if len(self.recent) == LEVEL_WINDOW and self.ready() >= LEVEL_UP and self.level < 1.0:
            self.level = min(1.0, self.level + LEVEL_STEP)
            self.recent.clear()   # judge the new level on its own episodes
            return True
        return False

    def passes_per_ep(self) -> float:
        return float(np.mean([i["passes"] for i in self.recent]))

    def reward_per_ep(self) -> float:
        return float(np.mean([i["reward"] for i in self.recent]))

    def crash_rate(self) -> float:
        return float(np.mean([i["outcome"] in CRASHES for i in self.recent]))

    def ready(self) -> float:
        """Share of recent episodes with LEVEL_PASSES+ passes, however they ended."""
        return float(np.mean([i["passes"] >= LEVEL_PASSES for i in self.recent]))

    def summary(self) -> str:
        if not self.recent:
            return "-"
        # ready: how close the level-up is. It counts over up to LEVEL_WINDOW
        # episodes, and the level only rises once the window is full.
        ready = f"ready {self.ready():.0%}/{LEVEL_UP:.0%} of {len(self.recent)}/{LEVEL_WINDOW} eps"
        # Episodes that end before the minute are crashes.
        seconds = np.mean([i["steps"] for i in self.recent]) * Drone().dt
        return (f"passes/ep {self.passes_per_ep():.2f}  reward/ep {self.reward_per_ep():.2f}  "
                f"crashed {self.crash_rate():.0%}  ep {seconds:.0f}s  level {self.level:.2f}  {ready}")

    def progress_line(self, steps: int) -> str | None:
        """The line to print once every `every` steps, else None. N parallel envs raise
        the count by N, so an exact multiple may never come round: it fires on the
        step that crosses one instead."""
        if steps // self.every <= self.printed:
            return None
        self.printed = steps // self.every
        elapsed = time.time() - self.start
        eta = elapsed / max(1, steps - self.start_steps) * (self.total - steps)
        return (f"progress {steps:>7}/{self.total} ({steps / self.total:.0%})  {self.summary()}  "
                f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m")


class Progress(BaseCallback):
    """Logs how the last training episodes went, raises the curriculum level,
    and prints a one-line progress report every `every` steps."""

    def __init__(self, total: int, every: int = 10_000):
        super().__init__()
        self.episodes = Episodes(total, every=every)

    def _on_training_start(self) -> None:
        self.episodes.start_clock(self.num_timesteps)
        self.episodes.level = self.training_env.get_attr("level")[0]

    def _on_step(self) -> bool:
        episodes = self.episodes
        for info, done in zip(self.locals["infos"], self.locals["dones"]):
            if not done:
                continue
            episodes.add(info, info["episode"]["r"])   # Monitor's episode sum
            if info.get("run_in"):
                self.logger.record("rollout/run_in_passed", float(np.mean(episodes.run_ins)))
                continue
            self.logger.record("rollout/passes_per_ep", episodes.passes_per_ep())
            self.logger.record("rollout/crash_rate", episodes.crash_rate())
            self.logger.record("curriculum/level", episodes.level)
            if episodes.level_up():
                self.training_env.env_method("set_level", episodes.level)
                print(f"level    {episodes.level:.2f} at step {self.num_timesteps}", flush=True)
        line = episodes.progress_line(self.num_timesteps)
        if line:
            print(line, flush=True)
        return True


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
        state = {"steps": self.num_timesteps, "total": self.total,
                 "level": float(self.training_env.get_attr("level")[0])}
        write_snapshot(self.folder, state, {"model.zip": self.model.save,
                                            "replay_buffer.pkl": self.model.save_replay_buffer})


def write_snapshot(folder: Path, state: dict, files: dict):
    """Write `files` ({name: writer(file)}) and state.json ({steps, total, level}, which
    ds resume reads) into folder. Every file is written in full before any is replaced,
    so a job killed mid-save still leaves the previous snapshot to resume from.
    src.ppo's snapshots go through here too."""
    folder.mkdir(parents=True, exist_ok=True)
    files = {**files, "state.json": lambda f: f.write(json.dumps(state).encode())}
    for file, write in files.items():
        with open(folder / f"{file}.tmp", "wb") as f:
            write(f)
    for file in files:
        os.replace(folder / f"{file}.tmp", folder / file)
    print(f"snapshot at step {state['steps']}", flush=True)


def settings(extra: dict | None = None, **args) -> str:
    """One `settings` line for the log: the run's options, every constant in
    contract.py and sim.py, the constants here, any `extra` (src.ppo's), and the git commit.
    src.report reads it back, so a report shows what the run was trained with
    even after contract.py or sim.py is edited for the next experiment."""
    constants = {k: v for scope in (vars(contract), vars(sim), globals()) for k, v in scope.items()
                 if k.isupper() and isinstance(v, (int, float, str, tuple))}
    constants.update(extra or {})
    try:  # compute nodes have no git
        git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except FileNotFoundError:
        git = "unknown"
    pairs = [*args.items(), *constants.items(), ("git", git)]
    return "settings " + " ".join(
        f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in pairs)


def make_env(level: float, randomise: bool, envs: int, run_in_share: float = 0.0):
    """One environment, or `envs` of them in their own processes.

    Stepping is numpy and the policy runs on CPU, so copies scale nearly
    linearly. train_freq and gradient_steps rise with them, which keeps the
    ratio of steps flown to gradients taken the same as with one.
    """
    def one():
        return Monitor(HoopEnv(level=level, randomise=randomise, run_in_share=run_in_share))
    return one() if envs == 1 else SubprocVecEnv([one] * envs)


class PolicyWarmup(SAC):
    """SAC that fills its buffer with its own policy's actions before learning
    starts, not uniform random ones, so a loaded policy keeps flying as it did."""

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def grown(init: str, env, seed: int, envs: int, buffer: dict | None = None) -> SAC:
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
                         learning_starts=INIT_WARMUP, buffer_size=BUFFER_SIZE, **(buffer or {}),
                         tensorboard_log=str(RUNS))
    n_old = old.observation_space.shape[0]
    added = model.observation_space.shape[0] - n_old
    if added < 0:
        raise ValueError(f"{init} takes {n_old} inputs, more than the {n_old + added} the env gives")
    if added > 1:
        # Growing only appends inputs at the end (the flag). A different history
        # layout moves every input, so zero columns at the end would scramble the old weights.
        raise ValueError(f"{init} takes {n_old} inputs and the env gives {n_old + added}: only the flag can be "
                         f"added. Set contract.SLOT_STEPS to the layout the model was trained on")

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


def demo_buffer(demos: int, share: float) -> dict:
    """SAC's replay buffer options: with demos, a DemoReplayBuffer to hold them."""
    if not demos:
        return {}
    return {"replay_buffer_class": DemoReplayBuffer,
            "replay_buffer_kwargs": {"demo_size": demos, "demo_share": share}}


def add_demos(model: SAC, steps: int, randomise: bool, seed: int) -> str:
    """Fly src.teacher for `steps` steps into model's teacher buffer, one flight per
    environment at a time, and say how the flights went."""
    n = model.n_envs
    buffer = model.replay_buffer.demos
    envs = [HoopEnv(level=DEMO_LEVEL, randomise=randomise) for _ in range(n)]
    rng = np.random.default_rng(DEMO_SEED + seed)
    teachers = [Teacher(rng) for _ in range(n)]
    obs = []
    for k, (env, teacher) in enumerate(zip(envs, teachers)):
        obs.append(env.reset(seed=DEMO_SEED + 1000 * seed + k)[0])
        teacher.reset(env)
    flights = passes = crashes = 0
    for _ in range(steps // n):
        before, actions, after, rewards, dones, infos = list(obs), [], [], [], [], []
        for k, (env, teacher) in enumerate(zip(envs, teachers)):
            action = np.clip(teacher.act(env, obs[k]) + rng.normal(0.0, DEMO_NOISE, contract.ACTION_DIM),
                             -1.0, 1.0).astype(np.float32)
            o, r, terminated, truncated, info = env.step(action)
            actions.append(action)
            after.append(o)
            rewards.append(r)
            dones.append(terminated or truncated)
            # As SB3's VecEnv marks it: a timeout bootstraps, a crash does not.
            infos.append({"TimeLimit.truncated": truncated and not terminated})
            if terminated or truncated:
                flights, passes, crashes = flights + 1, passes + info["passes"], crashes + terminated
                o = env.reset()[0]
                teacher.reset(env)
            obs[k] = o
        buffer.add(np.array(before), np.array(after), np.array(actions),
                   np.array(rewards), np.array(dones), infos)
    return (f"{steps // n * n} steps, {flights} flights: {passes / max(flights, 1):.2f} passes/flight, "
            f"crashed {crashes / max(flights, 1):.0%}")


def train(steps: int, name: str, seed: int, level: float, randomise: bool,
          resume: bool = False, envs: int = 1, init: str | None = None, demos: int = 0,
          demo_share: float = DEMO_SHARE, run_in_share: float = RUN_IN_SHARE,
          select_episodes: int = SELECT_EPISODES) -> Path:
    print(settings(steps=steps, seed=seed, level=level, randomise=randomise, resume=resume,
                   envs=envs, init=init or "none", demos=demos, demo_share=demo_share,
                   run_in_share=run_in_share, select_episodes=select_episodes), flush=True)
    out = RUNS / name
    snapshot = out / "resume"
    if resume:
        level = json.loads((snapshot / "state.json").read_text())["level"]
    env = make_env(level, randomise, envs, run_in_share)
    if resume:
        model = SAC.load(snapshot / "model.zip", env=env, device="cpu")
        model.load_replay_buffer(snapshot / "replay_buffer.pkl")
        # A fresh seed, so the resumed run does not replay its first episode layouts.
        model.set_random_seed(seed + model.num_timesteps)
        print(f"resumed  from step {model.num_timesteps} at level {level:.2f}", flush=True)
    elif init:
        model = grown(init, env, seed, envs, demo_buffer(demos, demo_share))
        print(f"init     from {init} at level {level:.2f}, flying it for {INIT_WARMUP} steps "
              f"before the first update", flush=True)
    else:
        model = SAC("MlpPolicy", env, seed=seed, device="cpu", verbose=0,
                    gamma=contract.GAMMA,
                    train_freq=envs, gradient_steps=envs,
                    learning_starts=5_000,   # fill the buffer with random flying first
                    buffer_size=BUFFER_SIZE, **demo_buffer(demos, demo_share),
                    tensorboard_log=str(RUNS))
    if demos and not resume:   # a resumed run's buffer has them already
        print(f"demos    {add_demos(model, demos, randomise, seed)}", flush=True)
    checkpoints = CheckpointCallback(CHECKPOINT_EVERY // envs, str(out / "checkpoints"),
                                     name_prefix="model")
    # With reset_num_timesteps=False, learn() counts total_timesteps on from the
    # loaded step count and appends to the same tensorboard run.
    model.learn(total_timesteps=steps - model.num_timesteps,
                callback=[Progress(steps), checkpoints, Snapshot(snapshot, steps)],
                tb_log_name=name, reset_num_timesteps=not resume)
    model.save(out / "final")
    return pick_best(out, randomise, select_episodes)


def pick_best(out: Path, randomise: bool, episodes: int) -> Path:
    """Copy the best of the checkpoints and final.zip to out/model.zip, scored on
    `episodes` selection episodes, and write every score to checkpoints.json.
    With episodes 0, final.zip is the model."""
    candidates = [out / "final.zip"]
    if episodes:
        candidates += sorted((out / "checkpoints").glob("model_*_steps.zip"),
                             key=lambda f: int(f.stem.split("_")[1]))
    table = {}
    for path in candidates if len(candidates) > 1 else []:
        results = run_episodes(load_policy(str(path)), episodes, HoopEnv(randomise=randomise), seed=SELECT_SEED)
        table[path.name] = scores(results)
        print(f"select   {path.name:<24} ready {table[path.name]['ready']:.0%}  "
              f"passes/ep {table[path.name]['passes_per_ep']:.2f}  crashed {table[path.name]['crashed']:.0%}",
              flush=True)
    best = (max(table, key=lambda k: (table[k]["ready"], table[k]["passes_per_ep"], -table[k]["crashed"]))
            if table else "final.zip")
    (out / "checkpoints.json").write_text(json.dumps({"best": best, "scores": table}, indent=1))
    source = out / best if best == "final.zip" else out / "checkpoints" / best
    shutil.copyfile(source, out / "model.zip")
    print(f"best     {best} -> {out / 'model.zip'}", flush=True)
    return out / "model.zip"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=("sac", "ppo"), default="sac",
                        help="ppo: src.ppo's privileged-critic PPO, warm-started by DAgger")
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
    parser.add_argument("--demos", type=int, default=0,
                        help="fly src.teacher for this many steps first and keep its flights for the whole run")
    parser.add_argument("--demo-share", type=float, default=DEMO_SHARE,
                        help="share of every batch drawn from the teacher's flights")
    parser.add_argument("--run-in-share", type=float, default=RUN_IN_SHARE,
                        help="share of training episodes that start lined up in front of the hoop")
    parser.add_argument("--select-episodes", type=int, default=SELECT_EPISODES,
                        help="episodes each checkpoint flies to pick the best; 0 keeps the final model")
    parser.add_argument("--dagger", type=int,
                        help="ppo: DAgger rounds before PPO, 0 for none (default: all of ppo.DAGGER_BETA)")
    parser.add_argument("--rollout-envs", type=int,
                        help="ppo: simulators stepped in turn on the seed's one core (default ppo.ROLLOUT_ENVS)")
    parser.add_argument("--bc-weight", type=float,
                        help="ppo: weight of the pull towards the teacher's labels as PPO starts (default ppo.BC_WEIGHT)")
    args = parser.parse_args()
    if args.init and args.resume:
        parser.error("--init and --resume do not mix")
    if args.algo == "ppo" and (args.init or args.demos or args.envs != 1):
        parser.error("--init, --demos and --envs are SAC's; ppo warm-starts by --dagger and steps --rollout-envs")
    if args.algo == "sac" and any(v is not None for v in (args.dagger, args.rollout_envs, args.bc_weight)):
        parser.error("--dagger, --rollout-envs and --bc-weight need --algo ppo")

    if args.algo == "ppo":
        from src import ppo
        if args.rollout_envs is not None and args.rollout_envs < ppo.MINIBATCHES:
            parser.error(f"--rollout-envs must be at least ppo.MINIBATCHES ({ppo.MINIBATCHES})")
        given = {"dagger": args.dagger, "rollout_envs": args.rollout_envs, "bc_weight": args.bc_weight}
        path = ppo.train_ppo(args.steps, args.name, args.seed, args.level, args.randomise, args.resume,
                             run_in_share=args.run_in_share, select_episodes=args.select_episodes,
                             **{k: v for k, v in given.items() if v is not None})
    else:
        path = train(args.steps, args.name, args.seed, args.level, args.randomise,
                     args.resume, args.envs, args.init, args.demos, args.demo_share,
                     args.run_in_share, args.select_episodes)
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
