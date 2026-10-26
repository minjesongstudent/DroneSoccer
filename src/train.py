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
CHECKPOINT_EVERY = 250_000
BUFFER_SIZE = 1_000_000
INIT_WARMUP = 200_000
DEMO_LEVEL = 1.0
DEMO_NOISE = 0.1
DEMO_SHARE = 0.25
DEMO_SEED = 2_000_000

RUN_IN_SHARE = 0.1

SELECT_EPISODES = 50
SELECT_SEED = 3_000_000

LEVEL_STEP = 0.25
LEVEL_WINDOW = 200
LEVEL_UP = 0.7
LEVEL_PASSES = 1
MISSION_PASSES = 2


class Episodes:

    def __init__(self, total: int, level: float = 0.0, every: int = 10_000):
        self.total, self.level, self.every = total, level, every
        self.recent = deque(maxlen=LEVEL_WINDOW)
        self.run_ins = deque(maxlen=LEVEL_WINDOW)
        self.start_clock(0)

    def start_clock(self, steps: int):
        self.start, self.start_steps = time.time(), steps
        self.printed = steps // self.every

    def add(self, info: dict, reward: float):
        if info.get("run_in"):
            self.run_ins.append(info["passes"] > 0)
        else:
            self.recent.append({**info, "reward": reward})

    def level_up(self) -> bool:
        if len(self.recent) == LEVEL_WINDOW and self.ready() >= LEVEL_UP and self.level < 1.0:
            self.level = min(1.0, self.level + LEVEL_STEP)
            self.recent.clear()
            return True
        return False

    def passes_per_ep(self) -> float:
        return float(np.mean([i["passes"] for i in self.recent]))

    def reward_per_ep(self) -> float:
        return float(np.mean([i["reward"] for i in self.recent]))

    def crash_rate(self) -> float:
        return float(np.mean([i["outcome"] in CRASHES for i in self.recent]))

    def ready(self) -> float:
        return float(np.mean([i["passes"] >= LEVEL_PASSES for i in self.recent]))

    def summary(self) -> str:
        if not self.recent:
            return "-"
        ready = f"ready {self.ready():.0%}/{LEVEL_UP:.0%} of {len(self.recent)}/{LEVEL_WINDOW} eps"
        seconds = np.mean([i["steps"] for i in self.recent]) * Drone().dt
        return (f"passes/ep {self.passes_per_ep():.2f}  reward/ep {self.reward_per_ep():.2f}  "
                f"crashed {self.crash_rate():.0%}  ep {seconds:.0f}s  level {self.level:.2f}  {ready}")

    def progress_line(self, steps: int) -> str | None:
        if steps // self.every <= self.printed:
            return None
        self.printed = steps // self.every
        elapsed = time.time() - self.start
        eta = elapsed / max(1, steps - self.start_steps) * (self.total - steps)
        return (f"progress {steps:>7}/{self.total} ({steps / self.total:.0%})  {self.summary()}  "
                f"elapsed {elapsed / 60:.0f}m  eta {eta / 60:.0f}m")


class Progress(BaseCallback):

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
            episodes.add(info, info["episode"]["r"])
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
    return {"ready": float(np.mean([r["outcome"] == "timeout" and r["passes"] >= MISSION_PASSES
                                    for r in results])),
            "passes_per_ep": float(np.mean([r["passes"] for r in results])),
            "crashed": float(np.mean([r["outcome"] in CRASHES for r in results]))}


class Snapshot(BaseCallback):

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
    folder.mkdir(parents=True, exist_ok=True)
    files = {**files, "state.json": lambda f: f.write(json.dumps(state).encode())}
    for file, write in files.items():
        with open(folder / f"{file}.tmp", "wb") as f:
            write(f)
    for file in files:
        os.replace(folder / f"{file}.tmp", folder / file)
    print(f"snapshot at step {state['steps']}", flush=True)


def settings(extra: dict | None = None, **args) -> str:
    constants = {k: v for scope in (vars(contract), vars(sim), globals()) for k, v in scope.items()
                 if k.isupper() and isinstance(v, (int, float, str, tuple))}
    constants.update(extra or {})
    try:
        git = subprocess.run(["git", "describe", "--always", "--dirty"], cwd=ROOT,
                             capture_output=True, text=True).stdout.strip() or "unknown"
    except FileNotFoundError:
        git = "unknown"
    pairs = [*args.items(), *constants.items(), ("git", git)]
    return "settings " + " ".join(
        f"{k}={','.join(map(str, v)) if isinstance(v, tuple) else v}" for k, v in pairs)


def make_env(level: float, randomise: bool, envs: int, run_in_share: float = 0.0):
    def one():
        return Monitor(HoopEnv(level=level, randomise=randomise, run_in_share=run_in_share))
    return one() if envs == 1 else SubprocVecEnv([one] * envs)


class PolicyWarmup(SAC):

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)


def grown(init: str, env, seed: int, envs: int, buffer: dict | None = None) -> SAC:
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
        raise ValueError(f"{init} takes {n_old} inputs and the env gives {n_old + added}: only the flag can be "
                         f"added. Set contract.SLOT_STEPS to the layout the model was trained on")

    def widen(old_t, new_t):
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
    if not demos:
        return {}
    return {"replay_buffer_class": DemoReplayBuffer,
            "replay_buffer_kwargs": {"demo_size": demos, "demo_share": share}}


def add_demos(model: SAC, steps: int, randomise: bool, seed: int) -> str:
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
                    learning_starts=5_000,
                    buffer_size=BUFFER_SIZE, **demo_buffer(demos, demo_share),
                    tensorboard_log=str(RUNS))
    if demos and not resume:
        print(f"demos    {add_demos(model, demos, randomise, seed)}", flush=True)
    checkpoints = CheckpointCallback(CHECKPOINT_EVERY // envs, str(out / "checkpoints"),
                                     name_prefix="model")
    model.learn(total_timesteps=steps - model.num_timesteps,
                callback=[Progress(steps), checkpoints, Snapshot(snapshot, steps)],
                tb_log_name=name, reset_num_timesteps=not resume)
    model.save(out / "final")
    return pick_best(out, randomise, select_episodes)


def pick_best(out: Path, randomise: bool, episodes: int) -> Path:
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
    parser.add_argument("--privileged-actor", nargs="?", const="full", choices=("full", "gated"),
                        help="ppo: the actor also reads the sim's true state, all of it (full, the "
                             "default) or only as far as a student could know it (gated: "
                             "sim.HoopEnv.gated). Trains a teacher for --teacher, never a policy for the drone")
    parser.add_argument("--teacher",
                        help="ppo: label DAgger with this privileged actor (a --privileged-actor run's "
                             "model) instead of src.teacher's teacher3")
    args = parser.parse_args()
    if args.init and args.resume:
        parser.error("--init and --resume do not mix")
    if args.algo == "ppo" and (args.init or args.demos or args.envs != 1):
        parser.error("--init, --demos and --envs are SAC's; ppo warm-starts by --dagger and steps --rollout-envs")
    if args.algo == "sac" and (any(v is not None for v in (args.dagger, args.rollout_envs, args.bc_weight))
                               or args.privileged_actor or args.teacher):
        parser.error("--dagger, --rollout-envs, --bc-weight, --privileged-actor and --teacher need --algo ppo")

    if args.algo == "ppo":
        from src import ppo
        if args.rollout_envs is not None and args.rollout_envs < ppo.MINIBATCHES:
            parser.error(f"--rollout-envs must be at least ppo.MINIBATCHES ({ppo.MINIBATCHES})")
        if args.teacher and not Path(args.teacher).exists():
            named = ROOT / "trainedModels" / f"{args.teacher}.zip"
            if not named.exists():
                parser.error(f"--teacher {args.teacher}: no such file, and no {named}")
            args.teacher = str(named)
        given = {"dagger": args.dagger, "rollout_envs": args.rollout_envs, "bc_weight": args.bc_weight,
                 "privileged_actor": args.privileged_actor or None, "teacher": args.teacher}
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
    try:
        print(f"plots    {save_plots(args.name)}")
    except Exception as e:
        print(f"plots    FAILED: {e!r}. Retry: python -m src.plots {args.name}")
