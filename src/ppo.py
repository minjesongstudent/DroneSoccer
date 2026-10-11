"""PPO with a privileged critic and a recurrent actor, warm-started by DAgger on src.teacher.

Run: python -m src.train --algo ppo --steps 20000000 --name ppo-v1 --randomise --level 1
     (hpc: ./ds submit 6 20000000 ppo-v1 --algo ppo --randomise --level 1)

Why not SAC: at 3M steps the teacher3 students (teacher3-12s-height) passed the first
hoop on 82-88% of flights and never lapped, and 38-46% of flights ended turning back
through the hoop (wrong_way). Three changes:
  - The critic sees HoopEnv.privileged(), the true state, as well as the policy input.
    The lap is flown blind, and a critic on blank frames values every blind state
    alike, so it cannot say which blind command was the right one. The actor still sees
    the policy input and nothing else, so what flies on the drone is unchanged.
  - The actor has a GRU on top of the 12 s of slots. It can carry where it is round
    the lap for as long as the hoop stays out of view, past the window's end too.
  - On-policy, with no replay buffer: a seed steps ROLLOUT_ENVS simulators in turn on
    its one core, and takes no gradient step per env step. SAC trained at ~50 steps/s
    on the 430 inputs, held back by its updates; the sim alone runs ~1500 steps/s.

Warm start (DAgger): teacher3 decides its blind moves from the policy input alone, so a
student can copy those exactly. The first round flies the teacher, with DEMO_NOISE on
its commands. Later rounds fly the student more and more (DAGGER_BETA), and the teacher
labels every state the student reaches, so the student learns to recover from its own
mistakes, which a fixed set of demos never shows it. After every round the actor is
cloned on all the labels so far. Then the critic learns the student's values with
the actor frozen (CRITIC_WARMUP), because the first policy steps would otherwise follow
a random critic (the SAC fine-tune collapse). Then PPO takes over, still pulled towards the
teacher's labels by a BC term that halves every BC_HALF_LIFE steps. Every step flown
counts towards --steps, the teacher's too. The search stays learned: the teacher
only labels training states and never flies on the drone.
"""
from __future__ import annotations

import math
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from src import contract, plots, train
# LOG_STD_* and MEMORY are imported so the settings line logs them with the rest.
from src.actor import HIDDEN, LOG_STD_INIT, LOG_STD_MIN, MEMORY, Actor, normalised, save_actor  # noqa: F401
from src.sim import PRIVILEGED_DIM, HoopEnv
from src.teacher import Teacher

ROLLOUT_ENVS = 16        # simulators per seed, stepped in turn on one core
ROLLOUT_STEPS = 128      # steps per simulator per update: 2048 a rollout
EPOCHS = 5
MINIBATCHES = 4          # of whole simulator sequences, so the GRU runs through them in order
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.2
TARGET_KL = 0.02         # stop an update's epochs past 1.5x this, so a cloned actor is not thrown off
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
MAX_GRAD_NORM = 1.0
# Values in units of 10 reward: a pass is 4, a crash -3, a minute of laps a few tens.
REWARD_SCALE = 0.1
# The actor's sizes and its spread (HIDDEN, MEMORY, LOG_STD_*) are in src/actor.py.
STD_FLOOR = 0.1          # input normalisers never divide by less: inputs are O(1) already
CHECKPOINT_EVERY = 1_000_000

DAGGER_BETA = (1.0, 0.5, 0.25, 0.0, 0.0, 0.0)   # per round: share of flights the teacher flies
DAGGER_STEPS = 100_000   # per round
# Over every label so far, after each round, and after the last. 3 a round left the clone
# flying 0.01-0.12 passes/flight against the teacher's 2.9 (200 updates a round). A further
# 40 epochs on the same labels (2026-10-10, 30 eval flights): loss 0.029 -> 0.014, 0.73
# passes/flight, half its tries at a second pass made it, no wrong_way. Most of what is left is the
# forward command with the hoop in view: the teacher starts its run in on true state
# (within 3 cm and 3 degrees of the axis, still), which the input does not show, so the
# clone averages hovering with running in. PPO's critic sees that state; that is its job.
BC_EPOCHS = 5
BC_FINAL_EPOCHS = 40     # ~23 s an epoch on 600k labels: ~15 min
BC_BATCH = 64            # sequences
BC_BURN_IN = 32          # steps at the start of a mid-episode sequence that only rebuild the memory
BC_LR = 3e-4
BC_WEIGHT = 1.0          # BC term's weight once PPO starts: squared command error
BC_HALF_LIFE = 2_000_000
BC_PPO_SEQUENCES = 8     # label sequences per PPO minibatch for the BC term, whatever --rollout-envs
BC_MIN = 0.01            # below this weight the BC term is dropped, ~13M steps into PPO
CRITIC_WARMUP = 500_000  # steps after DAgger with the actor frozen


# --- the critic ----------------------------------------------------------------------------

class Critic(nn.Module):
    """Policy input and true state -> value, in REWARD_SCALE units."""

    def __init__(self, obs_dim: int = contract.POLICY_INPUT_DIM, priv_dim: int = PRIVILEGED_DIM):
        super().__init__()
        self.register_buffer("mean", torch.zeros(obs_dim + priv_dim))
        self.register_buffer("std", torch.ones(obs_dim + priv_dim))
        self.net = nn.Sequential(nn.Linear(obs_dim + priv_dim, HIDDEN), nn.ELU(),
                                 nn.Linear(HIDDEN, HIDDEN), nn.ELU(), nn.Linear(HIDDEN, 1))

    def forward(self, obs, priv):
        return self.net(normalised(torch.cat([obs, priv], -1), self.mean, self.std)).squeeze(-1)


def fit_normaliser(module: nn.Module, *parts: np.ndarray):
    """Set module's input scales from samples of each part of its input, (..., width) each,
    in the order the input joins them."""
    flat = [p.reshape(-1, p.shape[-1]) for p in parts]
    module.mean.copy_(torch.as_tensor(np.concatenate([p.mean(0) for p in flat])))
    module.std.copy_(torch.as_tensor(np.maximum(np.concatenate([p.std(0) for p in flat]), STD_FLOOR)))


def log_prob(action, mean, std):
    return (-0.5 * ((action - mean) / std) ** 2 - torch.log(std) - 0.5 * math.log(2 * math.pi)).sum(-1)


# --- the simulators ---------------------------------------------------------------------

class Fleet:
    """ROLLOUT_ENVS simulators on this core, each with a teacher to label its states.
    A finished episode resets at once, so obs is always the next step's input."""

    def __init__(self, n: int, level: float, randomise: bool, run_in_share: float, seed: int,
                 rng: np.random.Generator):
        self.envs = [HoopEnv(level=level, randomise=randomise, run_in_share=run_in_share) for _ in range(n)]
        self.teachers = [Teacher() for _ in range(n)]
        self.rng, self.n, self.beta = rng, n, 0.0
        self.expert = np.zeros(n, bool)   # this episode is flown by the teacher
        self.obs = np.zeros((n, contract.POLICY_INPUT_DIM), np.float32)
        self.priv = np.zeros((n, PRIVILEGED_DIM), np.float32)
        self.starts = np.ones(n, np.float32)
        self.ep_reward = np.zeros(n)
        for k in range(n):
            self._reset(k, seed=seed * 10_000 + k)

    def _reset(self, k: int, seed=None):
        env = self.envs[k]
        self.obs[k] = env.reset(seed=seed)[0]
        self.priv[k] = env.privileged()
        self.teachers[k].reset(env)
        self.expert[k] = self.rng.uniform() < self.beta
        self.starts[k], self.ep_reward[k] = 1.0, 0.0

    def restart(self):
        """Start every simulator on a new episode, drawn at the current beta. At a DAgger
        round boundary this keeps an episode from being flown by one round's pilot and
        counted in the next: the episodes cut off here are never reported."""
        for k in range(self.n):
            self._reset(k)

    def set_level(self, level: float):
        for env in self.envs:
            env.set_level(level)

    def labels(self) -> np.ndarray:
        """The teacher's command for every simulator's current state. Call it every step
        of an episode or none: the teacher counts steps since it last saw the hoop."""
        return np.stack([t.act(e, o) for t, e, o in zip(self.teachers, self.envs, self.obs)])

    def step(self, actions: np.ndarray, value_of):
        """Step every simulator. Returns rewards, dones and the episodes that ended
        (info, flown by the teacher, episode reward). A timeout bootstraps: its reward
        gets GAMMA times the critic's value of the state it was cut off in."""
        rewards, dones, ended = np.zeros(self.n, np.float32), np.zeros(self.n, np.float32), []
        self.starts[:] = 0.0
        for k, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(actions[k])
            self.ep_reward[k] += reward
            rewards[k] = REWARD_SCALE * reward
            self.obs[k], self.priv[k] = obs, env.privileged()
            if terminated or truncated:
                if truncated and not terminated:
                    rewards[k] += contract.GAMMA * value_of(self.obs[k], self.priv[k])
                dones[k] = 1.0
                ended.append((info, bool(self.expert[k]), float(self.ep_reward[k])))
                self._reset(k)
        return rewards, dones, ended


def collect(fleet: Fleet, actor: Actor, critic: Critic, h: torch.Tensor, steps: int,
            label: bool = False, noise: float = train.DEMO_NOISE):
    """steps steps of every simulator. The student acts, sampled, except in episodes the
    teacher flies (fleet.beta of them), where the teacher's command plus noise does.
    Returns the rollout as (steps, n, ...) arrays, the GRU state after it, and the
    episodes that ended."""
    n = fleet.n
    roll = {"obs": np.zeros((steps, n, contract.POLICY_INPUT_DIM), np.float32),
            "priv": np.zeros((steps, n, PRIVILEGED_DIM), np.float32),
            "actions": np.zeros((steps, n, contract.ACTION_DIM), np.float32),
            "logp": np.zeros((steps, n), np.float32), "values": np.zeros((steps, n), np.float32),
            "rewards": np.zeros((steps, n), np.float32), "dones": np.zeros((steps, n), np.float32),
            "starts": np.zeros((steps, n), np.float32), "h0": h.numpy().copy()}
    if label:
        roll["labels"] = np.zeros((steps, n, contract.ACTION_DIM), np.float32)

    def value_of(obs, priv):
        with torch.no_grad():
            return float(critic(torch.as_tensor(obs)[None], torch.as_tensor(priv)[None])[0])

    ended = []
    with torch.no_grad():
        std = actor.std_now()
        for t in range(steps):
            obs, priv = torch.as_tensor(fleet.obs), torch.as_tensor(fleet.priv)
            starts = torch.as_tensor(fleet.starts)
            mean, h = actor.step(obs, h * (1.0 - starts)[:, None])
            action = mean + std * torch.randn_like(mean)
            roll["obs"][t], roll["priv"][t], roll["starts"][t] = fleet.obs, fleet.priv, fleet.starts
            roll["logp"][t] = log_prob(action, mean, std).numpy()
            roll["values"][t] = critic(obs, priv).numpy()
            act = action.numpy()
            if label:
                labels = fleet.labels()
                roll["labels"][t] = labels
                taught = np.clip(labels + fleet.rng.normal(0.0, noise, labels.shape), -1.0, 1.0)
                act = np.where(fleet.expert[:, None], taught, act).astype(np.float32)
            roll["actions"][t] = act
            roll["rewards"][t], roll["dones"][t], done = fleet.step(act, value_of)
            ended += done
        roll["last_value"] = critic(torch.as_tensor(fleet.obs), torch.as_tensor(fleet.priv)).numpy()
    return roll, h, ended


def advantages(roll: dict, gamma: float = contract.GAMMA, lam: float = GAE_LAMBDA):
    """GAE over the rollout. dones[t] means step t ended its episode, so nothing after it
    counts; a timeout's bootstrap is already in its reward."""
    values, rewards, dones = roll["values"], roll["rewards"], roll["dones"]
    adv = np.zeros_like(rewards)
    last = np.zeros(rewards.shape[1], np.float32)
    for t in reversed(range(len(rewards))):
        following = roll["last_value"] if t == len(rewards) - 1 else values[t + 1]
        keep = 1.0 - dones[t]
        delta = rewards[t] + gamma * following * keep - values[t]
        last = delta + gamma * lam * keep * last
        adv[t] = last
    return adv, adv + values


# --- DAgger's labels ----------------------------------------------------------------------

class Labels:
    """Every DAgger round's states as sequences of ROLLOUT_STEPS steps, with the teacher's
    command for each. float16, so 600k steps of 430 inputs take ~0.5 GB.

    The GRU's memory is not stored: what the collecting actor held means nothing to the
    actor after the next clone. Each sequence replays from a clear memory, and its first
    BC_BURN_IN steps only rebuild it and are not scored, unless an episode starts there,
    where a clear memory is the true one."""

    def __init__(self):
        self.parts = {"obs": [], "labels": [], "starts": []}
        self._joined = None

    def add(self, roll: dict):
        self.parts["obs"].append(roll["obs"].transpose(1, 0, 2).astype(np.float16))
        self.parts["labels"].append(roll["labels"].transpose(1, 0, 2))
        self.parts["starts"].append(roll["starts"].T)
        self._joined = None

    @property
    def joined(self) -> dict:
        if self._joined is None:
            self._joined = {k: np.concatenate(v) for k, v in self.parts.items()}
            self.parts = {k: [v] for k, v in self._joined.items()}   # one copy, not two
        return self._joined

    def __len__(self):
        return sum(len(x) for x in self.parts["obs"])

    def loss(self, actor: Actor, idx) -> torch.Tensor:
        """Mean squared error of the actor's command against the teacher's, over sequences
        idx, on the steps scored (the class docstring)."""
        d = self.joined
        obs = torch.as_tensor(d["obs"][idx].astype(np.float32)).transpose(0, 1)
        starts = torch.as_tensor(d["starts"][idx]).T
        mean, _ = actor(obs, actor.initial(len(idx)), starts)
        scored = (torch.arange(len(obs))[:, None] >= BC_BURN_IN) | (starts.cummax(0).values > 0)
        error = ((mean - torch.as_tensor(d["labels"][idx]).transpose(0, 1)) ** 2).mean(-1)
        return (error * scored).sum() / scored.sum().clamp(min=1)

    def save(self, path: Path):
        np.savez(path, **self.joined)

    @classmethod
    def load(cls, path: Path) -> "Labels":
        labels = cls()
        with np.load(path) as f:
            for k in labels.parts:
                labels.parts[k].append(f[k])
        return labels


def clone(actor: Actor, labels: Labels, optimiser, rng, epochs: int = BC_EPOCHS) -> float:
    """Fit the actor to every label so far; returns the last epoch's mean loss."""
    for _ in range(epochs):
        losses = []
        order = rng.permutation(len(labels))
        for start in range(0, len(order), BC_BATCH):
            loss = labels.loss(actor, order[start:start + BC_BATCH])
            optimiser.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            optimiser.step()
            losses.append(loss.item())
    return float(np.mean(losses))


# --- progress -------------------------------------------------------------------------------

class Tracker:
    """The student's episodes, in a train.Episodes as train.Progress keeps them for SAC, and
    the TensorBoard numbers. Teacher-flown episodes are left out."""

    def __init__(self, total: int, level: float, writer: SummaryWriter, start_steps: int):
        self.episodes = train.Episodes(total, level)
        self.episodes.start_clock(start_steps)
        self.writer = writer
        self.rewards, self.lengths = deque(maxlen=100), deque(maxlen=100)

    @property
    def level(self) -> float:
        return self.episodes.level

    def add(self, ended, fleet: Fleet, steps: int):
        for info, expert, reward in ended:
            if expert:
                continue
            self.episodes.add(info, reward)
            if info.get("run_in"):
                continue
            self.rewards.append(reward)
            self.lengths.append(info["steps"])
            if self.episodes.level_up():
                fleet.set_level(self.episodes.level)
                print(f"level    {self.episodes.level:.2f} at step {steps}", flush=True)

    def record(self, steps: int, scalars: dict):
        e = self.episodes
        if e.recent:
            scalars = {"rollout/passes_per_ep": e.passes_per_ep(), "rollout/crash_rate": e.crash_rate(),
                       "rollout/ep_rew_mean": np.mean(self.rewards), "rollout/ep_len_mean": np.mean(self.lengths),
                       **scalars}
        if e.run_ins:
            scalars["rollout/run_in_passed"] = np.mean(e.run_ins)
        scalars["curriculum/level"] = e.level
        scalars["time/fps"] = (steps - e.start_steps) / max(time.time() - e.start, 1e-9)
        for tag, value in scalars.items():
            self.writer.add_scalar(tag, float(value), steps)
        line = e.progress_line(steps)
        if line:
            print(line, flush=True)


def _flights(ended, expert: bool) -> str:
    mine = [info for info, e, _ in ended if e == expert and not info.get("run_in")]
    if not mine:
        return "no flights"
    laps = np.mean([i["passes"] >= train.MISSION_PASSES for i in mine])
    return (f"{len(mine)} flights {np.mean([i['passes'] for i in mine]):.2f} passes/flight, "
            f"2+ passes {laps:.0%}, crashed {np.mean([i['outcome'] in contract.CRASHES for i in mine]):.0%}")


def _log_dir(name: str, resume: bool) -> Path:
    """runs/<name>_<N>, as SB3 numbers them: the newest on --resume, else a new one."""
    train.RUNS.mkdir(parents=True, exist_ok=True)
    try:
        newest = plots.find_log_dir(name, train.RUNS)
    except FileNotFoundError:
        return train.RUNS / f"{name}_1"
    n = int(newest.name.rsplit("_", 1)[1])
    return newest if resume else train.RUNS / f"{name}_{n + 1}"


# --- training ---------------------------------------------------------------------------------

def update(actor, critic, opt_actor, opt_critic, roll, rng, train_actor: bool,
           labels: Labels | None, bc_weight: float) -> dict:
    """PPO's epochs over one rollout. Minibatches are whole simulator sequences, so the
    actor's GRU replays them from the state it started the rollout in."""
    adv, ret = advantages(roll)
    adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)
    t = {k: torch.as_tensor(roll[k]) for k in ("obs", "priv", "actions", "logp", "starts", "h0")}
    adv_t, ret_t = torch.as_tensor(adv_n), torch.as_tensor(ret)
    n = roll["obs"].shape[1]
    stats = {"critic_loss": [], "actor_loss": [], "approx_kl": [], "bc_loss": [], "clip_fraction": []}
    for _ in range(EPOCHS):
        kls = []
        for mb in np.array_split(rng.permutation(n), MINIBATCHES):
            value = critic(t["obs"][:, mb], t["priv"][:, mb])
            critic_loss = ((value - ret_t[:, mb]) ** 2).mean()
            opt_critic.zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(critic.parameters(), MAX_GRAD_NORM)
            opt_critic.step()
            stats["critic_loss"].append(critic_loss.item())
            if not train_actor:
                continue
            mean, _ = actor(t["obs"][:, mb], t["h0"][mb], t["starts"][:, mb])
            logp = log_prob(t["actions"][:, mb], mean, actor.std_now())
            ratio = torch.exp(logp - t["logp"][:, mb])
            a = adv_t[:, mb]
            loss = -torch.min(ratio * a, ratio.clamp(1 - CLIP_RANGE, 1 + CLIP_RANGE) * a).mean()
            stats["actor_loss"].append(loss.item())
            if labels is not None and bc_weight > 0:
                bc = labels.loss(actor, rng.integers(0, len(labels), BC_PPO_SEQUENCES))
                loss = loss + bc_weight * bc
                stats["bc_loss"].append(bc.item())
            opt_actor.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), MAX_GRAD_NORM)
            opt_actor.step()
            with torch.no_grad():
                log_ratio = logp - t["logp"][:, mb]
                kls.append(((ratio - 1) - log_ratio).mean().item())
                stats["clip_fraction"].append(((ratio - 1).abs() > CLIP_RANGE).float().mean().item())
        stats["approx_kl"] += kls
        if kls and np.mean(kls) > 1.5 * TARGET_KL:
            break
    return {f"train/{k}": float(np.mean(v)) for k, v in stats.items() if v}


def snapshot(folder: Path, total: int, state: dict, labels: Labels | None = None):
    """What --resume needs, through train.write_snapshot as SAC's: state.pt holds the
    networks, optimisers and where the run is (state's keys), and labels.npz the teacher's
    labels. Labels are written only while DAgger adds to them: written each time, a
    snapshot would cost a 0.5 GB write a million steps."""
    files = {"state.pt": lambda f: torch.save(state, f)}
    if labels is not None:
        files["labels.npz"] = lambda f: np.savez(f, **labels.joined)
    run = state["run"]
    train.write_snapshot(folder, {"steps": run["steps"], "total": total, "level": run["level"]}, files)


def train_ppo(steps: int, name: str, seed: int, level: float, randomise: bool, resume: bool = False,
              dagger: int = len(DAGGER_BETA), run_in_share: float = train.RUN_IN_SHARE,
              select_episodes: int = train.SELECT_EPISODES, rollout_envs: int = ROLLOUT_ENVS,
              bc_weight: float = BC_WEIGHT) -> Path:
    constants = {k: v for k, v in globals().items() if k.isupper() and isinstance(v, (int, float, str, tuple))}
    print(train.settings(extra=constants, algo="ppo", steps=steps, seed=seed, level=level, randomise=randomise,
                         resume=resume, dagger=dagger, run_in_share=run_in_share,
                         select_episodes=select_episodes, rollout_envs=rollout_envs, bc_weight=bc_weight),
          flush=True)
    if rollout_envs < MINIBATCHES:
        # Fewer simulators than minibatches leaves some empty: NaN losses, no KL stop.
        raise ValueError(f"rollout_envs {rollout_envs} is below MINIBATCHES {MINIBATCHES}")
    out = train.RUNS / name
    folder = out / "resume"
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    actor, critic = Actor(), Critic()
    opt_actor = torch.optim.Adam(actor.parameters(), ACTOR_LR)
    opt_critic = torch.optim.Adam(critic.parameters(), CRITIC_LR)
    opt_clone = torch.optim.Adam(actor.parameters(), BC_LR)
    # Where the run is, all saved in every snapshot, so a resume follows the run it carries on
    # and not the options it was resumed with, nor files an older run of the name left behind.
    run = {"steps": 0, "level": level, "phase": "dagger" if dagger else "scales", "next_round": 0,
           "rounds": min(dagger, len(DAGGER_BETA)), "ppo_start": 0,
           "warmup": CRITIC_WARMUP if dagger else 0, "labels": False}
    labels = None
    if resume:
        state = torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
        for net, key in ((actor, "actor"), (critic, "critic"), (opt_actor, "opt_actor"),
                         (opt_critic, "opt_critic"), (opt_clone, "opt_clone")):
            net.load_state_dict(state[key])
        run = state["run"]
        level = run["level"]
        if run["labels"]:
            labels = Labels.load(folder / "labels.npz")
        # A fresh seed, so the resumed run does not replay its first episode layouts.
        torch.manual_seed(seed + run["steps"])
        rng = np.random.default_rng(seed + run["steps"])
        print(f"resumed  from step {run['steps']} at level {level:.2f}", flush=True)
    done_steps = run["steps"]
    fleet = Fleet(rollout_envs, level, randomise, run_in_share, seed + done_steps, rng)
    writer = SummaryWriter(str(_log_dir(name, resume)))
    tracker = Tracker(steps, level, writer, done_steps)
    h = actor.initial(rollout_envs)
    per_rollout = ROLLOUT_STEPS * rollout_envs

    def save(with_labels: bool = False):
        run.update(steps=done_steps, level=tracker.level, labels=labels is not None)
        snapshot(folder, steps, {"actor": actor.state_dict(), "critic": critic.state_dict(),
                                 "opt_actor": opt_actor.state_dict(), "opt_critic": opt_critic.state_dict(),
                                 "opt_clone": opt_clone.state_dict(), "run": dict(run)},
                 labels if with_labels else None)

    def fly(rounds: int, label: bool):
        """Rollouts with no update, counted and tracked like any. Returns their episodes, the
        policy inputs flown (None when labelled: the labels hold them) and the true states."""
        nonlocal h, done_steps
        ended, obs, privs = [], [], []
        for _ in range(rounds):
            roll, h, done = collect(fleet, actor, critic, h, ROLLOUT_STEPS, label=label)
            if label:
                labels.add(roll)
            else:
                obs.append(roll["obs"])
            privs.append(roll["priv"])
            ended += done
            done_steps += per_rollout
            tracker.add(done, fleet, done_steps)
            tracker.record(done_steps, {})
        return ended, None if label else np.concatenate(obs), np.concatenate(privs)

    rollouts = max(1, DAGGER_STEPS // per_rollout)
    if run["phase"] == "dagger":
        labels = Labels() if labels is None else labels
        for i in range(run["next_round"], run["rounds"]):
            beta = DAGGER_BETA[i]
            fleet.beta = beta
            fleet.restart()
            ended, _, privs = fly(rollouts, label=True)
            if i == 0:   # the teacher's flights set the input scales, all ~100k steps of them
                obs = labels.joined["obs"].astype(np.float32)
                fit_normaliser(actor, obs)
                fit_normaliser(critic, obs, privs)
                del obs
            last = i == run["rounds"] - 1
            loss = clone(actor, labels, opt_clone, rng, BC_FINAL_EPOCHS if last else BC_EPOCHS)
            writer.add_scalar("train/clone_loss", loss, done_steps)
            print(f"dagger   round {i} beta {beta:.2f} at step {done_steps}: teacher {_flights(ended, True)}; "
                  f"student {_flights(ended, False)}; clone loss {loss:.4f} on {len(labels)} sequences",
                  flush=True)
            run["next_round"] = i + 1
            if last:
                fleet.beta = 0.0
                fleet.restart()   # no teacher-flown episode runs on into PPO
                # The clone alone, before PPO changes it: pick_best scores it with the rest.
                save_actor(actor, out / "checkpoints" / f"model_{done_steps}_steps.zip")
                run.update(phase="ppo", ppo_start=done_steps)
            # Every round, so a job cut off in DAgger loses one round at most.
            save(with_labels=True)
    elif run["phase"] == "scales":
        # No teacher: the untrained actor's own flights set the scales, as many as DAgger's first round.
        _, obs, privs = fly(rollouts, label=False)
        fit_normaliser(actor, obs)
        fit_normaliser(critic, obs, privs)
        del obs
        run.update(phase="ppo", ppo_start=done_steps)
        save()

    ppo_start, warmup = run["ppo_start"], run["warmup"]
    saved = done_steps // CHECKPOINT_EVERY
    while done_steps < steps:
        roll, h, ended = collect(fleet, actor, critic, h, ROLLOUT_STEPS)
        done_steps += per_rollout
        tracker.add(ended, fleet, done_steps)
        since = done_steps - ppo_start
        weight = bc_weight * 0.5 ** (since / BC_HALF_LIFE) if labels is not None else 0.0
        weight = weight if weight >= BC_MIN else 0.0   # it costs half the update
        scalars = update(actor, critic, opt_actor, opt_critic, roll, rng, since > warmup, labels, weight)
        scalars.update({"train/std": actor.std_now().mean().item(), "train/bc_weight": weight})
        tracker.record(done_steps, scalars)
        if done_steps // CHECKPOINT_EVERY > saved:
            saved = done_steps // CHECKPOINT_EVERY
            save_actor(actor, out / "checkpoints" / f"model_{done_steps}_steps.zip")
            save()
    writer.close()
    save_actor(actor, out / "final.zip")
    return train.pick_best(out, randomise, select_episodes)
