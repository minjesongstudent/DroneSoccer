"""src.ppo's actor, and loading a saved one to fly. Needs torch and contract.py only, so
evaluate.py and labcheck.py load a policy without the training stack.

A saved actor is a torch.save file named .zip like SB3's, so run_seeds.sh and
train.pick_best handle both kinds. is_ppo tells them apart.
"""
from __future__ import annotations

import math
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src import contract

HIDDEN = 256
MEMORY = 128             # GRU units
LOG_STD_INIT = math.log(0.15)   # DAgger flies the student at this; demo noise is 0.1
LOG_STD_MIN = math.log(0.02)


def normalised(x, mean, std):
    return (x - mean) / std


class Actor(nn.Module):
    """Policy input -> mean command, through an MLP and a GRU. The head reads both, so
    the slots reach it directly and the GRU only has to add what they lack."""

    def __init__(self, obs_dim: int = contract.POLICY_INPUT_DIM, act_dim: int = contract.ACTION_DIM,
                 hidden: int = HIDDEN, memory: int = MEMORY):
        super().__init__()
        self.obs_dim, self.hidden, self.memory = obs_dim, hidden, memory
        self.register_buffer("mean", torch.zeros(obs_dim))
        self.register_buffer("std", torch.ones(obs_dim))
        self.encoder = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ELU(), nn.Linear(hidden, hidden), nn.ELU())
        self.gru = nn.GRU(hidden, memory)
        self.head = nn.Sequential(nn.Linear(hidden + memory, hidden), nn.ELU(), nn.Linear(hidden, act_dim))
        nn.init.orthogonal_(self.head[-1].weight, 0.01)
        nn.init.zeros_(self.head[-1].bias)
        self.log_std = nn.Parameter(torch.full((act_dim,), LOG_STD_INIT))

    def initial(self, batch: int) -> torch.Tensor:
        return torch.zeros(batch, self.memory)

    def step(self, x, h):
        """One step for a batch: x (B, obs), h (B, memory) -> mean (B, act), new h."""
        z = self.encoder(normalised(x, self.mean, self.std))
        h = self.gru(z[None], h[None])[1][0]
        return self.head(torch.cat([z, h], -1)), h

    def forward(self, x, h, starts):
        """Sequences: x (T, B, obs), h (B, memory) before the first step, starts (T, B)
        1 where an episode begins, which clears the memory before that step."""
        z = self.encoder(normalised(x, self.mean, self.std))
        # One GRU call per stretch with no episode starting inside it: stepping it one
        # step at a time from Python made the update twice as slow as the rollout.
        cuts = [0, *(torch.nonzero(starts[1:].any(1)).flatten() + 1).tolist(), x.shape[0]]
        out = []
        for a, b in zip(cuts[:-1], cuts[1:]):
            o, h = self.gru(z[a:b], (h * (1.0 - starts[a])[:, None])[None])
            out.append(o)
            h = h[0]
        return self.head(torch.cat([z, torch.cat(out)], -1)), h

    def std_now(self):
        return self.log_std.clamp(min=LOG_STD_MIN).exp()


def save_actor(actor: Actor, path: Path, privileged: bool = False):
    """privileged: the actor reads the policy input followed by sim.HoopEnv.privileged(),
    the true state. Such a one is a teacher for the sim only: it cannot fly the drone."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"algo": "ppo", "obs_dim": actor.obs_dim, "hidden": actor.hidden, "memory": actor.memory,
                "privileged": privileged, "actor": actor.state_dict()}, path)


def _resolve(path) -> Path:
    """The file SB3's load would open: it adds .zip when the path has none."""
    path = Path(path)
    if not path.exists() and path.with_name(path.name + ".zip").exists():
        return path.with_name(path.name + ".zip")
    return path


def is_ppo(path) -> bool:
    """SB3 saves policy.pth inside its zip; torch.save does not. Anything that is not a
    zip at all is left for SB3's load to report."""
    path = _resolve(path)
    if not zipfile.is_zipfile(path):
        return False
    with zipfile.ZipFile(path) as z:
        return "policy.pth" not in z.namelist()


def _saved(path) -> dict:
    return torch.load(_resolve(path), map_location="cpu", weights_only=True)


def is_privileged(path) -> bool:
    """A teacher that reads the sim's true state (save_actor's privileged)."""
    return bool(_saved(path).get("privileged", False))


def load(path) -> tuple[Actor, bool]:
    """The saved actor and whether it is privileged, from one read of the file."""
    saved = _saved(path)
    actor = Actor(saved["obs_dim"], hidden=saved["hidden"], memory=saved["memory"])
    actor.load_state_dict(saved["actor"])
    return actor.eval(), bool(saved.get("privileged", False))


def load_actor(path) -> Actor:
    return load(path)[0]


class Policy:
    """A saved actor flying deterministically, for evaluate, replay and labcheck.
    reset() clears its memory between flights; run_episodes and replay.fly call it."""

    def __init__(self, actor: Actor):
        self.actor = actor
        self.observation_size = actor.obs_dim
        self.reset()

    def reset(self):
        self.h = self.actor.initial(1)

    def __call__(self, obs) -> np.ndarray:
        obs = np.asarray(obs, np.float32)
        # An actor saved before the motion readings flies on today's input without them.
        if len(obs) == contract.POLICY_INPUT_DIM and self.actor.obs_dim == contract.SLOTS_V1_DIM:
            obs = contract.slots_v1(obs)
        with torch.no_grad():
            mean, self.h = self.actor.step(torch.as_tensor(obs)[None], self.h)
        return mean[0].numpy().clip(-1.0, 1.0)


class PrivilegedPolicy(Policy):
    """A privileged actor flying in the sim: bind(env) after every reset, and each call
    reads env.privileged() after the policy input, as it was trained."""

    def bind(self, env):
        self.env = env

    def __call__(self, obs) -> np.ndarray:
        return super().__call__(np.concatenate([np.asarray(obs, np.float32), self.env.privileged()]))


def load_policy(path) -> Policy:
    actor, privileged = load(path)
    return (PrivilegedPolicy if privileged else Policy)(actor)
