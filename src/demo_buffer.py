"""SB3's replay buffer plus a permanent one for src.teacher's flights (train.py --demos).

In the ordinary buffer, demos are just the oldest rows: their share of each batch falls
from ~98% at the first update to 20% once the buffer fills, and they are overwritten by
1M steps. That is before the curriculum reaches the real start, which is what they show
(firstpass-curriculum reached levels 0.5-0.75 at 0.4-3.6M steps). Here they sit in a
buffer of their own that nothing overwrites, and every batch takes demo_share of its
rows from it and the rest from the agent's own flying, as RLPD does (Ball et al. 2023,
at 50%).

It lives in its own module so that the pickled buffer in a --resume snapshot names a
class that every entry point can import (train.py runs as __main__).
"""
from __future__ import annotations

import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class DemoReplayBuffer(ReplayBuffer):
    """The agent's own steps in this buffer, the teacher's in self.demos."""

    def __init__(self, *args, demo_size: int = 1, demo_share: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_share = demo_share
        # Sized in steps, as SB3's own: it divides by n_envs for the rows.
        self.demos = ReplayBuffer(demo_size, self.observation_space, self.action_space,
                                  device=self.device, n_envs=self.n_envs,
                                  handle_timeout_termination=self.handle_timeout_termination)

    def sample(self, batch_size: int, env=None) -> ReplayBufferSamples:
        n_demo = round(batch_size * self.demo_share) if self.demos.size() else 0
        own = super().sample(batch_size - n_demo, env=env)
        if n_demo == 0:
            return own
        demo = self.demos.sample(n_demo, env=env)
        return ReplayBufferSamples(*(None if a is None else th.cat([a, b]) for a, b in zip(own, demo)))
