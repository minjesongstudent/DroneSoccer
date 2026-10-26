from __future__ import annotations

import torch as th
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples


class DemoReplayBuffer(ReplayBuffer):

    def __init__(self, *args, demo_size: int = 1, demo_share: float = 0.25, **kwargs):
        super().__init__(*args, **kwargs)
        self.demo_share = demo_share
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
