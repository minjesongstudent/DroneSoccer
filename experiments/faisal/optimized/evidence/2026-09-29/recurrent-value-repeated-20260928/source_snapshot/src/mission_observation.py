"""Camera/command memory plus the requested task, available outside simulation."""
import gymnasium as gym
import numpy as np

from src.observation_memory import ObservationMemory, POLICY_INPUT_DIM

VERSION = 'camera-command-memory-and-requested-task-v1'
OBSERVATION_DIM = POLICY_INPUT_DIM + 1


class MissionObservation(gym.Wrapper):
    def __init__(self, env, single_pass=None):
        super().__init__(ObservationMemory(env))
        self.single_pass_override = single_pass
        self.single_pass = False
        self.observation_space = gym.spaces.Box(
            np.append(self.env.observation_space.low, 0.).astype(np.float32),
            np.append(self.env.observation_space.high, 1.).astype(np.float32), dtype=np.float32)

    def _append(self, obs):
        return np.append(obs, float(self.single_pass)).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.single_pass = (bool(info.get('single_pass_task', False))
                            if self.single_pass_override is None else self.single_pass_override)
        return self._append(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        return self._append(obs), reward, term, trunc, info
