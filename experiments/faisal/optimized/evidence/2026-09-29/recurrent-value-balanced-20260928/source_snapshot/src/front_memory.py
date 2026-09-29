"""Experimental additive front-sighting memory; not enabled in live training.

Uses only camera detections and issued commands with fixed nominal dynamics.
Keeps the existing100 inputs byte-for-byte, including requested-task index99.
This estimate is not measured pose and cannot reveal unseen cage boundaries.
"""
import gymnasium as gym
import numpy as np

from src import contract
from src.observation_memory import CameraCommandMemory, MEMORY_LOW, MEMORY_HIGH

VERSION = 'public-front-sighting-memory-append-v1'
ORIGINAL_DIM = 100
TASK_INDEX = 99
OBSERVATION_DIM = 115


class FrontSightingMemory(CameraCommandMemory):
    def _detect(self, base):
        stop = contract.OBS_STACK * contract.OBS_DIM
        frame = base[stop-contract.OBS_DIM:stop]
        if frame[6] > .5 and frame[7] > .5 and np.isfinite(frame).all():
            super()._detect(base)


class FrontMemoryObservation(gym.Wrapper):
    """Normal-start wrapper; old return snapshots lack its earlier history.

    Training on authentic mid-flight resets requires a separately saved full
    wrapper history from a genuine source-policy flight. Old snapshots cannot
    initialize this memory from their current back view or simulator truth.
    """
    def __init__(self, env, *, nominal_drone=None):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape != (100,):
            raise ValueError('Front-memory append requires original100D public observations.')
        self.front_memory = FrontSightingMemory(nominal_drone)
        self.observation_space = gym.spaces.Box(
            np.concatenate([env.observation_space.low, MEMORY_LOW]),
            np.concatenate([env.observation_space.high, MEMORY_HIGH]), dtype=np.float32)

    def _append(self, observation):
        original = np.asarray(observation, dtype=np.float32)
        if original.shape != (100,):
            raise ValueError('Original100D prefix required.')
        memory = self.front_memory.encode(original[:84])[84:]
        return np.concatenate([original, memory])

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        if info.get('return_practice'):
            raise ValueError('Mid-flight resets require authentic saved front-memory history.')
        self.front_memory.reset(np.asarray(observation)[:84])
        return self._append(observation), info

    def step(self, action):
        issued = np.clip(np.asarray(action, dtype=np.float32), -1., 1.).copy()
        observation, reward, terminated, truncated, info = self.env.step(issued)
        self.front_memory.advance(issued, np.asarray(observation)[:84])
        return self._append(observation), reward, terminated, truncated, info
