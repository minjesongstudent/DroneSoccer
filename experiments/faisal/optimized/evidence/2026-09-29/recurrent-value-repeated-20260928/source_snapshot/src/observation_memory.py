"""Camera/action memory that can be computed on the real control loop.

The original 84 policy inputs remain an exact prefix. The extra 15 values keep
the last valid ring detection and a nominal estimate of movement since that
detection. This is uncertain command dead reckoning, not measured position,
ring pose, a reward phase, or a controller. Its estimates become stale during
long periods without a detection. No environment state or ``info`` enters it.
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import gymnasium as gym
import numpy as np

from src import contract
from src.params import Drone


VERSION = 'camera-command-memory-v1'
BASE_OBS_DIM = contract.POLICY_INPUT_DIM
MEMORY_DIM = 15
POLICY_INPUT_DIM = BASE_OBS_DIM + MEMORY_DIM
AGE_SCALE_SECONDS = 60.0
DISPLACEMENT_SCALE_METRES = 3.0
MEMORY_FIELDS = (*('last_' + name for name in contract.FIELDS), 'has_seen',
                 'age_fraction', 'forward_displacement', 'left_displacement',
                 'up_displacement', 'sin_heading_change', 'cos_heading_change')

_FRAME_LOW = np.array([-contract.CLIP, -contract.CLIP, 0., 0., -1., -1., 0., 0.],
                      dtype=np.float32)
_FRAME_HIGH = np.array([contract.CLIP, contract.CLIP, contract.CLIP, 1., 1., 1., 1., 1.],
                       dtype=np.float32)
MEMORY_LOW = np.concatenate([_FRAME_LOW, [0., 0., -1., -1., -1., -1., -1.]]).astype(np.float32)
MEMORY_HIGH = np.concatenate([_FRAME_HIGH, np.ones(7)]).astype(np.float32)


class CameraCommandMemory:
    """Portable estimator driven only by detections and issued RC commands.

    ``nominal_drone`` is a fixed calibration/design configuration, copied at
    construction. In particular it must never be the simulator's randomized
    per-episode dynamics. Units before normalization are metres and seconds.
    Displacement uses the camera heading at the last detection: forward, left,
    up. The nominal response continues across detections; only its coordinate
    anchor resets. Episode reset also clears pending commands and response.
    """

    def __init__(self, nominal_drone: Drone | None = None):
        self.nominal = replace(nominal_drone if nominal_drone is not None else Drone())
        d = self.nominal
        if (not np.isfinite([d.dt, d.tau, d.max_speed, d.max_yaw_rate_deg]).all()
                or d.dt <= 0 or d.tau <= 0 or d.max_speed <= 0
                or d.max_yaw_rate_deg <= 0 or int(d.delay_steps) != d.delay_steps
                or d.delay_steps < 0):
            raise ValueError('Nominal dynamics must have positive finite rates and a nonnegative integer delay')
        self._clear()

    def _clear(self):
        self._last = np.zeros(contract.OBS_DIM, dtype=np.float32)
        self._has_seen = False
        self._age = 0.0
        self._displacement = np.zeros(3, dtype=np.float64)
        self._velocity = np.zeros(3, dtype=np.float64)
        self._yaw = self._yaw_rate = 0.0
        self._pending = deque(np.zeros(contract.ACTION_DIM, dtype=np.float32)
                              for _ in range(int(self.nominal.delay_steps)))

    @staticmethod
    def _base(observation):
        base = np.asarray(observation, dtype=np.float32)
        if base.shape != (BASE_OBS_DIM,):
            raise ValueError(f'Expected the original {BASE_OBS_DIM}-element observation')
        return base

    def _detect(self, base):
        stop = contract.OBS_STACK * contract.OBS_DIM
        frame = base[stop-contract.OBS_DIM:stop]
        if frame[6] > .5 and np.isfinite(frame).all():
            self._last = np.clip(frame, _FRAME_LOW, _FRAME_HIGH).copy()
            self._has_seen = True
            self._age = 0.0
            self._displacement.fill(0.)
            # Rotate the existing response into the new last-sighting frame.
            # Dropping it would falsely claim that a new sighting stops motion.
            c, s = np.cos(self._yaw), np.sin(self._yaw)
            vx, vy = self._velocity[:2].copy()
            self._velocity[:2] = [c*vx+s*vy, -s*vx+c*vy]
            self._yaw = 0.0

    def reset(self, observation):
        self._clear()
        base = self._base(observation)
        self._detect(base)
        return self.encode(base)

    def advance(self, issued_action, observation):
        action = np.asarray(issued_action, dtype=np.float32)
        if action.shape != (contract.ACTION_DIM,) or not np.isfinite(action).all():
            raise ValueError('Expected four finite issued control commands')
        self._pending.append(np.clip(action, -1., 1.).copy())
        right, forward, up, yaw_cw = self._pending.popleft()
        d = self.nominal
        c, s = np.cos(self._yaw), np.sin(self._yaw)
        command = d.max_speed*np.array([forward*c+right*s, forward*s-right*c, up])
        alpha = 1.-np.exp(-d.dt/d.tau)
        self._velocity += alpha*(command-self._velocity)
        self._yaw_rate += alpha*(-float(yaw_cw)*np.deg2rad(d.max_yaw_rate_deg)-self._yaw_rate)
        self._displacement += self._velocity*d.dt
        self._yaw = (self._yaw+self._yaw_rate*d.dt+np.pi) % (2*np.pi)-np.pi
        if self._has_seen:
            self._age = min(AGE_SCALE_SECONDS, self._age+d.dt)
        base = self._base(observation)
        self._detect(base)
        return self.encode(base)

    def encode(self, observation):
        """Append current memory without advancing it or reading simulator data."""
        base = self._base(observation)
        extra = np.zeros(MEMORY_DIM, dtype=np.float32)
        if self._has_seen:
            extra[:contract.OBS_DIM] = self._last
            extra[8] = 1.
            extra[9] = self._age/AGE_SCALE_SECONDS
            extra[10:13] = np.clip(self._displacement/DISPLACEMENT_SCALE_METRES, -1., 1.)
            extra[13:15] = [np.sin(self._yaw), np.cos(self._yaw)]
        return np.concatenate([base, extra]).astype(np.float32)


class ObservationMemory(gym.ObservationWrapper):
    """Add portable camera/command memory, preserving ``unwrapped`` for viewers."""

    def __init__(self, env, *, nominal_drone: Drone | None = None):
        super().__init__(env)
        if not isinstance(env.observation_space, gym.spaces.Box) or env.observation_space.shape != (BASE_OBS_DIM,):
            raise ValueError(f'ObservationMemory requires the original {BASE_OBS_DIM}-element Box')
        self.memory = CameraCommandMemory(nominal_drone)
        self.observation_space = gym.spaces.Box(
            np.concatenate([env.observation_space.low, MEMORY_LOW]),
            np.concatenate([env.observation_space.high, MEMORY_HIGH]), dtype=np.float32)

    def observation(self, observation):
        return self.memory.encode(observation)

    def reset(self, *, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.memory.reset(observation), info

    def step(self, action):
        issued = np.clip(np.asarray(action, dtype=np.float32), -1., 1.).copy()
        observation, reward, terminated, truncated, info = self.env.step(issued)
        return self.memory.advance(issued, observation), reward, terminated, truncated, info


ObservationMemoryWrapper = ObservationMemory
