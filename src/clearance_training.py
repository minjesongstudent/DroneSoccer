"""Isolated 85-input training adapter; no changes to the shared simulator.

Only reward accounting and diagnostic logs can read simulator truth. The actor
receives public camera features and its actually issued actions through the same
CameraHistory used by the deployable preview.
"""
from pathlib import Path
import json

import gymnasium as gym
import numpy as np

from src import contract
from src.camera_history import CameraHistory
from src.pass_audit import PassageAudit
from src.policy_runtime import SPECS


PROFILE = SPECS['seenflag-s1-level1']


def passage_reward_delta(arm, centre_crossed, full_clearances):
    if arm not in ('control', 'clearance'):
        raise ValueError('Unknown reward arm')
    if full_clearances not in (0, 1):
        raise ValueError('At most one full clearance per simulator step')
    if arm == 'control':
        return 0.0
    return contract.R_PASS * (full_clearances - int(centre_crossed))


def public_features(public):
    public = np.asarray(public, dtype=np.float32)
    if public.shape != (contract.POLICY_INPUT_DIM,):
        raise ValueError('Expected the public simulator observation')
    end = contract.OBS_STACK * contract.OBS_DIM
    return public[end-contract.OBS_DIM:end].copy()


class ClearanceTrainingEnv(gym.Wrapper):
    def __init__(self, env, arm, seed_base=11_000_000, trace_dir=None, guard_alignment=False):
        super().__init__(env)
        passage_reward_delta(arm, False, 0)
        self.arm = arm
        self.guard_alignment = guard_alignment
        self.seed_base = seed_base
        self.episode_index = 0
        self.trace_dir = Path(trace_dir) if trace_dir is not None else None
        if self.trace_dir is not None:
            self.trace_dir.mkdir(parents=True, exist_ok=False)
        self.observation_space = gym.spaces.Box(
            -np.inf, np.inf, (PROFILE.inputs,), dtype=np.float32)
        self._active = False

    def reset(self, *, seed=None, options=None):
        if self._active:
            self.flush_partial('external_reset')
        # Every new episode has an explicit FIT seed; SB3's algorithm seed never
        # silently changes the layout schedule. Explicit seeds are for unit tests.
        self.episode_seed = self.seed_base + self.episode_index if seed is None else int(seed)
        if not 11_000_000 <= self.episode_seed < 11_250_000:
            raise ValueError('Training reset seed outside the declared FIT range')
        self.episode_index += 1
        public, info = self.env.reset(seed=self.episode_seed, options=options)
        self.history = CameraHistory(PROFILE.lags)
        self.current_obs = self.history.update(public_features(public), np.zeros(4, np.float32))
        self.audit = PassageAudit()
        self.audit.update(self.env.unwrapped)
        self._centre_count = 0
        self._trace = {key: [] for key in (
            'actor_observations', 'next_actor_observations', 'actions',
            'base_rewards', 'training_rewards', 'positions', 'headings',
            'centre_events', 'clearance_events', 'terminated', 'truncated', 'alignment_penalties')}
        self._last_info = info
        self._active = True
        return self.current_obs.copy(), info

    def step(self, action):
        if not self._active:
            raise RuntimeError('Reset required before stepping')
        action = np.asarray(action, np.float32)
        if action.shape != (4,) or not np.isfinite(action).all() or np.any(abs(action) > 1):
            raise ValueError('Finite normalized learned action required')
        penalty=0.0
        if self.guard_alignment:
            from src.guard_alignment_reward import alignment_penalties
            state=self.env.unwrapped
            penalty=float(alignment_penalties(state,state.pos[None],np.array([state.heading]))[0])
        public, base_reward, terminated, truncated, info = self.env.step(action)
        truth = self.env.unwrapped
        # Native outcome becomes "timeout" on the final step, so count deltas
        # preserve a crossing that coincides with truncation.
        centre_crossed = info['passes'] > self._centre_count
        self._centre_count = info['passes']
        previous_count = self.audit.count
        self.audit.update(truth, 'passed' if centre_crossed else info['outcome'],
                          unsafe=terminated or info['wall_steps'] > 0)
        cleared = self.audit.count - previous_count
        reward = base_reward + passage_reward_delta(self.arm, centre_crossed, cleared) + penalty
        obs = self.history.update(public_features(public), action)
        if not np.isfinite(reward):
            raise FloatingPointError('Non-finite reward')
        values = dict(actor_observations=self.current_obs.copy(),
                      next_actor_observations=obs.copy(), actions=action.copy(),
                      base_rewards=base_reward, training_rewards=reward,
                      positions=truth.pos.copy(), headings=truth.heading,
                      centre_events=centre_crossed, clearance_events=cleared,
                      terminated=terminated, truncated=truncated, alignment_penalties=penalty)
        for key, value in values.items():
            self._trace[key].append(value)
        self.current_obs = obs
        self._last_info = info
        if terminated or truncated:
            self._write_episode('complete')
            self._active = False
        # Info is for logs only. It is never concatenated into actor observations.
        return obs.copy(), float(reward), terminated, truncated, info

    def _write_episode(self, status):
        if self.trace_dir is None or not self._trace['actions']:
            return
        truth = self.env.unwrapped
        name = f'{self.episode_index:06d}-{self.episode_seed}-{status}.npz'
        np.savez_compressed(self.trace_dir / name,
            **{k: np.asarray(v) for k, v in self._trace.items()},
            hoop_centre=truth.hoop_centre, hoop_normal=truth.hoop_normal,
            drone_half_box=truth._half_box)
        record = dict(seed=self.episode_seed, status=status, trace=name,
            steps=truth.steps, outcome=self._last_info['outcome'],
            centre_passes=self._centre_count, full_body_passes=self.audit.count,
            returns_to_front=self.audit.returns, wall_steps=truth.wall_steps,
            base_return=float(sum(self._trace['base_rewards'])),
            training_return=float(sum(self._trace['training_rewards'])))
        with (self.trace_dir / 'episodes.jsonl').open('a', encoding='utf-8') as log:
            log.write(json.dumps(record) + '\n')

    def flush_partial(self, status='budget_end'):
        if self._active:
            self._write_episode(status)
            self._active = False

    def close(self):
        self.flush_partial('closed')
        return super().close()
