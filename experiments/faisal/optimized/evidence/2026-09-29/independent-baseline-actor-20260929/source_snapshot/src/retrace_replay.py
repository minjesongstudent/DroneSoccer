"""Preparation only: public-input, task-balanced contiguous replay sequences.

The collector must supply the exact joint behavior log density and pre-tanh
Gaussian sample for the action it executed. This buffer never reconstructs
behavior probabilities, imports historical bank actions, or changes actor
observations. Sequence boundaries and Bellman terminal masks are distinct:
timeouts end a sequence but retain their true next observation for bootstrap.

Call ``break_stream`` before storing a transition following an external reset,
disk resume into a new environment, or dropped/unrecorded collection step.
Insertion serials detect circular overwrite; they cannot detect an environment
step the collector never reported. No learning rule is activated by this file.
"""
from __future__ import annotations

from typing import NamedTuple

from gymnasium import spaces
import numpy as np
import torch

from src.anchored_policy import TaskBalancedReplayBuffer


VERSION = 'task-balanced-soft-retrace-sequences-v1'
BEHAVIOR_DENSITY_VERSION = 'joint-squashed-gaussian-with-pre-tanh-sample-v1'
MAX_SEQUENCE_LENGTH = 8


class RetraceReplaySamples(NamedTuple):
    # The ordinary SAC fields refer to each selected anchor transition.
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor | None
    # Sequence scalars have shape [batch, length, 1]; masks/IDs use [batch, length].
    seq_observations: torch.Tensor
    seq_actions: torch.Tensor
    seq_next_observations: torch.Tensor
    seq_dones: torch.Tensor
    seq_rewards: torch.Tensor
    seq_episode_ends: torch.Tensor
    seq_timeouts: torch.Tensor
    seq_behavior_log_prob: torch.Tensor
    seq_behavior_gaussian_actions: torch.Tensor
    seq_valid: torch.Tensor
    seq_insertion_ids: torch.Tensor
    seq_episode_ids: torch.Tensor
    seq_stream_ids: torch.Tensor


class RetraceReplayBuffer(TaskBalancedReplayBuffer):
    """Balance anchors 50/50 by task, then follow their real recorded successors.

    Supports one environment, raw float32 100D observations, normalized four-
    component actions, explicit next-observation storage, and lengths 1..8.
    Short sequences are retained, including terminal anchors and newest tails.
    Invalid suffixes are padded with zeros (IDs -1), never another episode.
    """
    def __init__(self, *args, sequence_length=MAX_SEQUENCE_LENGTH, **kwargs):
        if (isinstance(sequence_length, bool) or int(sequence_length) != sequence_length
                or not 1 <= sequence_length <= MAX_SEQUENCE_LENGTH):
            raise ValueError('sequence_length must be an integer between 1 and 8.')
        super().__init__(*args, **kwargs)
        if (not isinstance(self.observation_space, spaces.Box)
                or self.observation_space.dtype != np.float32
                or not isinstance(self.action_space, spaces.Box)
                or self.action_space.shape != (4,)
                or not np.all(self.action_space.low == -1.)
                or not np.all(self.action_space.high == 1.)):
            raise ValueError('Retrace replay requires float32 public100D and normalized Box(-1,1,(4,)) actions.')
        if not self.handle_timeout_termination:
            raise ValueError('Retrace replay requires separate timeout and true-terminal masks.')
        self.sequence_length = int(sequence_length)
        self.replay_version = VERSION
        self.sequence_replay_version = VERSION
        self.behavior_density_version = BEHAVIOR_DENSITY_VERSION
        self.behavior_log_probs = np.zeros((self.buffer_size, 1), np.float32)
        self.behavior_gaussian_actions = np.zeros((self.buffer_size, 1, 4), np.float32)
        self.insertion_ids = np.full(self.buffer_size, -1, np.int64)
        self.episode_ids = np.full(self.buffer_size, -1, np.int64)
        self.stream_ids = np.full(self.buffer_size, -1, np.int64)
        self._next_insertion_id = 0
        self._episode_id = 0
        self._stream_id = 0

    def break_stream(self):
        """Separate future data without changing existing rewards or done masks."""
        self._stream_id += 1
        self._episode_id += 1

    def reset(self):
        super().reset()
        # BaseBuffer construction does not currently call reset; this guard
        # also keeps initialization safe if an upstream constructor does so.
        if hasattr(self, 'insertion_ids'):
            self.insertion_ids.fill(-1)
            self.episode_ids.fill(-1)
            self.stream_ids.fill(-1)
            self.behavior_log_probs.fill(0.)
            self.behavior_gaussian_actions.fill(0.)
            self._single_pass_slots.fill(False)
            self.break_stream()

    @staticmethod
    def _finite(value, shape, name):
        with np.errstate(over='ignore', invalid='ignore'):
            array = np.asarray(value, dtype=np.float32)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f'{name} requires finite shape {shape}.')
        return array

    def add(self, obs, next_obs, action, reward, done, infos):
        # Validate every new field before the cyclic slot, position, or labels
        # are changed. A rejected sample must leave the existing replay intact.
        if len(infos) != 1 or not isinstance(infos[0], dict):
            raise ValueError('One behavior metadata dictionary is required.')
        info = infos[0]
        if 'behavior_log_prob' not in info or 'behavior_gaussian_actions' not in info:
            raise ValueError('Missing exact behavior_log_prob or behavior_gaussian_actions.')
        density = np.asarray(info['behavior_log_prob'])
        if density.shape not in ((), (1,)):
            raise ValueError('behavior_log_prob must be one joint scalar, not per-action densities.')
        density = self._finite(density.reshape(1), (1,), 'behavior_log_prob')
        gaussian = self._finite(info['behavior_gaussian_actions'], (4,), 'behavior_gaussian_actions')
        obs = self._finite(obs, (1, self.observation_dim), 'Observation')
        next_obs = self._finite(next_obs, (1, self.observation_dim), 'Next observation')
        action = self._finite(action, (1, 4), 'Action')
        reward = self._finite(reward, (1,), 'Reward')
        done = self._finite(done, (1,), 'Episode end')
        if np.any((done != 0.) & (done != 1.)):
            raise ValueError('Episode end must be boolean.')
        if obs[0, 99] not in (0., 1.) or next_obs[0, 99] != obs[0, 99]:
            raise ValueError('Public task index99 must be binary and unchanged in the true next observation.')
        if np.any(np.abs(action) > 1.):
            raise ValueError('Recorded action must be normalized to [-1,1].')
        # Tanh kernels can differ by a few float32 ulps across CPU/CUDA. Do not
        # invert rounded/saturated actions; preserve the original latent sample.
        if not np.allclose(np.tanh(gaussian), action[0], atol=2e-6, rtol=0.):
            raise ValueError('behavior_gaussian_actions does not generate the recorded action.')
        timeout = np.asarray(info.get('TimeLimit.truncated', False))
        if timeout.shape != () or timeout.item() not in (False, True):
            raise ValueError('TimeLimit.truncated must be one boolean.')
        if bool(timeout) and not bool(done[0]):
            raise ValueError('A timeout must also end its episode.')
        if max(self._next_insertion_id, self._episode_id, self._stream_id) >= np.iinfo(np.int64).max:
            raise OverflowError('Replay identity counter exhausted.')
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.behavior_log_probs[slot, 0] = density[0]
        self.behavior_gaussian_actions[slot, 0] = gaussian
        self.insertion_ids[slot] = self._next_insertion_id
        self.episode_ids[slot] = self._episode_id
        self.stream_ids[slot] = self._stream_id
        self._next_insertion_id += 1
        if bool(done[0]):
            self._episode_id += 1

    def sample(self, batch_size, env=None):
        if env is not None:
            raise ValueError('Retrace behavior densities require raw observations; VecNormalize is unsupported.')
        return super().sample(batch_size, env=None)

    def _sequence_indices(self, batch_inds):
        anchors = np.asarray(batch_inds)
        if (anchors.ndim != 1 or not len(anchors)
                or not np.issubdtype(anchors.dtype, np.integer)
                or np.any(anchors < 0) or np.any(anchors >= self.size())):
            raise ValueError('Sequence anchors must identify occupied replay slots.')
        if np.any(self.insertion_ids[anchors] < 0):
            raise ValueError('Sequence anchor lacks its insertion identity.')
        slots = (anchors[:, None]+np.arange(self.sequence_length)[None, :]) % self.buffer_size
        valid = np.zeros(slots.shape, dtype=bool)
        valid[:, 0] = True
        for step in range(1, self.sequence_length):
            previous, current = slots[:, step-1], slots[:, step]
            valid[:, step] = (
                valid[:, step-1]
                & ~self.dones[previous, 0].astype(bool)
                & (self.insertion_ids[current] == self.insertion_ids[previous]+1)
                & (self.episode_ids[current] == self.episode_ids[previous])
                & (self.stream_ids[current] == self.stream_ids[previous])
                & (self._single_pass_slots[current] == self._single_pass_slots[previous])
                & np.all(self.next_observations[previous, 0] == self.observations[current, 0], axis=1))
        return slots, valid

    def _get_samples(self, batch_inds, env=None):
        if env is not None:
            raise ValueError('Retrace behavior densities require raw observations; VecNormalize is unsupported.')
        slots, valid = self._sequence_indices(batch_inds)
        first = super()._get_samples(np.asarray(batch_inds), env=None)

        def values(array, transform=None):
            selected = array[slots, 0]
            if transform is not None:
                selected = transform(selected)
            if selected.ndim == 2:
                selected = selected[..., None]
            return self.to_torch(np.where(valid[..., None], selected, np.zeros((), dtype=selected.dtype)))

        def identities(array):
            return self.to_torch(np.where(valid, array[slots], -1))

        return RetraceReplaySamples(
            first.observations, first.actions, first.next_observations, first.dones,
            first.rewards, getattr(first, 'discounts', None),
            values(self.observations), values(self.actions), values(self.next_observations),
            values(self.dones, lambda value: value*(1-self.timeouts[slots, 0])), values(self.rewards),
            values(self.dones, lambda value: value.astype(bool)),
            values(self.timeouts, lambda value: value.astype(bool)),
            values(self.behavior_log_probs), values(self.behavior_gaussian_actions),
            self.to_torch(valid), identities(self.insertion_ids),
            identities(self.episode_ids), identities(self.stream_ids))
