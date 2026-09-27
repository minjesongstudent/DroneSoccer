"""Sequence integrity across circular overwrite, terminal clocks and resumes."""
from copy import deepcopy
import pickle

import gymnasium as gym
import numpy as np
import pytest
import torch

from src.retrace_replay import RetraceReplayBuffer, VERSION


OBSERVATION_SPACE = gym.spaces.Box(-np.inf, np.inf, (100,), np.float32)
ACTION_SPACE = gym.spaces.Box(-1., 1., (4,), np.float32)


def buffer(capacity=16, **kwargs):
    return RetraceReplayBuffer(capacity, OBSERVATION_SPACE, ACTION_SPACE, device='cpu', **kwargs)


def add(replay, state, *, task=0, done=False, timeout=False, next_state=None, metadata=None, gaussian=None):
    observation = np.zeros((1, 100), np.float32)
    observation[0, 0], observation[0, 99] = state, task
    following = observation.copy()
    following[0, 0] = state+1 if next_state is None else next_state
    gaussian = np.array([.1, -.2, .3, -.4] if gaussian is None else gaussian, np.float32)
    info = {'behavior_log_prob': -.75-float(state)/100., 'behavior_gaussian_actions': gaussian,
            'TimeLimit.truncated': timeout}
    if metadata is not None:
        info = metadata
    replay.add(observation, following, np.tanh(gaussian).reshape(1, 4),
               np.array([state+.25], np.float32), np.array([done]), [info])


def test_newest_tail_and_unfilled_storage_never_invent_successors():
    replay = buffer()
    for state in range(3):
        add(replay, state)
    data = replay._get_samples(np.array([0, 1, 2]))
    assert data.seq_valid.tolist() == [[True]*3+[False]*5, [True]*2+[False]*6, [True]+[False]*7]
    assert data.seq_insertion_ids[0].tolist() == [0, 1, 2, -1, -1, -1, -1, -1]
    assert data.seq_rewards.shape == data.seq_dones.shape == data.seq_behavior_log_prob.shape == (3, 8, 1)
    assert data.seq_observations.shape == (3, 8, 100)
    assert data.seq_actions.shape == data.seq_behavior_gaussian_actions.shape == (3, 8, 4)
    assert data.seq_valid.dtype == data.seq_episode_ends.dtype == data.seq_timeouts.dtype == torch.bool
    assert data.seq_insertion_ids.dtype == torch.int64
    assert data.discounts is None
    assert not data.dones.any() and not data.seq_dones.any()
    assert data.seq_next_observations[0, 2, 0].item() == 3.
    assert not data.seq_observations[~data.seq_valid].any()
    assert not data.seq_behavior_log_prob[~data.seq_valid].any()
    np.testing.assert_array_equal(data.observations.numpy(), replay.observations[:3, 0])
    np.testing.assert_array_equal(data.actions.numpy(), data.seq_actions[:, 0].numpy())
    np.testing.assert_array_equal(data.rewards.numpy(), data.seq_rewards[:, 0].numpy())
    np.testing.assert_array_equal(data.next_observations.numpy(), data.seq_next_observations[:, 0].numpy())


def test_circular_wrap_can_follow_new_data_but_never_reenter_overwritten_past():
    replay = buffer(5)
    for state in range(9):
        add(replay, state)
    # Occupied chronological order is slot4->0->1->2->3, serials4..8.
    data = replay._get_samples(np.array([4, 0, 3]))
    assert data.seq_insertion_ids.tolist() == [
        [4, 5, 6, 7, 8, -1, -1, -1], [5, 6, 7, 8, -1, -1, -1, -1], [8, -1, -1, -1, -1, -1, -1, -1]]
    for row in range(3):
        length = int(data.seq_valid[row].sum())
        np.testing.assert_array_equal(data.seq_observations[row, :length, 0].numpy(),
                                      data.seq_insertion_ids[row, :length].numpy())
    assert data.seq_behavior_log_prob[0, 0, 0].item() == pytest.approx(-.79)


@pytest.mark.parametrize('timeout', [False, True])
def test_both_episode_endings_stop_trace_but_only_true_terminal_stops_bootstrap(timeout):
    replay = buffer()
    add(replay, 0)
    add(replay, 1, done=True, timeout=timeout)
    add(replay, 2)  # Identical boundary observation deliberately tests IDs/end flags.
    data = replay._get_samples(np.array([0, 1, 2]))
    assert data.seq_valid.sum(1).tolist() == [2, 1, 1]
    assert data.seq_episode_ends[0, :2, 0].tolist() == [False, True]
    assert data.seq_timeouts[0, 1, 0].item() == timeout
    assert data.seq_dones[0, 1, 0].item() == (not timeout)
    assert data.dones[1, 0].item() == (not timeout)
    assert data.seq_next_observations[0, 1, 0].item() == 2.
    assert data.seq_episode_ids[0, 1] != data.seq_episode_ids[2, 0]


def test_explicit_stream_break_separates_identical_observations_without_false_terminal():
    replay = buffer()
    add(replay, 0)
    replay.break_stream()
    add(replay, 1)
    data = replay._get_samples(np.array([0, 1]))
    assert data.seq_valid.sum(1).tolist() == [1, 1]
    assert not data.seq_dones.any() and not data.seq_episode_ends.any()
    assert data.seq_stream_ids[0, 0] != data.seq_stream_ids[1, 0]
    assert data.seq_episode_ids[0, 0] != data.seq_episode_ids[1, 0]
    assert data.seq_next_observations[0, 0, 0].item() == 1.


@pytest.mark.parametrize('gap', ['serial', 'episode', 'stream', 'observation'])
def test_contiguity_checks_fail_closed_at_a_gap(gap):
    replay = buffer()
    for state in range(4):
        add(replay, state)
    if gap == 'serial':
        replay.insertion_ids[2] += 10
    elif gap == 'episode':
        replay.episode_ids[2] += 10
    elif gap == 'stream':
        replay.stream_ids[2] += 10
    else:
        replay.observations[2, 0, 7] = .5
    data = replay._get_samples(np.array([0]))
    assert data.seq_valid[0].tolist() == [True, True]+[False]*6
    assert data.seq_next_observations[0, 1, 0].item() == 2.
    assert not data.seq_dones.any()


def test_balancing_applies_to_anchors_and_every_offset_keeps_its_episode_task():
    replay = buffer(7)
    for state in range(12):
        add(replay, state, task=(state//3) % 2, done=state % 3 == 2)
    np.random.seed(17)
    data = replay.sample(200)
    assert data.observations[:, 99].sum().item() == 100
    for row in range(200):
        valid = data.seq_valid[row]
        assert torch.all(data.seq_observations[row, valid, 99] == data.observations[row, 99])
        ids = data.seq_insertion_ids[row, valid]
        assert torch.all(ids[1:]-ids[:-1] == 1)
        assert len(torch.unique(data.seq_episode_ids[row, valid])) == 1
    assert set(data.seq_insertion_ids[:, 0].tolist()) == set(range(5, 12))


def test_single_task_population_and_length_one_are_supported():
    replay = buffer(sequence_length=1)
    for state in range(4):
        add(replay, state, task=1)
    data = replay.sample(24)
    assert data.seq_valid.shape == (24, 1) and data.seq_valid.all()
    assert data.observations[:, 99].sum().item() == 24
    np.testing.assert_array_equal(data.seq_dones[:, 0].numpy(), data.dones.numpy())


def test_disk_roundtrip_and_new_stream_do_not_join_unfinished_old_episode():
    replay = buffer(5)
    for state in range(7):
        add(replay, state)
    restored = pickle.loads(pickle.dumps(replay))
    assert restored.replay_version == restored.sequence_replay_version == VERSION
    for name in ('insertion_ids', 'stream_ids', 'episode_ids', 'behavior_log_probs',
                 'behavior_gaussian_actions', 'observations', 'next_observations'):
        np.testing.assert_array_equal(getattr(replay, name), getattr(restored, name))
    restored.break_stream()
    add(restored, 7)
    # Old serial6 in slot1 has a valid next observation but no trace successor.
    data = restored._get_samples(np.array([1, 2]))
    assert data.seq_insertion_ids[:, 0].tolist() == [6, 7]
    assert data.seq_valid.sum(1).tolist() == [1, 1]
    assert not data.dones.any()
    assert data.next_observations[0, 0].item() == 7.


def test_reset_invalidates_old_metadata_without_reusing_identity_numbers():
    replay = buffer(4)
    for state in range(6):
        add(replay, state)
    next_id = replay._next_insertion_id
    replay.reset()
    assert replay.size() == 0 and np.all(replay.insertion_ids == -1)
    assert not replay.behavior_log_probs.any() and not replay._single_pass_slots.any()
    with pytest.raises(ValueError, match='empty'):
        replay.sample(2)
    add(replay, 100, task=1)
    data = replay.sample(8)
    assert data.seq_insertion_ids[:, 0].tolist() == [next_id]*8
    assert data.seq_valid.sum(1).tolist() == [1]*8


@pytest.mark.parametrize('metadata', [
    {}, {'behavior_log_prob': -1.},
    {'behavior_log_prob': np.zeros(4), 'behavior_gaussian_actions': np.zeros(4)},
    {'behavior_log_prob': np.nan, 'behavior_gaussian_actions': np.zeros(4)},
    {'behavior_log_prob': -1., 'behavior_gaussian_actions': np.zeros(3)},
    {'behavior_log_prob': -1., 'behavior_gaussian_actions': np.full(4, np.inf)},
    {'behavior_log_prob': -1., 'behavior_gaussian_actions': np.zeros(4)},
])
def test_invalid_behavior_metadata_is_rejected_without_any_replay_mutation(metadata):
    replay = buffer(4)
    add(replay, 0)
    before = deepcopy(replay.__dict__)
    with pytest.raises(ValueError):
        add(replay, 1, metadata=metadata)
    for name, value in before.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(replay, name), value)
    assert replay.pos == 1 and replay._next_insertion_id == 1


def test_saturated_actions_keep_exact_pre_tanh_values_and_joint_density():
    replay = buffer()
    gaussian = np.array([12., -15., .2, -.3], np.float32)
    density = np.float32(13.125)
    add(replay, 0, gaussian=gaussian,
        metadata={'behavior_log_prob': density, 'behavior_gaussian_actions': gaussian})
    data = replay.sample(1)
    assert data.actions[0, :2].tolist() == [1., -1.]
    np.testing.assert_array_equal(data.seq_behavior_gaussian_actions[0, 0].numpy(), gaussian)
    assert data.seq_behavior_log_prob[0, 0, 0].item() == density


@pytest.mark.parametrize('kwargs', [{'n_envs': 2}, {'optimize_memory_usage': True},
                                   {'handle_timeout_termination': False},
                                   {'sequence_length': 0}, {'sequence_length': 9}])
def test_unsupported_modes_fail_closed(kwargs):
    with pytest.raises(ValueError):
        buffer(**kwargs)


def test_vecnormalize_and_wrong_actor_or_action_contracts_fail_closed():
    replay = buffer(); add(replay, 0)
    with pytest.raises(ValueError, match='VecNormalize'):
        replay.sample(1, env=object())
    with pytest.raises(ValueError, match='VecNormalize'):
        replay._get_samples(np.array([0]), env=object())
    with pytest.raises(ValueError, match='100-value'):
        RetraceReplayBuffer(4, gym.spaces.Box(-1., 1., (84,), np.float32), ACTION_SPACE)
    with pytest.raises(ValueError, match='normalized'):
        RetraceReplayBuffer(4, OBSERVATION_SPACE, gym.spaces.Box(-2., 2., (4,), np.float32))


def test_false_timeout_and_reset_observation_task_change_are_rejected():
    replay = buffer()
    with pytest.raises(ValueError, match='also end'):
        add(replay, 0, timeout=True)
    obs = np.zeros((1, 100), np.float32)
    reset_obs = obs.copy(); reset_obs[0, 99] = 1.
    with pytest.raises(ValueError, match='true next'):
        replay.add(obs, reset_obs, np.zeros((1, 4), np.float32), np.zeros(1), np.ones(1),
                   [{'behavior_log_prob': -1., 'behavior_gaussian_actions': np.zeros(4)}])
    assert replay.size() == 0
