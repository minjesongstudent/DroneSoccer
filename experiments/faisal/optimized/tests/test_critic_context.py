"""Privileged critic data must never modify actor inputs or terminal state."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src.critic_context import (CONTEXT_DIM, CONTEXT_SLICES, OMITTED_FRAME_LAGS,
                                CriticContextWrapper, critic_context)
from src.curriculum import REWARD_VERSION
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation, VERSION as OBSERVATION_VERSION
from src.params import Drone
from src.return_mixture import ReturnMixtureEnv
from src.return_practice import (VERSION as BANK_VERSION, ReturnPracticeEnv,
                                ReturnSnapshotLibrary, _code_hashes, _pack_snapshot,
                                _restore_snapshot)


def mission(stage=4, **kwargs):
    return MissionObservation(GuidedLessonEnv(stage=stage, **kwargs), single_pass=stage < 4)


def actor_input(env):
    return env._append(env.env.memory.encode(env.unwrapped._policy_input()))


@pytest.fixture(scope='module')
def bank(tmp_path_factory):
    """Small real-step snapshot fixture; never production training material."""
    env = mission()
    for seed in range(2_000_000, 2_000_016):
        observation, info = env.reset(seed=seed)
        for _ in range(400):
            observation, _, terminated, truncated, info = env.step(np.array([0., .6, 0., 0.], np.float32))
            if info['passes'] or terminated or truncated:
                break
        if info['passes'] and not terminated and not truncated:
            break
    else:
        raise AssertionError('Fixture failed to reach a genuine first passage.')
    payload = _pack_snapshot(env, observation, info)
    directory = tmp_path_factory.mktemp('critic-return-bank')
    (directory/'snapshot-0000.pkl').write_bytes(payload)
    record = {'id': 'snapshot-0000', 'file': 'snapshot-0000.pkl',
              'sha256': hashlib.sha256(payload).hexdigest(), 'phase': 'return_side',
              'seed': seed, 'source_step': env.unwrapped.steps, 'historical_passes': 1}
    manifest = {'version': BANK_VERSION, 'split': 'train', 'reward_version': REWARD_VERSION,
                'observation_version': OBSERVATION_VERSION, 'observation_dim': 100,
                'source_sha256': 'fixture', 'seed_start': seed, 'seed_end': seed,
                'minimum_remaining_steps': 200, 'snapshots': [record],
                'code_sha256': _code_hashes(),
                'geometry': {name: asdict(getattr(env.unwrapped, name))
                             for name in ('cage', 'drone', 'hoop', 'camera', 'pad')}}
    (directory/'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
    return ReturnSnapshotLibrary(directory)


def test_context_read_has_no_side_effects_and_preserves_actor_contract():
    reference, wrapped = mission(), CriticContextWrapper(Monitor(mission()))
    with pytest.raises(RuntimeError, match='Reset'):
        _ = wrapped.current_critic_context
    expected, _ = reference.reset(seed=123)
    actual, info = wrapped.reset(seed=123)
    assert actual.dtype == np.float32 and actual.shape == (100,)
    assert actual.tobytes() == expected.tobytes()
    assert wrapped.observation_space == reference.observation_space
    before_rng = deepcopy(wrapped.unwrapped.np_random.bit_generator.state)
    before_state = deepcopy(wrapped.env.env)
    context = critic_context(wrapped)
    assert CONTEXT_DIM == 249 and context.shape == (CONTEXT_DIM,)
    assert context.dtype == np.float32 and np.isfinite(context).all()
    np.testing.assert_array_equal(context, info['critic_context'])
    np.testing.assert_array_equal(context, wrapped.current_critic_context)
    assert before_rng == wrapped.unwrapped.np_random.bit_generator.state
    np.testing.assert_array_equal(actor_input(before_state), actor_input(wrapped.env.env))
    for action in np.random.default_rng(7).uniform(-.2, .2, (8, 4)).astype(np.float32):
        wanted = reference.step(action)
        old = wrapped.current_critic_context
        got = wrapped.step(action)
        assert got[0].tobytes() == wanted[0].tobytes()
        assert got[1:4] == wanted[1:4]
        np.testing.assert_array_equal(got[4]['critic_context'], old)
        np.testing.assert_array_equal(got[4]['next_critic_context'], critic_context(wrapped))
        np.testing.assert_array_equal(reference.unwrapped._actions[-1], wrapped.unwrapped._actions[-1])
    returned = wrapped.current_critic_context
    returned[:] = -999
    assert not np.any(wrapped.current_critic_context == -999)


@pytest.mark.parametrize('change,block', [
    ('velocity', 'motion'), ('phase', 'route'), ('pending_crossing', 'route'),
    ('gate', 'route'), ('hidden_frame', 'omitted_frames'), ('memory_velocity', 'memory_state'),
    ('memory_displacement', 'memory_state'), ('delay', 'dynamics'),
])
def test_hidden_aliases_change_only_critic_features_not_actor_input(change, block):
    first = mission(); first.reset(seed=17)
    second = deepcopy(first)
    base, memory = second.unwrapped, second.env.memory
    if change == 'velocity':
        base.vel[0] += .2
    elif change == 'phase':
        base._lap_phase = 'return_side'
    elif change == 'pending_crossing':
        base._crossed_pending = True
    elif change == 'gate':
        base._return_gate = base._gates[0].copy()
    elif change == 'hidden_frame':
        base._frames[-4] = base._frames[-4].copy()
        base._frames[-4][0] += .6
    elif change == 'memory_velocity':
        memory._velocity[1] += .2
    elif change == 'memory_displacement':
        first.env.memory._displacement[0] = 6.
        memory._displacement[0] = 9.
    elif change == 'delay':
        from collections import deque
        base.dyn = replace(base.dyn, delay_steps=2)
        base._pending = deque(np.zeros(4, np.float32) for _ in range(2))
    np.testing.assert_array_equal(actor_input(first), actor_input(second))
    a, b = critic_context(first), critic_context(second)
    assert not np.array_equal(a[CONTEXT_SLICES[block]], b[CONTEXT_SLICES[block]])
    # An arbitrary deterministic actor using only its public input still emits
    # the exact same action. The companion context never enters this mapping.
    weight = np.random.default_rng(8).normal(size=(4, 100)).astype(np.float32)
    np.testing.assert_array_equal(np.tanh(weight @ actor_input(first)),
                                  np.tanh(weight @ actor_input(second)))
    if change == 'hidden_frame':
        start = CONTEXT_SLICES['omitted_frames'].start+OMITTED_FRAME_LAGS.index(3)*8
        assert b[start]-a[start] == pytest.approx(.2)


def test_restored_snapshot_context_and_continuation_stay_identical(bank):
    restored = bank.restore('snapshot-0000')
    base = restored['environment']
    second = _restore_snapshot(_pack_snapshot(base, restored['observation'], restored['info']))
    np.testing.assert_array_equal(critic_context(base), critic_context(second['environment']))
    practice = CriticContextWrapper(Monitor(ReturnPracticeEnv(bank, horizon=200)))
    observation, _ = practice.reset(seed=9, options={'snapshot_id': 'snapshot-0000'})
    assert observation.tobytes() == restored['observation'].tobytes()
    expected_context = critic_context(base)
    actual_context = critic_context(practice)
    # The physical/memory state is exactly restored. Only local-cap metadata differs.
    mask = np.ones(CONTEXT_DIM, dtype=bool); mask[CONTEXT_SLICES['clock']] = False
    np.testing.assert_array_equal(actual_context[mask], expected_context[mask])
    assert actual_context[CONTEXT_SLICES['clock']][2] == 1.
    for action in np.zeros((3, 4), np.float32):
        expected, actual = base.step(action), practice.step(action)
        assert expected[0].tobytes() == actual[0].tobytes()
        assert expected[1:4] == actual[1:4]
        np.testing.assert_array_equal(actual[4]['next_critic_context'][mask], critic_context(base)[mask])


def test_practice_truncation_keeps_post_state_before_vector_autoreset(bank):
    practice = CriticContextWrapper(Monitor(ReturnPracticeEnv(bank, horizon=1)))
    vector = DummyVecEnv([lambda: practice])
    vector.seed(11); vector.reset()
    expected = deepcopy(practice.env.env)
    expected.step(np.zeros(4, np.float32))
    terminal = critic_context(expected)
    _, _, done, infos = vector.step(np.zeros((1, 4), np.float32))
    assert done[0] and infos[0]['TimeLimit.truncated']
    np.testing.assert_array_equal(infos[0]['next_critic_context'], terminal)
    assert not np.array_equal(terminal, practice.current_critic_context)
    assert infos[0]['terminal_observation'].shape == (100,)
    saved = infos[0]['next_critic_context'].copy()
    vector.reset()
    np.testing.assert_array_equal(infos[0]['next_critic_context'], saved)
    vector.close()


class CrashOnReset(gym.Wrapper):
    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.unwrapped.pos[0] = .151
        self.unwrapped.vel[0] = -.4
        return observation, info


def test_crash_context_is_terminal_position_not_autoreset_position():
    wrapped = CriticContextWrapper(CrashOnReset(mission()))
    vector = DummyVecEnv([lambda: wrapped]); vector.seed(12); vector.reset()
    reference = deepcopy(wrapped.env)
    reference.step(np.zeros(4, np.float32))
    expected = critic_context(reference)
    _, _, done, infos = vector.step(np.zeros((1, 4), np.float32))
    assert done[0] and not infos[0]['TimeLimit.truncated']
    assert infos[0]['outcome'] == 'left_cage'
    assert infos[0]['next_critic_context'][CONTEXT_SLICES['wall_clearances']][0] < 0.
    np.testing.assert_array_equal(infos[0]['next_critic_context'], expected)
    assert not np.array_equal(expected, wrapped.current_critic_context)
    vector.close()


def test_mixture_active_branch_is_resolved_again_after_every_reset(bank):
    wrapped = CriticContextWrapper(Monitor(ReturnMixtureEnv(bank)))
    seen_practice = seen_regular = False
    for seed in range(30):
        observation, info = wrapped.reset(seed=seed)
        mixture = wrapped.env.env
        assert observation.shape == (100,)
        np.testing.assert_array_equal(info['critic_context'], critic_context(mixture.active))
        seen_practice |= info['return_practice']
        seen_regular |= not info['return_practice']
    assert seen_practice and seen_regular
    wrapped.close()


@pytest.mark.parametrize('target', ['true', 'nominal'])
def test_unsupported_delay_fails_closed(target):
    env = mission(drone=Drone(delay_steps=4)) if target == 'true' else mission()
    env.reset(seed=13)
    if target == 'nominal':
        env.env.memory.nominal.delay_steps = 4
    with pytest.raises(ValueError, match='delay must be an integer'):
        critic_context(env)


def test_invalid_shape_nonfinite_history_and_queue_mismatch_fail_closed():
    with pytest.raises(ValueError, match='100D'):
        CriticContextWrapper(GuidedLessonEnv(stage=4))
    env = mission(); env.reset(seed=1)
    env.unwrapped._frames[-4] = env.unwrapped._frames[-4].copy()
    env.unwrapped._frames[-4][0] = np.nan
    with pytest.raises(ValueError, match='camera history'):
        critic_context(env)
    env.reset(seed=1)
    env.unwrapped._pending.append(np.zeros(4, np.float32))
    with pytest.raises(ValueError, match='queue length'):
        critic_context(env)


def test_delay_queue_order_and_padding_are_explicit():
    env = mission(drone=Drone(delay_steps=2)); env.reset(seed=3)
    action = np.array([.1, .2, .3, .4], np.float32)
    env.step(action)
    pending = critic_context(env)[CONTEXT_SLICES['pending_commands']].reshape(3, 4)
    np.testing.assert_array_equal(pending[0], np.zeros(4))
    np.testing.assert_array_equal(pending[1], action)
    np.testing.assert_array_equal(pending[2], np.zeros(4))
