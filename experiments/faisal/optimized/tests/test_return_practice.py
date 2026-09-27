"""Authentic reset-state fidelity, provenance, and continuation semantics."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation
from src.params import Drone
from src.return_practice import (ReturnPracticeEnv, ReturnSnapshotLibrary,
                                _pack_snapshot, _restore_snapshot, generate_return_library)


@pytest.fixture(scope='module')
def checkpoint(tmp_path_factory):
    """Small saved neural-policy fixture, never production training examples."""
    torch.set_num_threads(1)
    env = MissionObservation(GuidedLessonEnv(stage=4), single_pass=False)
    model = SAC('MlpPolicy', env, device='cpu', seed=4,
                policy_kwargs={'net_arch': [16, 16]}, buffer_size=16)
    with torch.no_grad():
        for parameter in model.actor.parameters():
            parameter.zero_()
        model.actor.mu.bias[1] = torch.atanh(torch.tensor(.6))
        model.actor.log_std.bias.fill_(-4.)
    path = tmp_path_factory.mktemp('return-policy')/'policy.zip'
    model.save(path)
    return path


@pytest.fixture(scope='module')
def collected(checkpoint, tmp_path_factory):
    directory = tmp_path_factory.mktemp('return-library')/'train'
    manifest = generate_return_library(checkpoint, directory, episodes=32, device='cpu')
    assert manifest['phase_counts'].get('return_side', 0) > 0
    return directory, manifest


def test_collection_provenance_reproducibility_and_no_invented_later_phases(checkpoint, collected, tmp_path):
    directory, manifest = collected
    assert manifest['source_sha256'] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest['split'] == 'train' and manifest['seed_start'] == 2_000_000
    assert sum(manifest['episode_outcomes'].values()) == 32
    # This saved fixture flies straight: it cannot invent a later return leg.
    assert 'return_front' not in manifest['phase_counts']
    assert all(r['historical_passes'] == 1 and r['source_step'] <= 400 for r in manifest['snapshots'])
    repeated = generate_return_library(checkpoint, tmp_path/'repeat', episodes=32, device='cpu')
    assert repeated['snapshots'] == manifest['snapshots']
    assert repeated['source_sha256'] == manifest['source_sha256']
    bank = ReturnSnapshotLibrary(directory)
    assert bank.bank_sha256 == hashlib.sha256((directory/'manifest.json').read_bytes()).hexdigest()


def test_restoration_preserves_full_history_velocity_rng_and_action_delay(checkpoint, collected):
    seed = collected[1]['snapshots'][0]['seed']
    model = SAC.load(checkpoint, device='cpu')
    env = MissionObservation(GuidedLessonEnv(stage=4, drone=Drone(delay_steps=2)), single_pass=False)
    obs, _ = env.reset(seed=seed)
    for _ in range(300):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        if info['passes'] > 0 or terminated or truncated:
            break
    assert info['passes'] == 1 and not terminated and not truncated
    assert np.linalg.norm(env.unwrapped.vel) > .1
    clone = _restore_snapshot(_pack_snapshot(env, obs, info))
    restored = clone['environment']
    np.testing.assert_array_equal(clone['observation'], obs)
    np.testing.assert_array_equal(restored.unwrapped.vel, env.unwrapped.vel)
    assert len(restored.unwrapped._pending) == len(env.unwrapped._pending) == 2
    for left, right in zip(restored.unwrapped._frames, env.unwrapped._frames):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(restored.unwrapped._actions, env.unwrapped._actions):
        np.testing.assert_array_equal(left, right)
    assert restored.unwrapped.np_random.bit_generator.state == env.unwrapped.np_random.bit_generator.state
    assert restored.unwrapped.np_random.uniform() == env.unwrapped.np_random.uniform()
    for action in np.random.default_rng(9).uniform(-.1, .1, (8, 4)):
        expected, actual = env.step(action), restored.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:4] == expected[1:4]
        np.testing.assert_array_equal(restored.unwrapped.pos, env.unwrapped.pos)
        np.testing.assert_array_equal(restored.env.memory._velocity, env.env.memory._velocity)


def test_local_counts_unchanged_reward_and_bootstrapable_practice_cap(collected):
    bank = ReturnSnapshotLibrary(collected[0])
    record = bank.records[0]
    expected_env = bank.restore(record['id'])['environment']
    practice = ReturnPracticeEnv(bank, horizon=1)
    observation, info = practice.reset(seed=1, options={'snapshot_id': record['id']})
    assert observation.shape == (100,) and observation[-1] == 0
    assert info['steps'] == info['passes'] == 0
    assert info['historical_passes'] == 1 and info['source_step'] > 0
    assert info['outcome'] == 'running' and info['return_practice']
    assert not info['practice_success'] and not info['full_mission_success']
    action = np.zeros(4, dtype=np.float32)
    expected = expected_env.step(action)
    actual = practice.step(action)
    np.testing.assert_array_equal(actual[0], expected[0])
    assert actual[1] == expected[1] and not actual[2] and actual[3]
    assert actual[4]['outcome'] == 'timeout'
    assert actual[4]['physical_steps'] == record['source_step']+1
    assert actual[4]['steps'] == 1 and actual[4]['passes'] == 0
    assert not actual[4]['full_mission_success']
    with pytest.raises(RuntimeError, match='Reset return practice'):
        practice.step(action)
    # SB3's vector wrapper must keep this cap bootstrapable, not a crash done.
    vector = DummyVecEnv([lambda: ReturnPracticeEnv(bank, horizon=1)])
    vector.reset()
    _, _, done, infos = vector.step(action.reshape(1, 4))
    assert done[0] and infos[0]['TimeLimit.truncated']
    assert 'terminal_observation' in infos[0]


def test_restores_are_independent_and_reset_selection_is_reproducible(collected):
    first, second = ReturnPracticeEnv(collected[0]), ReturnPracticeEnv(collected[0])
    obs_a, info_a = first.reset(seed=31)
    obs_b, info_b = second.reset(seed=31)
    assert info_a['snapshot_id'] == info_b['snapshot_id']
    np.testing.assert_array_equal(obs_a, obs_b)
    first.step(np.ones(4, np.float32))
    np.testing.assert_array_equal(second.env.unwrapped._policy_input(), obs_b[:84])
    again, info_again = first.reset(seed=31)
    np.testing.assert_array_equal(again, obs_b)
    assert info_again['passes'] == 0 and info_again['steps'] == 0
    with pytest.raises(ValueError, match='never captured'):
        first.reset(options={'phase': 'return_front'})


@pytest.mark.parametrize('seed,split', [(5_000_000, 'train'), (7_000_000, 'development'),
                                       (2_000_000, 'development'), (8_000_000, 'train')])
def test_eval_and_training_seed_ranges_cannot_mix(checkpoint, tmp_path, seed, split):
    with pytest.raises(ValueError, match='dedicated'):
        generate_return_library(checkpoint, tmp_path/'invalid', seed_start=seed, split=split, device='cpu')
    assert not (tmp_path/'invalid').exists()


def test_development_library_uses_its_own_seeds(checkpoint, tmp_path):
    manifest = generate_return_library(checkpoint, tmp_path/'development', episodes=32,
                                      split='development', device='cpu')
    assert manifest['split'] == 'development' and manifest['seed_start'] == 8_000_000
    bank = ReturnSnapshotLibrary(tmp_path/'development')
    assert all(8_000_000 <= r['seed'] < 9_000_000 for r in bank.records)


def test_library_rejects_tampering_wrong_versions_and_empty_states(collected, tmp_path):
    directory = tmp_path/'tampered'
    shutil.copytree(collected[0], directory)
    first = collected[1]['snapshots'][0]['file']
    (directory/first).write_bytes((directory/first).read_bytes()+b'changed')
    with pytest.raises(ValueError, match='checksum'):
        ReturnSnapshotLibrary(directory)
    manifest = deepcopy(collected[1])
    manifest['reward_version'] = 'different'
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='reward_version'):
        ReturnSnapshotLibrary(directory)
    manifest = deepcopy(collected[1])
    manifest['snapshots'] = []
    (directory/'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='No genuine return states'):
        ReturnSnapshotLibrary(directory)
