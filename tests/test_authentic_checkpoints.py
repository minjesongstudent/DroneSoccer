"""Fixed actions here are serialization fixtures, never learned-flight evidence."""
import json
import pickle

import numpy as np
import pytest

from src import authentic_checkpoints as snapshots
from src.clearance_training import ClearanceTrainingEnv, PROFILE
from src.sim import HoopEnv

SEED = 11_123_456  # In-memory fixture only: no bank, model actions or study outputs.


def fixture_env(steps=6, randomise=True, trace_dir=None):
    env = ClearanceTrainingEnv(HoopEnv(randomise=randomise), 'clearance', trace_dir=trace_dir)
    env.reset(seed=SEED)
    for i in range(steps):
        action = np.array([.05,-.03,.02,.01],np.float32) if i%2 else np.zeros(4,np.float32)
        _,_,terminated,truncated,_ = env.step(action)
        assert not terminated and not truncated
    return env


def capture(env):
    return snapshots.capture_checkpoint(env, source_sha256=PROFILE.sha256, fit_seeds=[SEED])


def test_roundtrip_preserves_rng_delay_history_counters_and_detached_branches():
    env = fixture_env()
    # Nontrivial pending audit state is an explicit serialization unit fixture.
    env.audit.armed = True; env.audit.crossed = True
    env.audit.count = 1; env.audit.steps = [3]
    env.audit.returns = 1; env.audit.waiting_return = True
    before = snapshots.state_fingerprint(env)
    checkpoint = capture(env)
    a = snapshots.restore_checkpoint(checkpoint, trusted_local=True)
    b = snapshots.restore_checkpoint(checkpoint, trusted_local=True)
    snapshots.assert_state_equal(env,a.env)
    assert snapshots.state_fingerprint(env) == before
    assert a.public_prefix.shape == (6,85) and a.current_obs.shape == (85,)
    np.testing.assert_array_equal(a.public_prefix,env._trace['actor_observations'])
    np.testing.assert_array_equal(a.current_obs,env.current_obs)
    assert len(a.env.unwrapped._pending) == env.unwrapped.dyn.delay_steps
    assert a.env.unwrapped.np_random.bit_generator.state == env.unwrapped.np_random.bit_generator.state
    for _ in range(9):
        action = np.array([.03,-.02,.01,.04],np.float32)
        expected, actual = env.step(action),a.env.step(action)
        np.testing.assert_array_equal(expected[0],actual[0])
        assert expected[1:4] == actual[1:4]
        snapshots.assert_state_equal(env,a.env)
    assert snapshots.state_fingerprint(b.env) == before
    a.current_obs[0] += 1; a.public_prefix[0,0] += 1
    assert snapshots.state_fingerprint(a.env) == snapshots.state_fingerprint(env)
    metadata = checkpoint.metadata; metadata['source_step'] = -1
    assert checkpoint.metadata['source_step'] == 6


def test_restore_never_calls_reset_place_start_or_history_update(monkeypatch):
    from src.camera_history import CameraHistory
    checkpoint = capture(fixture_env())
    def forbidden(*args,**kwargs):
        raise AssertionError('Restore advanced/reset state')
    for name in ('reset','place','_start_episode'):
        monkeypatch.setattr(HoopEnv,name,forbidden)
    monkeypatch.setattr(CameraHistory,'update',forbidden)
    restored = snapshots.restore_checkpoint(checkpoint,trusted_local=True)
    assert restored.env.unwrapped.steps == 6


def test_original_episode_horizon_is_not_restarted():
    env = ClearanceTrainingEnv(HoopEnv(randomise=False),'clearance')
    env.reset(seed=SEED)
    for _ in range(3): env.step(np.zeros(4,np.float32))
    restored = snapshots.restore_checkpoint(capture(env),trusted_local=True)
    assert restored.env.unwrapped.steps == 3
    for i in range(597):
        expected = env.step(np.zeros(4,np.float32))
        actual = restored.env.step(np.zeros(4,np.float32))
        np.testing.assert_array_equal(expected[0],actual[0])
        assert expected[1:4] == actual[1:4]
        assert actual[3] is (i == 596)
    snapshots.assert_state_equal(env,restored.env)
    assert restored.env.unwrapped.steps == 600 and not restored.env._active
    with pytest.raises(ValueError,match='Active post-step'):
        capture(restored.env)


def test_logging_source_paths_are_refused_and_output_never_overwrites(tmp_path):
    with pytest.raises(ValueError,match='trace_dir'):
        capture(fixture_env(trace_dir=tmp_path/'source_traces'))
    checkpoint = capture(fixture_env())
    record = snapshots.write_checkpoint(tmp_path/'source.pkl',checkpoint)
    loaded = snapshots.read_checkpoint(record['path'],expected_manifest_sha256=record['manifest_sha256'],trusted_local=True)
    restored = snapshots.restore_checkpoint(loaded,trusted_local=True)
    assert restored.env.trace_dir is None
    with pytest.raises(FileExistsError): snapshots.write_checkpoint(tmp_path/'source.pkl',checkpoint)
    assert (tmp_path/'source.pkl').read_bytes() == checkpoint.payload


def test_payload_manifest_code_and_model_tamper_rejected_before_deserialization(tmp_path,monkeypatch):
    checkpoint = capture(fixture_env())
    record = snapshots.write_checkpoint(tmp_path/'source.pkl',checkpoint)
    def forbidden(*args,**kwargs): raise AssertionError('Unverified payload was deserialized')
    monkeypatch.setattr(pickle,'loads',forbidden)
    with pytest.raises(ValueError,match='trusted local'): snapshots.restore_checkpoint(checkpoint)
    bad = snapshots.Checkpoint(checkpoint.payload+b'bad',checkpoint.manifest)
    with pytest.raises(ValueError,match='payload hash'): snapshots.restore_checkpoint(bad,trusted_local=True)
    with pytest.raises(ValueError,match='manifest hash'):
        snapshots.read_checkpoint(record['path'],expected_manifest_sha256='0'*64,trusted_local=True)
    meta = checkpoint.metadata; meta['source_sha256'] = '0'*64
    with pytest.raises(ValueError,match='manifest'):
        snapshots.restore_checkpoint(snapshots.Checkpoint(checkpoint.payload,json.dumps(meta).encode()),trusted_local=True)
    meta = checkpoint.metadata; meta['code_sha256']['src/sim.py'] = '0'*64
    with pytest.raises(ValueError,match='source code'):
        snapshots.restore_checkpoint(snapshots.Checkpoint(checkpoint.payload,json.dumps(meta).encode()),trusted_local=True)


@pytest.mark.parametrize('seeds',[(7_000_000,), (9_410_000,), (SEED,SEED), (SEED+1,)])
def test_wrong_allocation_and_development_seeds_rejected(seeds):
    with pytest.raises(ValueError,match='FIT allocation'):
        snapshots.capture_checkpoint(fixture_env(),source_sha256=PROFILE.sha256,fit_seeds=seeds)


def test_wrong_model_current_observation_and_initial_boundary_refused():
    env = fixture_env()
    with pytest.raises(ValueError,match='source profile'):
        snapshots.capture_checkpoint(env,source_sha256='0'*64,fit_seeds=[SEED])
    env.current_obs[0] += .1
    with pytest.raises(ValueError,match='state disagree'): capture(env)
    with pytest.raises(ValueError,match='post-step'): capture(fixture_env(steps=0))


def test_clearance_eligibility_is_metadata_not_invented_phase():
    env = fixture_env()
    assert not capture(env).metadata['eligible_first_or_second_clearance']
    env.audit.count = 2; env._trace['clearance_events'][-1] = 1
    meta = capture(env).metadata
    assert meta['eligible_first_or_second_clearance'] and meta['remaining_steps'] == 594
    assert meta['historical_full_body_passes'] == 2 and meta['prefix_excludes_current']
    checkpoint = capture(env)
    meta = checkpoint.metadata; meta['historical_full_body_passes'] = 1
    with pytest.raises(ValueError,match='state/prefix'):
        snapshots.restore_checkpoint(snapshots.Checkpoint(checkpoint.payload,json.dumps(meta).encode()),trusted_local=True)
