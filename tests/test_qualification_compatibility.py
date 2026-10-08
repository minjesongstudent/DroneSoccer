"""Qualification compatibility integrity, original-evaluator parity and public bridge."""
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from src.camera_history import CameraHistory
from src.policy_runtime import LearnedPolicy, SPECS
from tools.evaluate_qualification_compatibility import (
    ROOT, DEV_SEED_MIN, LEGACY_FILES, IMPORT_REMAPS, run, snapshot, validate_range, verify_frozen,
)
from tools.qualification_legacy_worker import verify_manifest


@pytest.mark.parametrize('seed,episodes', [(5_000_000, 1), (7_000_000, 100),
    (10_099_999, 1), (10_100_999, 2), (DEV_SEED_MIN, 0), (DEV_SEED_MIN, 201),
    (DEV_SEED_MIN, True)])
def test_reserved_or_unbounded_evaluations_are_blocked(seed, episodes):
    with pytest.raises(ValueError):
        validate_range(seed, episodes)


@pytest.fixture(scope='module')
def evaluated(tmp_path_factory):
    out = tmp_path_factory.mktemp('legacy_compat') / 'run'
    result = run(out, episodes=1)
    return out, result


def test_frozen_sources_and_import_only_bridge(evaluated):
    out, result = evaluated
    manifest = json.loads((out / 'manifest.json').read_text())
    verify_frozen(out, manifest)
    execution = json.loads((out / 'execution.json').read_text())
    assert execution['returncode'] == 0 and all(execution['input_sources_unchanged'].values())
    for name in LEGACY_FILES:
        assert (out / 'source/legacy/src' / name).read_bytes() == (
            ROOT.parent / 'DroneSoccer-Optimized/src' / name).read_bytes()
    for name, (before, after) in IMPORT_REMAPS.items():
        original = (out / 'source/group_original/src' / name).read_bytes()
        assert (out / 'source/bridge/group_bridge' / name).read_bytes() == original.replace(
            before.encode(), after.encode())
    assert result['acceptance_attempted'] is False
    assert result['accepted_in_simulation'] is False
    assert result['physical_flight_validated'] is False
    assert result['protocols']['full_task']['geometry']['drone']['delay_steps'] == 0
    assert result['protocols']['full_task_delay_200ms']['geometry']['drone']['delay_steps'] == 2


def test_history_and_actions_match_actual_group85_runtime(evaluated):
    out, result = evaluated
    spec = SPECS['seenflag-s1-level1']
    policy = LearnedPolicy(ROOT / 'trainedModels/seenflag-s1-level1.zip', spec)
    for suite in result['suites']:
        trace = np.load(out / suite / f'{DEV_SEED_MIN}.npz')
        policy.reset()
        history = CameraHistory(spec.lags)
        previous = np.zeros(4, np.float32)
        for newest, actor_obs, recorded_action in zip(
                trace['public_camera'], trace['actor_observations'], trace['actions']):
            np.testing.assert_array_equal(actor_obs, history.update(newest, previous))
            actual_obs, actual_action = policy.predict(newest, previous)
            np.testing.assert_array_equal(actor_obs, actual_obs)
            np.testing.assert_array_equal(recorded_action, actual_action)
            previous = recorded_action
        assert trace['actor_observations'].shape[1] == 85
        np.testing.assert_array_equal(trace['actor_observations'][0, 72:84], np.zeros(12))


def test_original_unified_evaluator_outcome_parity(evaluated):
    """The original100-input wrappers only serve reference scoring here.

    The adapter deliberately reads their original newest8 prefix and constructs
    the documented85 history afresh. Their memory/task fields never enter SAC.
    This exercises the unmodified evaluator independently of the worker loop.
    """
    out, result = evaluated
    reference_script = r'''
import json, sys
from pathlib import Path
import numpy as np
import torch
root=Path(sys.argv[1])
sys.path[:0]=[str(root/'source/legacy'),str(root/'source/bridge')]
from group_bridge.policy_runtime import LearnedPolicy,SPECS
from src.unified_evaluation import evaluate
from src.params import CALIB_PATH
assert not CALIB_PATH.exists()
torch.set_num_threads(1)
class Adapter:
    num_timesteps=0
    def __init__(self):
        self.policy=LearnedPolicy(root/'source/models/seenflag-s1-level1.zip',SPECS['seenflag-s1-level1'])
        self.reset()
    def reset(self):
        self.policy.reset();self.previous=np.zeros(4,np.float32)
    def predict(self,observation,deterministic=True):
        assert deterministic
        _,action=self.policy.predict(observation[64:72],self.previous)
        self.previous=action.copy()
        return action,None
model=Adapter()
report=evaluate(model,0,episodes=1,seed_start=10100000,episode_reset=model.reset)
(root/'reference-parity.json').write_text(json.dumps(report,indent=2))
'''
    completed = subprocess.run([sys.executable, '-I', '-c', reference_script, str(out)],
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stderr
    reference = json.loads((out / 'reference-parity.json').read_text())
    for name, actual in result['suites'].items():
        expected = reference['suites'][name]
        for key in ('successful_episodes', 'success_rate', 'crash_rate', 'total_passes',
                    'episodes_with_pass_rate', 'episodes_with_two_passes',
                    'returned_to_side_episodes', 'returned_to_front_episodes', 'outcomes'):
            assert actual[key] == expected[key], (name, key)
        record = json.loads((out / name / 'episodes.jsonl').read_text())
        assert record['steps'] == expected['records'][0]['steps']
        assert record['outcome'] == expected['records'][0]['outcome']


def test_ambient_calibration_and_tampered_sources_rejected(tmp_path):
    out = tmp_path / 'fresh'
    manifest = snapshot(out, episodes=1)
    verify_manifest(out.resolve(), manifest)
    calibration = out / 'source/legacy/calib.npz'
    calibration.write_bytes(b'not a calibration')
    with pytest.raises(ValueError, match='calibration'):
        verify_manifest(out.resolve(), manifest)
    calibration.unlink()
    model = out / 'source/models/seenflag-s1-level1.zip'
    model.write_bytes(b'changed')
    with pytest.raises(ValueError, match='checksum'):
        verify_manifest(out.resolve(), manifest)
    with pytest.raises(FileExistsError):
        snapshot(out, episodes=1)


def test_worker_also_refuses_reserved_manifest(tmp_path):
    manifest = dict(kind='legacy compatibility DEVELOPMENT ONLY', training=False,
                    acceptance_attempted=False, episodes_per_suite=1, seed_start=7_000_000)
    with pytest.raises(ValueError, match='blocked'):
        verify_manifest(tmp_path.resolve(), manifest)
