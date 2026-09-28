import json
import pickle
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.success_history import SuccessHistoryLibrary, SuccessHistoryPractice, FIT, sha

BANK = Path('D:/DroneSoccerData/diagnostics/success-return-bank-v1')
pytestmark = pytest.mark.skipif(not BANK.exists(), reason='Local verified bank required')


def test_fit_split_and_authentic_clone():
    bank = SuccessHistoryLibrary(BANK)
    assert len(bank.records) == 20
    assert {r['seed'] for r in bank.records} == set(FIT)
    env = SuccessHistoryPractice(bank)
    for row in bank.records:
        original = bank.restore(row['id'])
        obs, info = env.reset(options={'snapshot_id': row['id']})
        assert np.array_equal(obs, original['observation'])
        assert pickle.dumps(env.env) == pickle.dumps(original['environment'])
        assert info['passes'] == 0 and info['historical_passes'] == 1
        assert not info['practice_success'] and not info['full_mission_success']


def test_reserved_and_horizon_guards():
    with pytest.raises(ValueError, match='evaluation_only'):
        SuccessHistoryLibrary(BANK, split='internal_check')
    bank = SuccessHistoryLibrary(BANK)
    bank.evaluation_only = True
    with pytest.raises(ValueError, match='training'):
        SuccessHistoryPractice(bank)
    with pytest.raises(ValueError, match='horizon'):
        SuccessHistoryPractice(bank, horizon=201, evaluation=True)


@pytest.mark.parametrize('damage', ['hash', 'history', 'step', 'split', 'version', 'duplicate'])
def test_rejects_corruption(tmp_path, damage):
    target = tmp_path / 'bank'
    shutil.copytree(BANK / 'snapshots', target / 'snapshots')
    shutil.copyfile(BANK / 'protocol.json', target / 'protocol.json')
    manifest = json.loads((BANK / 'bank_manifest.json').read_text())
    row = next(r for r in manifest['snapshots'] if r['split'] == 'fit')
    path = target / 'snapshots' / row['file']
    if damage == 'hash':
        path.write_bytes(path.read_bytes() + b'bad')
    elif damage in ('history', 'step'):
        saved = pickle.loads(path.read_bytes())
        if damage == 'history':
            saved['observation'][0] += .1
        else:
            saved['environment'].unwrapped.steps = 401
        payload = pickle.dumps(saved)
        path.write_bytes(payload)
        row['sha256'] = sha(payload)
    elif damage == 'split':
        row['split'] = 'internal_check'
    elif damage == 'version':
        manifest['version'] = 'unknown'
    else:
        manifest['snapshots'].append(row.copy())
    (target / 'bank_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        SuccessHistoryLibrary(target)


def test_success_requires_new_pass_and_survival():
    env = SuccessHistoryPractice(SuccessHistoryLibrary(BANK))
    env.reset(seed=5)
    env.elapsed = 200
    assert not env._info({'outcome': 'timeout'})['practice_success']
    env.unwrapped.passes += 1  # Scorer unit test only, never a flight result.
    env.elapsed = 199
    assert not env._info({'outcome': 'running'})['practice_success']
    env.elapsed = 200
    assert env._info({'outcome': 'timeout'})['practice_success']
    assert not env._info({'outcome': 'left_cage'})['practice_success']


@pytest.mark.skipif(not Path('D:/DroneSoccerData/diagnostics/success-return-bank-v2/result.json').exists(), reason='Expanded local bank required')
def test_expanded_bank_version_and_separation():
    from src.success_curriculum import BackwardScheduler, SuccessMixture
    bank = SuccessHistoryLibrary('D:/DroneSoccerData/diagnostics/success-return-bank-v2')
    assert len(bank.records) == 45 and len(bank.fit_seeds) == 9
    assert all(r['seed'] % 4 != 3 for r in bank.records)
    env = SuccessHistoryPractice(bank)
    _, info = env.reset(seed=1)
    assert info['bank_version'] == 'success-history-return-bank-v2'
    assert info['historical_passes'] == 1 and info['passes'] == 0
    with pytest.raises(ValueError, match='original FIT'):
        SuccessMixture(bank, BackwardScheduler())
