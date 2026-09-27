from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import pytest
import torch

from src.front_return_library import generate, FrontReturnLibrary, restore_payload
from src.return_practice import ReturnSnapshotLibrary, ReturnPracticeEnv

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def captured(tmp_path_factory):
    source=ROOT/'runs/return-route-40k-20260927/best_model.zip'
    if not torch.cuda.is_available() or not source.exists():
        pytest.skip('CUDA and preserved local source required')
    directory=tmp_path_factory.mktemp('front-bank')/'bank'
    generate(source,directory,split='train',episodes=2)
    return directory


def test_full_history_restore_reproduces_public_observations_and_next_physics(captured):
    bank=FrontReturnLibrary(captured)
    assert len(bank.records)==4 and set(bank.phases)=={'return_side','return_front'}
    for record in bank.records:
        a,b=bank.restore(record['id']),bank.restore(record['id'])
        assert a['observation'].shape==(115,) and a['observation'][99]==0
        assert a['observation'][108]==1  # actual earlier front sighting, never synthesized
        np.testing.assert_array_equal(a['observation'],b['observation'])
        action=np.array([.2,.1,-.1,.3],np.float32)
        first=a['environment'].step(action)
        second=b['environment'].step(action)
        np.testing.assert_array_equal(first[0],second[0])
        assert first[1:4]==second[1:4]
        assert first[0].shape==(115,)
        np.testing.assert_array_equal(a['environment'].unwrapped.pos,b['environment'].unwrapped.pos)


def test_practice_restores_front_history_without_reset_and_excludes_historical_passes(captured):
    bank=FrontReturnLibrary(captured)
    record=next(r for r in bank.records if r['phase']=='return_front')
    expected=bank.restore(record['id'])
    env=ReturnPracticeEnv(bank,horizon=3)
    obs,info=env.reset(options={'snapshot_id':record['id']})
    np.testing.assert_array_equal(obs,expected['observation'])
    assert obs[109]>0 and info['passes']==0 and info['historical_passes']>=1
    for _ in range(3):
        obs,_,terminated,truncated,info=env.step(np.zeros(4,np.float32))
    assert truncated and not terminated and obs.shape==(115,) and info['steps']==3


def test_legacy_loader_and_corrupted_extra_history_fail_closed(captured):
    with pytest.raises(ValueError,match='Incompatible snapshot'):
        ReturnSnapshotLibrary(captured)
    bank=FrontReturnLibrary(captured)
    snapshot=bank.restore(bank.records[0]['id'])
    snapshot['observation'][110]+=.1
    with pytest.raises(ValueError,match='history disagree'):
        restore_payload(pickle.dumps(snapshot))


def test_same_physical_entries_and_original_prefix_as_old_genuine_bank(captured):
    old=ReturnSnapshotLibrary(ROOT/'return-libraries/train-v1')
    new=FrontReturnLibrary(captured)
    lookup={(r['seed'],r['phase']):r for r in old.records}
    for record in new.records:
        previous=lookup[(record['seed'],record['phase'])]
        assert record['source_step']==previous['source_step']
        a=new.restore(record['id'])
        b=old.restore(previous['id'])
        np.testing.assert_array_equal(a['observation'][:100],b['observation'])
        np.testing.assert_array_equal(a['environment'].unwrapped.pos,b['environment'].unwrapped.pos)
        assert a['environment'].unwrapped.np_random.bit_generator.state==b['environment'].unwrapped.np_random.bit_generator.state


@pytest.mark.parametrize('split,episodes,seed', [('invalid',2,None),('train',0,None),('development',2,7000000)])
def test_invalid_collection_partitions_rejected_before_creating_output(tmp_path,split,episodes,seed):
    with pytest.raises(ValueError):
        generate('unused.zip',tmp_path/'unused',split=split,episodes=episodes,seed_start=seed)
    assert not (tmp_path/'unused').exists()
