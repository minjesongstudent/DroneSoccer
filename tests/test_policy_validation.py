from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pytest

from src import contract
from src.camera_history import CameraHistory
from src.pass_audit import PassageAudit
from src.policy_runtime import LearnedPolicy, SPECS
from src.sim import HoopEnv


@pytest.mark.parametrize('name',list(SPECS))
def test_named_history_matches_documented_simulator_lags(name):
    spec=SPECS[name]
    env=HoopEnv()
    env.reset(seed=444)
    bridge=CameraHistory(spec.lags)
    previous=np.zeros(4,np.float32)
    rng=np.random.default_rng(2)
    for _ in range(100):
        actual=bridge.update(env.observe(),previous)
        expected=np.concatenate([*(env._frames[-1-l] for l in spec.lags),
                                 *env._actions,[float(env._unseen)]]).astype(np.float32)
        np.testing.assert_array_equal(actual,expected)
        assert actual.shape==(spec.inputs,)
        previous=rng.uniform(-.05,.05,4).astype(np.float32)
        _,_,done,cutoff,_=env.step(previous)
        if done or cutoff: break


def test_wrong_model_profile_refused_before_load():
    path=Path(__file__).resolve().parents[1]/'trainedModels/lags60-scratch.zip'
    with pytest.raises(ValueError,match='checksum'):
        LearnedPolicy(path,SPECS['seenflag-s1-level1'])


def fake():
    return SimpleNamespace(heading=0.,hoop_normal=np.array([1.,0,0]),
        hoop_centre=np.zeros(3),_half_box=np.array([.1,.1,.1]),
        hoop=SimpleNamespace(thickness=.06),steps=0,pos=np.array([-.2,0,0]))


def test_requires_whole_body_and_no_double_count():
    env=fake(); audit=PassageAudit();audit.update(env)
    env.pos[0]=.01;env.steps=1;audit.update(env,'passed')
    assert audit.count==0
    env.pos[0]=.12;env.steps=2;audit.update(env)
    assert audit.count==0
    env.pos[0]=.14;env.steps=3;audit.update(env)
    assert audit.count==1 and audit.steps==[3]
    audit.update(env,'passed');assert audit.count==1
    env.pos[0]=-.2;audit.update(env);assert audit.returns==1
    env.pos[0]=.2;env.steps=4;audit.update(env,'passed');assert audit.count==2


def test_collision_or_side_crossing_cannot_score():
    env=fake();audit=PassageAudit();audit.update(env)
    env.pos[0]=.2;audit.update(env,'running');assert audit.count==0
    env.pos[0]=-.2;audit.update(env)
    env.pos[0]=.01;audit.update(env,'passed')
    env.pos[0]=.2;audit.update(env,'hit',unsafe=True)
    assert audit.count==0


@pytest.mark.parametrize('lags',[(0,1), (1,1,0), (1.,0), (), (2,1)])
def test_invalid_history_spec(lags):
    with pytest.raises(ValueError): CameraHistory(lags)
