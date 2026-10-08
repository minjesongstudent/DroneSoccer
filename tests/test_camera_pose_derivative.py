from copy import deepcopy
import numpy as np
import pytest
from src.sim import HoopEnv
from src.camera_pose_derivative import query,consistency


def scene():
    env=HoopEnv(randomise=False,hanger=True);env.reset(seed=13)
    env.place([.35,1.,1.],0,[1.,1.,1.],[1.,0,0],hoop_diameter=.45,hanger_half_width=.08)
    return env


def test_exact_source_and_complete_state_preservation():
    e=scene();p=e.pos;h=e.heading;v=e.vel.copy();rng=deepcopy(e.np_random.bit_generator.state)
    expected=e.observe().copy();o,_=query(e,np.r_[p,h]);assert np.array_equal(o,expected)
    consistency(e,np.r_[p,h],np.array([.5,.5,.5,.5]))
    assert e.pos is p and e.heading==h and np.array_equal(e.vel,v)
    assert e.np_random.bit_generator.state==rng and e.steps==0


def test_exception_restores_pose(monkeypatch):
    e=scene();p=e.pos;h=e.heading
    def fail():raise RuntimeError('camera error')
    monkeypatch.setattr(e,'observe',fail)
    with pytest.raises(RuntimeError):query(e,np.array([.1,.2,.3,1.]))
    assert e.pos is p and e.heading==h


def test_smooth_visible_derivative_and_invisible_zero():
    e=scene();pose=np.r_[e.pos,e.heading];direction=np.array([1.,0,0,0])
    r=consistency(e,pose,direction)
    assert r['visible'] and r['stable'] and r['agrees']
    assert np.linalg.norm(r['jacobian'])>0
    pose[3]=np.pi;r=consistency(e,pose,direction)
    assert not r['visible'] and r['stable'] and r['agrees'] and not r['jacobian'].any()


@pytest.mark.parametrize('pose',[np.zeros(3),[0,0,0,float('nan')]])
def test_bad_pose_rejected(pose):
    with pytest.raises(ValueError):query(scene(),pose)


def test_bad_direction_rejected():
    with pytest.raises(ValueError):consistency(scene(),np.zeros(4),np.ones(4))


def test_invisible_zeros_do_not_hide_bad_visible_derivatives():
    from tools.check_camera_pose_derivative import decide
    visible=[dict(visible=True,stable=True,agrees=i<94,zero_invisible=False) for i in range(100)]
    invisible=[dict(visible=False,stable=True,agrees=True,zero_invisible=True) for _ in range(1000)]
    r=decide(visible+invisible)
    assert not r['supports_local_feedback_derivative'] and r['stable_visible_agreement']==.94
    visible[-1]['agrees']=True
    assert decide(visible+invisible)['supports_local_feedback_derivative']


def test_branch_crossings_and_insufficient_coverage_fail():
    from tools.check_camera_pose_derivative import decide
    rows=[dict(visible=True,stable=i<100,agrees=True,zero_invisible=False) for i in range(112)]
    assert not decide(rows)['supports_local_feedback_derivative']
    assert not decide(rows[:99])['supports_local_feedback_derivative']
