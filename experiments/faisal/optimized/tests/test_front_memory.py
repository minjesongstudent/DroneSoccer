"""Public-input provenance and preservation before any training integration."""
from copy import deepcopy
import pickle
import gymnasium as gym
import numpy as np
import pytest
from src.front_memory import FrontSightingMemory, FrontMemoryObservation


def frame(visible=1., marker=1.):
    value = np.zeros(84, np.float32)
    value[64:72] = [.2, -.1, .4, .8, .1, .9, visible, marker]
    return value


def test_back_detection_retains_front_reference_and_motion_then_front_reanchors():
    memory = FrontSightingMemory()
    original = frame()
    memory.reset(original)
    back = frame(marker=0.)
    back[64] = -.8
    current = memory.advance([0.,1.,0.,.2],back)
    np.testing.assert_array_equal(current[84:92],original[64:72])
    assert current[93] > 0 and np.linalg.norm(current[94:97]) > 0
    assert current[97] != 0
    front = frame()
    front[64] = -.5
    current = memory.advance([0.,1.,0.,.2],front)
    np.testing.assert_array_equal(current[84:92],front[64:72])
    np.testing.assert_array_equal(current[93:97],np.zeros(4))
    assert np.linalg.norm(memory._velocity)>0  # sighting does not stop nominal motion


def test_back_only_start_does_not_invent_an_earlier_front_view():
    memory = FrontSightingMemory()
    value = memory.reset(frame(marker=0.))
    assert not value[84:].any()
    value = memory.advance([1.,1.,0.,1.],frame(marker=0.))
    assert not value[84:].any()


def test_reset_and_pickle_preserve_only_genuine_history():
    memory = FrontSightingMemory()
    memory.reset(frame())
    for _ in range(30):
        memory.advance([.3,.2,.1,.4],frame(marker=0.))
    restored = pickle.loads(pickle.dumps(memory))
    np.testing.assert_array_equal(memory.advance([0.,1.,0.,0.],frame(0,0)),
                                  restored.advance([0.,1.,0.,0.],frame(0,0)))
    assert not restored.reset(frame(0,0))[84:].any()


class PublicEnv(gym.Env):
    def __init__(self, info=None):
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(100,),np.float32)
        self.action_space=gym.spaces.Box(-1.,1.,(4,),np.float32)
        self.info=info or {}
    def reset(self, *, seed=None, options=None):
        return np.concatenate([frame(),np.arange(16,dtype=np.float32)]),deepcopy(self.info)
    def step(self, action):
        return np.concatenate([frame(marker=0),np.arange(16,dtype=np.float32)]),2.,False,False,deepcopy(self.info)


def test_prefix_reward_and_task_index_unchanged_despite_arbitrary_diagnostic_truth():
    one=FrontMemoryObservation(PublicEnv({'position':[100,200,300]}))
    two=FrontMemoryObservation(PublicEnv({'position':[-9,-8,-7]}))
    a,_=one.reset(); b,_=two.reset()
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:100],PublicEnv().reset()[0])
    assert a.shape==(115,) and a[99]==15
    a,r,t,u,_=one.step([.1,.2,.3,.4]); b,*_=two.step([.1,.2,.3,.4])
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:100],PublicEnv().step(None)[0])
    assert r==2. and not t and not u


def test_old_midflight_reset_is_rejected_instead_of_fabricating_history():
    with pytest.raises(ValueError,match='authentic saved'):
        FrontMemoryObservation(PublicEnv({'return_practice':True})).reset()
