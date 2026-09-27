import numpy as np
import pytest
from src.ring_estimate import RingCentreEstimate,MetricRingObservation
from src.params import Camera
from src.front_memory import FrontMemoryObservation
from src.mission_observation import MissionObservation
from src.guided_curriculum import GuidedLessonEnv

def public():
 x=np.zeros(115,np.float32);x[102]=.45;x[106:109]=1;x[114]=1
 return x

def test_pinhole_front_distance_offset_odometry_and_rotation():
 e=RingCentreEstimate();x=public();a=e.encode(x)
 np.testing.assert_array_equal(a[:115],x)
 np.testing.assert_allclose(a[115:118]*3,[1.05,0,0],atol=1e-6);assert a[118]==1
 x[110]=.2;x[113]=1;x[114]=0
 np.testing.assert_allclose(e.encode(x)[115:118]*3,[0,-.45,0],atol=1e-6)

@pytest.mark.parametrize('index,value',[(108,0),(107,0),(106,0),(102,0),(102,3),(100,3),(110,1)])
def test_missing_or_clipped_history_is_not_invented(index,value):
 x=public();x[index]=value
 np.testing.assert_array_equal(RingCentreEstimate().encode(x)[115:],np.zeros(4))

@pytest.mark.parametrize('kwargs',[{'diameter':0},{'diameter':float('nan')},{'camera':Camera(pitch_deg=10)},{'camera':Camera(fx=0)}])
def test_invalid_nominal_calibration_rejected(kwargs):
 with pytest.raises(ValueError):RingCentreEstimate(**kwargs)

def test_wrapper_preserves_public_prefix_rewards_and_task():
 def make():return FrontMemoryObservation(MissionObservation(GuidedLessonEnv(stage=4),single_pass=False))
 old=make();new=MetricRingObservation(make())
 a,ia=old.reset(seed=8);b,ib=new.reset(seed=8);np.testing.assert_array_equal(a,b[:115])
 for _ in range(15):
  action=np.array([.1,.2,0,.1],np.float32)
  a,ra,ta,tra,ia=old.step(action);b,rb,tb,trb,ib=new.step(action)
  np.testing.assert_array_equal(a,b[:115]);assert (ra,ta,tra)==(rb,tb,trb)
  assert new.observation_space.contains(b) and b[99]==0
 old.close();new.close()


@pytest.mark.parametrize('sine,cosine',[(0,0),(2,0),(0,.5)])
def test_invalid_rotation_cannot_produce_valid_estimate(sine,cosine):
 x=public();x[113:115]=[sine,cosine]
 assert RingCentreEstimate().encode(x)[118]==0


def test_nominal_calibration_is_copied_and_input_is_unchanged():
 camera=Camera();estimate=RingCentreEstimate(camera=camera)
 x=public();before=x.copy();expected=estimate.encode(x)
 camera.fx=1;camera.offset_forward=100
 np.testing.assert_array_equal(estimate.encode(x),expected)
 np.testing.assert_array_equal(x,before)


@pytest.mark.parametrize('x',[np.zeros(100),np.full(115,np.nan)])
def test_wrong_contract_rejected(x):
 with pytest.raises(ValueError):RingCentreEstimate().encode(x)
