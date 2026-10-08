import numpy as np
import pytest

from tools.run_ring_position_estimator import position_targets
from src.camera_history import CameraHistory
from tools.run_recurrent_estimator import PROFILE


def trace():
    history=CameraHistory(PROFILE.lags)
    observations=np.asarray([history.update(np.zeros(8,np.float32),np.zeros(4,np.float32)) for _ in range(5)])
    return dict(actor_observations=observations,actions=np.zeros((5,4),np.float32),
                positions=np.zeros((5,3)),headings=np.zeros(5),hoop_centre=np.array([1.,2.,3.]))


def test_pre_action_ring_offset_and_body_axes():
    z=trace();z['positions'][1]=[.25,.5,1]
    obs,y,mask,visible=position_targets(z)
    np.testing.assert_array_equal(y[2],[750,-1500,2000])
    np.testing.assert_array_equal(mask,[False,False,True,True,True])
    assert not y[:2].any() and not visible.any()
    np.testing.assert_array_equal(obs,z['actor_observations'])


def test_heading_rotation():
    z=trace();z['headings'][1]=np.pi/2
    np.testing.assert_allclose(position_targets(z)[1][2],[2000,1000,3000],atol=.001)


def test_current_and_future_truth_cannot_change_earlier_input_or_target():
    z=trace();before=position_targets(z)
    z['positions'][2:]+=50;z['headings'][2:]+=2
    after=position_targets(z)
    np.testing.assert_array_equal(before[0],after[0])
    np.testing.assert_array_equal(before[1][:3],after[1][:3])
    assert not np.array_equal(before[1][3:],after[1][3:])


@pytest.mark.parametrize('centre',[np.zeros(2),np.array([0.,float('nan'),1.])])
def test_invalid_label_geometry_is_rejected(centre):
    z=trace();z['hoop_centre']=centre
    with pytest.raises(ValueError):position_targets(z)
