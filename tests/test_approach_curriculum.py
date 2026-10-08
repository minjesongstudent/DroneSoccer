import numpy as np
from src.sim import HoopEnv
from src.approach_curriculum import ApproachCurriculumEnv
from src.clearance_training import ClearanceTrainingEnv,public_features


def test_odd_seed_is_exact_original_reset():
    a=HoopEnv(level=1,randomise=True,hanger=True);b=ApproachCurriculumEnv(level=1,randomise=True,hanger=True)
    x,_=a.reset(seed=11240001);y,_=b.reset(seed=11240001)
    np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(a.pos,b.pos)
    assert a.dyn==b.dyn and not b.curriculum_start


def test_curriculum_fresh_history_and_start_ranges():
    env=ClearanceTrainingEnv(ApproachCurriculumEnv(level=1,randomise=True,hanger=True),'clearance',seed_base=11240000)
    for seed in range(11240000,11240100,2):
        obs,_=env.reset(seed=seed);truth=env.unwrapped
        delta=truth.pos-truth.hoop_centre
        assert -.650001<=delta@truth.hoop_normal<=-.449999
        assert abs(delta@truth._axis_h)<=.180001 and abs(delta[2])<=.060001
        assert truth.steps==0 and truth.curriculum_start
        np.testing.assert_array_equal(obs[72:84],np.zeros(12))
        for i in range(9):np.testing.assert_array_equal(obs[i*8:(i+1)*8],obs[64:72])
    env.close()
