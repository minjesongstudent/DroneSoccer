import pickle
import numpy as np
from src.clearance_training import ClearanceTrainingEnv
from src.twelve_action_history import TwelveActionObservation
from src.sim import HoopEnv
from tools.audit_privileged_value import diagnostic_truth


def test_truth_read_is_nonmutating_and_separate_from_public_actor_observation():
    env=TwelveActionObservation(ClearanceTrainingEnv(HoopEnv(randomise=True),'clearance'))
    prior_names=None
    for seed in [11210000,11210001]:
        obs,_=env.reset(seed=seed)
        before=pickle.dumps(env)
        truth,names=diagnostic_truth(env)
        assert pickle.dumps(env)==before
        assert obs.shape==(121,)
        assert np.isfinite(truth).all() and len(truth)==len(names)==len(set(names))
        if prior_names is not None:assert names==prior_names
        prior_names=names
    env.close()
