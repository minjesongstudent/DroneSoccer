import numpy as np
import torch
from stable_baselines3 import SAC
from src.sim import HoopEnv
from src.clearance_training import ClearanceTrainingEnv
from src.guard_alignment_reward import alignment_penalties
from tools.run_clearance_experiment import ROOT


def test_reward_only_and_offline_parity_with_learned_actions():
    torch.set_num_threads(1)
    model=SAC.load(ROOT/'trainedModels/seenflag-s1-level1.zip',device='cpu')
    for seed in [11000000,11000007,11000017]:
        base=ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance')
        changed=ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance',guard_alignment=True)
        obs,_=base.reset(seed=seed);other,_=changed.reset(seed=seed)
        poses=[];headings=[];deltas=[]
        for _ in range(200):
            np.testing.assert_array_equal(obs,other)
            poses.append(base.unwrapped.pos.copy());headings.append(base.unwrapped.heading)
            action,_=model.predict(obs,deterministic=True)
            obs,reward,done,cut,info=base.step(action)
            other,new_reward,new_done,new_cut,new_info=changed.step(action)
            np.testing.assert_array_equal(obs,other)
            np.testing.assert_array_equal(base.unwrapped.pos,changed.unwrapped.pos)
            assert (done,cut,info['outcome'])==(new_done,new_cut,new_info['outcome'])
            deltas.append(new_reward-reward)
            if done or cut:break
        expected=alignment_penalties(base.unwrapped,np.array(poses),np.array(headings))
        np.testing.assert_allclose(deltas,expected,atol=1e-12)
        base.close();changed.close()


def test_penalty_bounded_and_zero_outside_front_band():
    env=HoopEnv(level=1);env.reset(seed=11000000)
    poses=env.hoop_centre+np.array([-.5,.1,.5])[:,None]*env.hoop_normal
    np.testing.assert_array_equal(alignment_penalties(env,poses,np.zeros(3)),np.zeros(3))
    env.close()
