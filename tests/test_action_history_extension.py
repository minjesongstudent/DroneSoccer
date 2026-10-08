import numpy as np
import torch
from stable_baselines3 import SAC
from src.action_history_extension import extend_replay_observations,extend_model
from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from src.action_history_extension import FourActionObservation
from src.clearance_training import ClearanceTrainingEnv
from src.sim import HoopEnv


def test_replay_matches_online_actual_fourth_action_and_resets():
    lag=SPECS['seenflag-s1-level1'].lags
    a=CameraHistory(lag);b=CameraHistory(lag,4)
    old=[];new=[];expected=[];expected_next=[];done=[]
    rng=np.random.default_rng(5)
    for episode in range(2):
        a=CameraHistory(lag);b=CameraHistory(lag,4)
        obs=rng.normal(size=8).astype(np.float32);action=np.zeros(4,np.float32)
        x=a.update(obs,action);y=b.update(obs,action)
        for step in range(10):
            old.append(x);expected.append(y)
            action=rng.uniform(-1,1,4).astype(np.float32);obs=rng.normal(size=8).astype(np.float32)
            x=a.update(obs,action);y=b.update(obs,action)
            new.append(x);expected_next.append(y);done.append(step==9)
    ex,nx=extend_replay_observations(old,new,done)
    np.testing.assert_array_equal(ex,expected);np.testing.assert_array_equal(nx,expected_next)


def test_transplant_preserves_actor_distribution_and_critics():
    source=SAC.load('trainedModels/seenflag-s1-level1.zip',device='cpu')
    model=extend_model(source)
    rng=np.random.default_rng(6)
    old=rng.normal(size=(50,85)).astype(np.float32)
    extra=rng.normal(size=(50,4)).astype(np.float32)
    new=np.concatenate([old[:,:72],extra,old[:,72:]],axis=1)
    action=torch.tensor(rng.uniform(-1,1,(50,4)),dtype=torch.float32)
    with torch.no_grad():
        for a,b in zip(source.actor.get_action_dist_params(torch.tensor(old))[:2],model.actor.get_action_dist_params(torch.tensor(new))[:2]):
            assert torch.equal(a,b)
        for critic in ('critic','critic_target'):
            for a,b in zip(getattr(source,critic)(torch.tensor(old),action),getattr(model,critic)(torch.tensor(new),action)):
                assert torch.equal(a,b)


def test_online_wrapper_reconstructs_four_commands_from_learned_steps():
    original=SAC.load('trainedModels/seenflag-s1-level1.zip',device='cpu')
    model=extend_model(original)
    base=ClearanceTrainingEnv(HoopEnv(), 'clearance')
    env=FourActionObservation(base)
    for seed in [11000001,11000002]:
        obs,_=env.reset(seed=seed)
        previous=[]
        for _ in range(8):
            assert obs.shape==(89,)
            expected=np.zeros(4,np.float32) if len(previous)<4 else previous[-4]
            np.testing.assert_array_equal(obs[72:76],expected)
            action,_=model.predict(obs,deterministic=True)
            previous.append(action.copy())
            obs,_,done,cut,_=env.step(action)
            if done or cut:break
    env.close()
