import numpy as np
import torch
from stable_baselines3 import SAC
from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from src.twelve_action_history import TwelveHistory, extend_twelve_replay, extend_twelve_model, TwelveActionObservation
from src.clearance_training import ClearanceTrainingEnv
from src.sim import HoopEnv


def test_history_and_replay_match_issued_commands_across_resets():
    rng=np.random.default_rng(17)
    old=[];nxt=[];done=[];expected=[];expected_next=[]
    for length in [20,5,14]:
        camera=CameraHistory(SPECS['seenflag-s1-level1'].lags)
        history=TwelveHistory();issued=[]
        obs=camera.update(rng.normal(size=8),np.zeros(4))
        extended=history.update(obs)
        for step in range(length):
            commands=np.zeros((12,4),np.float32)
            if issued:
                recent=issued[-12:];commands[-len(recent):]=recent
            np.testing.assert_array_equal(extended[72:120],commands.ravel())
            old.append(obs);expected.append(extended)
            action=rng.uniform(-1,1,4).astype(np.float32);issued.append(action)
            obs=camera.update(rng.normal(size=8),action);extended=history.update(obs)
            nxt.append(obs);expected_next.append(extended);done.append(step==length-1)
    ex,nx=extend_twelve_replay(old,nxt,done)
    np.testing.assert_array_equal(ex,expected);np.testing.assert_array_equal(nx,expected_next)


def test_transplant_reload_exact_actor_and_critics(tmp_path):
    for device in ['cpu']+(['cuda'] if torch.cuda.is_available() else []):
        source=SAC.load('trainedModels/seenflag-s1-level1.zip',device=device)
        model=extend_twelve_model(source)
        old=torch.randn(30,85,device=device)
        new=torch.cat([old[:,:72],torch.randn(30,36,device=device),old[:,72:]],dim=1)
        action=torch.tanh(torch.randn(30,4,device=device))
        model.save(tmp_path/device)
        reloaded=SAC.load(tmp_path/device,device=device)
        for candidate in [model,reloaded]:
            with torch.no_grad():
                for a,b in zip(source.actor.get_action_dist_params(old)[:2],candidate.actor.get_action_dist_params(new)[:2]):
                    assert torch.equal(a,b)
                for name in ['critic','critic_target']:
                    for a,b in zip(getattr(source,name)(old,action),getattr(candidate,name)(new,action)):
                        assert torch.equal(a,b)


def test_online_learned_actions_match_history_after_reset():
    source=SAC.load('trainedModels/seenflag-s1-level1.zip',device='cpu')
    model=extend_twelve_model(source)
    env=TwelveActionObservation(ClearanceTrainingEnv(HoopEnv(),'clearance'))
    for seed in [11000001,11000002]:
        obs,_=env.reset(seed=seed);issued=[]
        for _ in range(20):
            commands=np.zeros((12,4),np.float32)
            if issued:
                recent=issued[-12:];commands[-len(recent):]=recent
            np.testing.assert_array_equal(obs[72:120],commands.ravel())
            action,_=model.predict(obs,deterministic=True);issued.append(action.copy())
            obs,_,done,cut,_=env.step(action)
            if done or cut:break
    env.close()


def test_restricted_optimizer_updates_only_added_columns(tmp_path):
    from src.twelve_action_history import restrict_actor_to_new_columns,assert_original_actor_unchanged
    model=extend_twelve_model(SAC.load('trainedModels/seenflag-s1-level1.zip',device='cpu'))
    actor=model.actor
    before={k:v.clone() for k,v in actor.state_dict().items()}
    restrict_actor_to_new_columns(actor)
    for _ in range(3):
        obs=torch.randn(20,121)
        mean,_,_=actor.get_action_dist_params(obs)
        loss=mean.square().mean()
        actor.optimizer.zero_grad();loss.backward();actor.optimizer.step()
        assert_original_actor_unchanged(actor,before)
    assert actor.latent_pi[0].weight[:,72:108].abs().sum()>0
    model.save(tmp_path/'restricted')
    loaded=SAC.load(tmp_path/'restricted',device='cpu')
    for name,value in actor.state_dict().items():
        assert torch.equal(value,loaded.actor.state_dict()[name])
