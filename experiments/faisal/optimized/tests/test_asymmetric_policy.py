"""Actor isolation, context alignment and bounded CPU/CUDA lifecycle tests."""
from copy import deepcopy
import hashlib
from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure

from src import unified_policy
from src.anchored_policy import AnchoredSAC, TaskBalancedReplayBuffer
from src.asymmetric_policy import (AsymmetricSAC, AsymmetricReplayBuffer,
                                    critic_inputs, start_from_checkpoint)
from src.critic_context import CONTEXT_DIM, CriticContextWrapper
from src.return_mixture import ReturnMixtureEnv


class ContextEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(-1000.,1000.,(100,),np.float32)
        self.action_space = gym.spaces.Box(-1.,1.,(4,),np.float32)
        self.episode = 0

    def observation(self):
        obs = np.zeros(100,np.float32)
        obs[0], obs[-1] = self.episode*10+self.steps, self.episode%2
        return obs

    def context(self):
        result = np.zeros(CONTEXT_DIM,np.float32)
        result[0] = self.observation()[0]+100
        return result

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0
        self.episode += 1
        return self.observation(), {'critic_context':self.context()}

    def step(self, action):
        before = self.context()
        self.steps += 1
        last = self.steps == 3
        terminated, truncated = last and self.episode%2==1, last and self.episode%2==0
        return self.observation(),float(1.-np.square(action).sum()),terminated,truncated,{
            'critic_context':before,'next_critic_context':self.context()}


def snapshot(module):
    return {name:value.detach().cpu().clone() for name,value in module.state_dict().items()}


def assert_equal(first,second):
    if isinstance(first,dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_equal(first[key],second[key])
    elif isinstance(first,(list,tuple)):
        assert len(first)==len(second)
        for a,b in zip(first,second):
            assert_equal(a,b)
    elif torch.is_tensor(first):
        assert torch.equal(first.detach().cpu(),second.detach().cpu())
    else:
        assert first==second


def small_model(device='cpu'):
    torch.set_num_threads(1)
    model=AsymmetricSAC('MlpPolicy',ContextEnv(),device=device,seed=8,
        buffer_size=64,batch_size=8,learning_starts=0,ent_coef=.013,
        policy_kwargs={'net_arch':[16,16]})
    model.set_logger(configure(folder=None,format_strings=[]))
    return model


def add(buffer,index,*,timeout=False,context_override=None):
    obs=np.zeros((1,100),np.float32)
    obs[0,0],obs[0,99]=index,index%2
    next_obs=obs.copy()
    next_obs[0,0]+=.25
    ctx=np.full(CONTEXT_DIM,index+100,np.float32)
    next_ctx=np.full(CONTEXT_DIM,index+200,np.float32)
    info={'critic_context':ctx,'next_critic_context':next_ctx,'TimeLimit.truncated':timeout}
    if context_override is not None:
        info=context_override
    buffer.add(obs,next_obs,np.full((1,4),index,np.float32),np.array([index],np.float32),
               np.array([index%2],np.float32),[info])


def test_context_replay_balances_wraparound_and_preserves_terminal_context():
    env=ContextEnv()
    buffer=AsymmetricReplayBuffer(4,env.observation_space,env.action_space,device='cpu')
    for index in range(6):
        add(buffer,index,timeout=index==3)
    data=buffer.sample(200)
    ids=data.observations[:,0].numpy().astype(int)
    assert set(ids)=={2,3,4,5}
    assert data.observations[:,99].sum().item()==100
    np.testing.assert_array_equal(data.critic_context[:,0].numpy(),ids+100)
    np.testing.assert_array_equal(data.next_critic_context[:,0].numpy(),ids+200)
    np.testing.assert_array_equal(data.next_observations[:,0].numpy(),ids+.25)
    np.testing.assert_array_equal(data.dones[:,0].numpy(),np.where(ids==3,0,ids%2))
    buffer.reset()
    add(buffer,8)
    assert buffer.sample(10).critic_context[:,0].tolist()==[108.]*10


@pytest.mark.parametrize('info',[{}, {'critic_context':np.zeros(CONTEXT_DIM)},
    {'critic_context':np.zeros(CONTEXT_DIM-1),'next_critic_context':np.zeros(CONTEXT_DIM)},
    {'critic_context':np.full(CONTEXT_DIM,np.nan),'next_critic_context':np.zeros(CONTEXT_DIM)}])
def test_missing_or_invalid_context_fails_before_replay_mutation(info):
    env=ContextEnv()
    buffer=AsymmetricReplayBuffer(4,env.observation_space,env.action_space,device='cpu')
    add(buffer,1)
    before=buffer.observations.copy()
    with pytest.raises(ValueError):
        add(buffer,2,context_override=info)
    assert buffer.pos==buffer.size()==1
    np.testing.assert_array_equal(before,buffer.observations)


def test_critic_uses_private_context_but_actor_and_anchor_never_receive_it():
    model=small_model()
    actor_shapes,reference_shapes=[],[]
    actor_hook=model.actor.register_forward_pre_hook(lambda _,args:actor_shapes.append(args[0].shape[-1]))
    ref_original=model.reference_actor.get_action_dist_params
    def reference_params(observations):
        reference_shapes.append(observations.shape[-1])
        return ref_original(observations)
    model.reference_actor.get_action_dist_params=reference_params
    obs=torch.zeros((8,100))
    obs[:,99]=1
    context=torch.zeros((8,CONTEXT_DIM))
    with torch.no_grad():
        action=model.actor(obs,deterministic=True)
        q1=model.critic(critic_inputs(obs,context),action)[0]
        q2=model.critic(critic_inputs(obs,context+1),action)[0]
        assert not torch.equal(q1,q2)
        assert torch.equal(action,model.actor(obs,deterministic=True))
    model._actor_step(obs,model.ent_coef_tensor,context)
    assert set(actor_shapes)=={100} and set(reference_shapes)=={100}
    assert model.observation_space.shape==(100,)
    assert model.critic.observation_space.shape==(100+CONTEXT_DIM,)
    assert not ({id(p) for p in model.actor.parameters()} & {id(p) for p in model.critic.parameters()})
    with pytest.raises(ValueError,match='explicit simulator context'):
        model._actor_step(obs,model.ent_coef_tensor)
    with pytest.raises(ValueError):
        model.predict(np.zeros(100+CONTEXT_DIM,np.float32))
    actor_hook.remove()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_calibration_freezes_actor_and_adam_and_roundtrip_preserves_every_state(tmp_path,device):
    if device=='cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model=small_model(device)
    public_shapes=[]
    original_parameters=model.actor.get_action_dist_params
    def actor_parameters(observations):
        public_shapes.append(observations.shape[-1])
        return original_parameters(observations)
    model.actor.get_action_dist_params=actor_parameters
    model.refill_transitions,model.calibration_updates=4,2
    model.learn(12)
    assert model.critic_calibration_updates==2 and model.actor_attempts==7
    assert model.actor_updates>0
    assert model.actor.optimizer.state_dict()['state']
    assert any(not torch.equal(snapshot(model.actor)[k],v) for k,v in snapshot(model.reference_actor).items())
    actor,reference,critic=snapshot(model.actor),snapshot(model.reference_actor),snapshot(model.critic)
    optimizer=deepcopy(model.actor.optimizer.state_dict())
    attempts=model.actor_attempts
    model._calibrate_critics(3,8)
    assert public_shapes and set(public_shapes)=={100}
    model.actor.get_action_dist_params=original_parameters
    assert_equal(actor,snapshot(model.actor))
    assert_equal(reference,snapshot(model.reference_actor))
    assert_equal(optimizer,model.actor.optimizer.state_dict())
    assert model.actor_attempts==attempts
    assert any(not torch.equal(v,snapshot(model.critic)[k]) for k,v in critic.items())
    # Autoreset terminal observation and context must refer to the same state.
    b=model.replay_buffer
    n=b.size()
    np.testing.assert_array_equal(b.critic_contexts[:n,0,0],b.observations[:n,0,0]+100)
    np.testing.assert_array_equal(b.next_critic_contexts[:n,0,0],b.next_observations[:n,0,0]+100)
    assert b.timeouts[:n].sum()==2 and b.dones[:n].sum()==4
    model.save(tmp_path/'asymmetric.zip')
    model.save_replay_buffer(tmp_path/'replay.pkl')
    restored=AsymmetricSAC.load(tmp_path/'asymmetric.zip',env=ContextEnv(),device=device)
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    for module in ('actor','reference_actor','critic','critic_target'):
        assert_equal(snapshot(getattr(model,module)),snapshot(getattr(restored,module)))
    assert_equal(model.actor.optimizer.state_dict(),restored.actor.optimizer.state_dict())
    assert_equal(model.critic.optimizer.state_dict(),restored.critic.optimizer.state_dict())
    for name in ('observations','next_observations','actions','rewards','dones','timeouts',
                 'critic_contexts','next_critic_contexts','_single_pass_slots'):
        np.testing.assert_array_equal(getattr(model.replay_buffer,name),getattr(restored.replay_buffer,name))
    for name in ('num_timesteps','actor_attempts','actor_updates','anchor_rejected','critic_calibration_updates',
                 'context_version','context_dim','asymmetric_version'):
        assert getattr(model,name)==getattr(restored,name)
    obs=ContextEnv().reset(seed=4)[0]
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],restored.predict(obs,deterministic=True)[0])
    assert all(not p.requires_grad for p in restored.reference_actor.parameters())
    restored.learn(4,reset_num_timesteps=False)
    assert restored.num_timesteps==16


def test_plain_replay_and_checkpoint_cannot_silently_resume_asymmetric(tmp_path):
    env=ContextEnv()
    old=AnchoredSAC('MlpPolicy',env,device='cpu',buffer_size=8,
                    policy_kwargs={'net_arch':[16,16]},ent_coef=.013)
    old.save(tmp_path/'old.zip')
    old.save_replay_buffer(tmp_path/'old.pkl')
    with pytest.raises(ValueError,match='context contract'):
        AsymmetricSAC.load(tmp_path/'old.zip',device='cpu')
    model=small_model()
    original=model.replay_buffer
    with pytest.raises(ValueError,match='context contract'):
        model.load_replay_buffer(tmp_path/'old.pkl')
    assert model.replay_buffer is original


def test_changed_context_version_cannot_resume_model_or_replay(tmp_path):
    model=small_model()
    model.context_version='incompatible'
    model.save(tmp_path/'bad-model.zip')
    with pytest.raises(ValueError,match='context contract'):
        AsymmetricSAC.load(tmp_path/'bad-model.zip',device='cpu')
    model.replay_buffer.context_version='incompatible'
    model.save_replay_buffer(tmp_path/'bad-replay.pkl')
    fresh=small_model()
    original=fresh.replay_buffer
    with pytest.raises(ValueError,match='context contract'):
        fresh.load_replay_buffer(tmp_path/'bad-replay.pkl')
    assert fresh.replay_buffer is original


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA physical-environment smoke')
def test_real_cuda_actor_transfer_context_collection_and_disk_resume(tmp_path,monkeypatch):
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/return-route-40k-20260927/best_model.zip'
    bank=root/'return-libraries/train-v1'
    if not source.exists() or not (bank/'manifest.json').exists():
        pytest.skip('Local frozen source and bank are required')
    before=hashlib.sha256(source.read_bytes()).hexdigest()
    old=AnchoredSAC.load(source,device='cpu')
    monkeypatch.setattr(unified_policy,'REFILL_TRANSITIONS',8)
    monkeypatch.setattr(unified_policy,'CALIBRATION_UPDATES',4)
    env=CriticContextWrapper(ReturnMixtureEnv(bank))
    model,metadata=start_from_checkpoint(source,env,device='cuda',seed=7,buffer_size=128,
                                         tensorboard_log=tmp_path/'logs')
    assert_equal(snapshot(old.actor),snapshot(model.actor))
    assert_equal(snapshot(old.reference_actor),snapshot(model.reference_actor))
    assert not model.actor.optimizer.state_dict()['state'] and not model.critic.optimizer.state_dict()['state']
    assert model.replay_buffer.size()==model.num_timesteps==model.actor_attempts==0
    obs,info=env.reset(seed=7)
    assert info['return_practice'] and info['passes']==0
    np.testing.assert_allclose(old.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0],atol=2e-6,rtol=0)
    assert metadata['actor_receives_privileged_context'] is False
    actor=snapshot(model.actor)
    model.learn(11)  # Eight refill transitions and four critic updates, no actor step.
    assert model.critic_calibration_updates==4 and model.actor_attempts==0
    assert_equal(actor,snapshot(model.actor))
    assert next(model.critic.parameters()).device.type=='cuda'
    assert np.any(model.replay_buffer.critic_contexts[:11]!=0)
    model.learn(5,reset_num_timesteps=False)
    assert model.actor_attempts==5
    model.save(tmp_path/'real.zip')
    model.save_replay_buffer(tmp_path/'real.pkl')
    resumed=AsymmetricSAC.load(tmp_path/'real.zip',env=CriticContextWrapper(ReturnMixtureEnv(bank)),device='cuda')
    resumed.load_replay_buffer(tmp_path/'real.pkl')
    assert_equal(snapshot(model.actor),snapshot(resumed.actor))
    assert_equal(snapshot(model.reference_actor),snapshot(resumed.reference_actor))
    np.testing.assert_array_equal(model.replay_buffer.next_critic_contexts,resumed.replay_buffer.next_critic_contexts)
    np.testing.assert_array_equal(model.predict(obs,deterministic=True)[0],resumed.predict(obs,deterministic=True)[0])
    assert hashlib.sha256(source.read_bytes()).hexdigest()==before
    env.close()
