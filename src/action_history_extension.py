"""Explicit 89D experimental profile: same camera lags plus four issued actions."""
import gymnasium as gym
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.sac.policies import SACPolicy

from src.constrained_sac import ConstrainedSAC


class SplitInputLinear(torch.nn.Linear):
    """Keep original GEMM dimensions so zero added weights preserve numerics."""
    added_width=4
    def forward(self,x):
        end=72+self.added_width
        original=torch.cat([x[...,:72],x[...,end:]],dim=-1)
        weight=torch.cat([self.weight[:,:72],self.weight[:,end:]],dim=-1)
        return F.linear(original,weight,self.bias)+F.linear(x[...,72:end],self.weight[:,72:end])


class FourActionSACPolicy(SACPolicy):
    input_layer=SplitInputLinear
    def _build(self,lr_schedule):
        super()._build(lr_schedule)
        layers=[self.actor.latent_pi,self.critic.qf0,self.critic.qf1,
                self.critic_target.qf0,self.critic_target.qf1]
        for net in layers:
            previous=net[0]
            layer=self.input_layer(previous.in_features,previous.out_features)
            layer.load_state_dict(previous.state_dict());net[0]=layer
        # Replacing modules requires rebuilding optimizers against the new parameters.
        self.actor.optimizer=self.optimizer_class(self.actor.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)
        self.critic.optimizer=self.optimizer_class(self.critic.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)


class FourActionObservation(gym.ObservationWrapper):
    """Append only an older issued command derived from preceding public history."""
    def __init__(self,env):
        super().__init__(env)
        assert env.observation_space.shape==(85,)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(89,),np.float32)
        self.previous=None

    def reset(self,**kwargs):
        self.previous=None
        return super().reset(**kwargs)

    def observation(self,obs):
        extra=np.zeros(4,np.float32) if self.previous is None else self.previous[72:76]
        result=np.concatenate([obs[:72],extra,obs[72:]])
        self.previous=obs.copy()
        return result


def extend_replay_observations(observations, next_observations, dones):
    old=np.asarray(observations)
    nxt=np.asarray(next_observations)
    assert old.ndim==2 and old.shape[1]==85 and nxt.shape==old.shape
    fourth=np.zeros((len(old),4),np.float32)
    fourth[1:]=old[:-1,72:76]
    fourth[1:][np.asarray(dones).reshape(-1)[:-1]>.5]=0
    return (np.concatenate([old[:,:72],fourth,old[:,72:]],axis=1),
            np.concatenate([nxt[:,:72],old[:,72:76],nxt[:,72:]],axis=1))


def extend_model(original, *, policy_class=FourActionSACPolicy, added_width=4):
    # This environment supplies spaces only; it is never stepped.
    env=gym.Env()
    env.observation_space=gym.spaces.Box(-np.inf,np.inf,(85+added_width,),np.float32)
    env.action_space=original.action_space
    model=ConstrainedSAC(policy_class,env,device=original.device,
        policy_kwargs=original.policy_kwargs,buffer_size=1,seed=11530000,
        gamma=original.gamma,tau=original.tau,learning_starts=0,train_freq=1,gradient_steps=1)
    old=original.policy.state_dict(); target=model.policy.state_dict()
    changed=[]
    for key,value in old.items():
        if target[key].shape==value.shape:
            target[key]=value.clone()
        else:
            assert value.ndim==2 and target[key].shape==(value.shape[0],value.shape[1]+added_width)
            assert value.shape[1] in (85,89)
            expanded=torch.zeros_like(target[key])
            expanded[:,:72]=value[:,:72]
            expanded[:,72+added_width:]=value[:,72:]
            target[key]=expanded;changed.append(key)
    assert len(changed)==5 # actor first layer and both twin critic/target first layers
    model.policy.load_state_dict(target)
    with torch.no_grad():model.log_ent_coef.copy_(original.log_ent_coef)
    return model
