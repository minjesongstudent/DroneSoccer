"""Experimental 121D actor: original camera history plus twelve issued commands."""
import gymnasium as gym
import numpy as np
from src.action_history_extension import SplitInputLinear, FourActionSACPolicy, extend_model


class TwelveInputLinear(SplitInputLinear):
    added_width=36


class TwelveActionSACPolicy(FourActionSACPolicy):
    input_layer=TwelveInputLinear


class TwelveHistory:
    def __init__(self):
        self.older=np.zeros(36,np.float32)
        self.previous=None

    def update(self,obs):
        obs=np.asarray(obs,np.float32)
        assert obs.shape==(85,) and np.isfinite(obs).all()
        if self.previous is not None:
            self.older=np.concatenate([self.older[4:],self.previous[72:76]])
        result=np.concatenate([obs[:72],self.older,obs[72:]])
        self.previous=obs.copy()
        return result


class TwelveActionObservation(gym.ObservationWrapper):
    def __init__(self,env):
        super().__init__(env)
        assert env.observation_space.shape==(85,)
        self.observation_space=gym.spaces.Box(-np.inf,np.inf,(121,),np.float32)

    def reset(self,**kwargs):
        self.history=TwelveHistory()
        return super().reset(**kwargs)

    def observation(self,obs):
        return self.history.update(obs)


def extend_twelve_replay(observations,next_observations,dones):
    old=np.asarray(observations);nxt=np.asarray(next_observations)
    assert old.ndim==2 and old.shape[1]==85 and nxt.shape==old.shape
    done=np.asarray(dones).reshape(-1)
    assert len(done)==len(old)
    extra=np.zeros((len(old),36),np.float32)
    next_extra=np.zeros_like(extra)
    previous=np.zeros(36,np.float32)
    for i in range(len(old)):
        extra[i]=previous
        next_extra[i]=np.concatenate([previous[4:],old[i,72:76]])
        previous=np.zeros(36,np.float32) if done[i] else next_extra[i]
    return (np.c_[old[:,:72],extra,old[:,72:]],np.c_[nxt[:,:72],next_extra,nxt[:,72:]])


def extend_twelve_model(original):
    return extend_model(original,policy_class=TwelveActionSACPolicy,added_width=36)


def restrict_actor_to_new_columns(actor,learning_rate=1e-5):
    """Freeze the original actor and optimize only the added history weights."""
    import torch
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    weight=actor.latent_pi[0].weight
    assert weight.shape[1]==121
    weight.requires_grad_(True)
    mask=torch.zeros_like(weight);mask[:,72:108]=1
    weight.register_hook(lambda gradient: gradient*mask)
    # Fresh Adam, no weight decay or existing moments on frozen columns.
    # Keep standard parameter groups for SB3 save/reload; frozen parameters have no gradients.
    actor.optimizer=torch.optim.Adam(actor.parameters(),lr=learning_rate)


def assert_original_actor_unchanged(actor,reference):
    import torch
    for name,value in actor.state_dict().items():
        original=reference[name]
        if name=='latent_pi.0.weight':
            assert torch.equal(value[:,:72],original[:,:72])
            assert torch.equal(value[:,108:],original[:,108:])
        else:
            assert torch.equal(value,original),name
