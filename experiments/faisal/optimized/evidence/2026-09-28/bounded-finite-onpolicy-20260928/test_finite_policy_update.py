from copy import deepcopy
import numpy as np
import pytest
import torch
import gymnasium as gym
from stable_baselines3 import SAC
from src.finite_policy_update import soft_returns, clipped_gain, update_actor

def test_finite_returns_entropy_and_terminal():
    assert np.allclose(soft_returns([1,2],[-1,-2],.1,.5),[2.2,2.2])
    assert np.allclose(soft_returns([7],[3],.1,.99),[6.7])
    with pytest.raises(ValueError):soft_returns([1],[np.nan],.1,.99)

def test_clipping_handles_both_advantage_signs():
    ratio=torch.tensor([2.,.5,2.,.5])
    weights=torch.tensor([1.,1.,-1.,-1.])
    assert torch.allclose(clipped_gain(ratio.log(),weights),torch.tensor((1.1+.5-2-.9)/4))

@pytest.fixture
def sample():
    if not torch.cuda.is_available():pytest.skip('CUDA required for neural update test')
    torch.manual_seed(4);torch.set_num_threads(1)
    env=gym.make('Pendulum-v1')
    model=SAC('MlpPolicy',env,device='cuda',policy_kwargs={'net_arch':[16,16]},buffer_size=2)
    actor=model.actor
    with torch.no_grad():
        x=torch.randn(128,3,device='cuda')
        mu,ls,_=actor.get_action_dist_params(x)
        dist=actor.action_dist.proba_distribution(mu,ls)
        a=dist.get_actions(False);z=dist.gaussian_actions.clone();lp=dist.log_prob(a,z)
        w=(z-mu).sum(-1)
    yield actor,(x,a,z,w,lp)
    env.close()

def test_actual_update_and_transactional_rollback(sample):
    actor,data=sample;opt=torch.optim.Adam(actor.parameters(),lr=1e-4)
    before=deepcopy(actor.state_dict())
    rows=update_actor(actor,opt,*data,epochs=2,batch_size=128,mean_bound=1,max_bound=1)
    assert rows[0]['accepted']
    assert any(not torch.equal(before[k],v) for k,v in actor.state_dict().items())
    # Fresh collection densities for the changed policy; reject every proposal.
    x,a,z,w,_=data
    with torch.no_grad():
        mu,ls,_=actor.get_action_dist_params(x)
        lp=actor.action_dist.proba_distribution(mu,ls).log_prob(a,z)
    state=deepcopy(actor.state_dict());moments=deepcopy(opt.state_dict())
    rows=update_actor(actor,opt,x,a,z,w,lp,epochs=2,mean_bound=0,max_bound=0)
    assert not rows[0]['accepted'] and len(rows)==1
    assert all(torch.equal(state[k],v) for k,v in actor.state_dict().items())
    for key,values in moments['state'].items():
        for name,value in values.items():
            assert torch.equal(value,opt.state_dict()['state'][key][name])

def test_rejects_stale_behavior_and_wrong_pre_tanh(sample):
    actor,(x,a,z,w,lp)=sample;opt=torch.optim.Adam(actor.parameters())
    with pytest.raises(ValueError,match='current policy'):
        update_actor(actor,opt,x,a,z,w,lp+1)
    with pytest.raises(ValueError,match='pre-tanh'):
        update_actor(actor,opt,x,a,z+1,w,lp)
