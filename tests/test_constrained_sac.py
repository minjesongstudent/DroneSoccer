import copy
import numpy as np
import pytest
import torch
from stable_baselines3.common.buffers import ReplayBuffer

from src.clearance_training import ClearanceTrainingEnv
from src.constrained_sac import (ConstrainedSAC, episode_partition, gaussian_kl,
                                 guarded_optimizer_step, boundary_penalty)
from src.sim import HoopEnv


def test_kl_matches_torch_distribution():
    torch.manual_seed(4)
    a,b,c,d = [torch.randn(20,4)*.2 for _ in range(4)]
    expected = torch.distributions.kl_divergence(
        torch.distributions.Normal(a,b.exp()),torch.distributions.Normal(c,d.exp())).sum(-1)
    torch.testing.assert_close(gaussian_kl(a,b,c,d), expected)
    torch.testing.assert_close(gaussian_kl(a,b,a,b), torch.zeros(20))


def test_episode_partition_has_no_family_leakage():
    done = np.array([0,1]*12+[0,0,0])
    fit, heldout, seeds = episode_partition(done)
    assert len(heldout)>0
    assert set(seeds[fit]).isdisjoint(set(seeds[heldout]))
    np.testing.assert_array_equal(np.sort(np.r_[fit,heldout]), np.arange(len(done)))
    assert seeds[-1]==11_000_012


def test_boundary_penalty_pushes_back_without_changing_hard_guard():
    interior=torch.tensor([.001,.002],requires_grad=True)
    assert boundary_penalty(interior,.01,.1).item()==0
    near=torch.tensor([.001]*99+[.09],requires_grad=True)
    penalty=boundary_penalty(near,.01,.1)
    penalty.backward()
    assert penalty.item()>0 and near.grad[-1]>0
    assert torch.count_nonzero(near.grad[:-1])==0


def test_rejected_step_restores_optimizer_moments():
    actor=torch.nn.Linear(2,1)
    opt=torch.optim.Adam(actor.parameters(),lr=.1)
    actor(torch.ones(1,2)).sum().backward();opt.step();opt.zero_grad()
    weights=copy.deepcopy(actor.state_dict());state=copy.deepcopy(opt.state_dict())
    accepted,_=guarded_optimizer_step(actor,opt,actor(torch.ones(1,2)).sum(),
                                      lambda:(False,{'mean_kl':1.,'max_kl':2.}))
    assert not accepted
    for k,v in weights.items(): assert torch.equal(v,actor.state_dict()[k])
    for k,slot in state['state'].items():
        for key,v in slot.items(): assert torch.equal(v,opt.state_dict()['state'][k][key])


def test_backtracking_accepts_smaller_step_without_relaxing_guard():
    actor=torch.nn.Linear(1,1,bias=False)
    with torch.no_grad():actor.weight.zero_()
    optimizer=torch.optim.Adam(actor.parameters(),lr=.1)
    def check():
        delta=float(actor.weight.detach().abs().max())
        return delta<=.03,dict(delta=delta)
    accepted,metrics=guarded_optimizer_step(actor,optimizer,actor(torch.ones(1,1)).sum(),check,max_backtracks=4)
    assert accepted and metrics['backtracks']==2 and metrics['delta']<=.03
    assert optimizer.param_groups[0]['lr']==.1
    assert int(optimizer.state[actor.weight]['step'])==1


@pytest.mark.parametrize('uniform_actor', [False, True])
def test_critic_calibration_keeps_actor_and_entropy_exact(uniform_actor):
    env=ClearanceTrainingEnv(HoopEnv(), 'clearance')
    model=ConstrainedSAC('MlpPolicy',env,device='cpu',buffer_size=100,seed=5)
    obs,_=env.reset()
    model.configure(np.stack([obs]*8), uniform_actor_replay=uniform_actor)
    actor=copy.deepcopy(model.actor.state_dict());alpha=model.log_ent_coef.detach().clone()
    critic=copy.deepcopy(model.critic.state_dict())
    # Real learned actions, including resets if needed.
    for _ in range(32):
        action,_=model.predict(obs,deterministic=False)
        nxt,reward,done,cut,info=env.step(action)
        model.replay_buffer.add(obs,nxt,action,reward,done or cut,[{}])
        obs=nxt
        if done or cut: obs,_=env.reset()
    model.train(3,batch_size=8)
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in actor.items())
    assert torch.equal(alpha,model.log_ent_coef)
    assert any(not torch.equal(v,model.critic.state_dict()[k]) for k,v in critic.items())
    model.actor_enabled=True
    model.mean_limit=model.max_limit=-1 # force actual model rollback
    model.train(1,batch_size=8)
    assert model.rejected_updates==1 and model.accepted_updates==0
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in actor.items())
