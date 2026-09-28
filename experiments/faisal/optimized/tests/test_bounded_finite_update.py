from copy import deepcopy
import torch
import pytest
from test_finite_policy_update import sample
from src.bounded_finite_update import bounded_update

def test_retries_preserve_actor_optimizer_and_rate_on_failure(sample):
    actor,data=sample
    opt=torch.optim.Adam(actor.parameters(),lr=1e-4)
    state=deepcopy(actor.state_dict())
    history=bounded_update(actor,opt,*data,retries=2,epochs=1,mean_bound=0,max_bound=0)
    assert len(history)==3 and not any(h['accepted'] for h in history)
    assert [h['factor'] for h in history]==[1,.5,.25]
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in state.items())
    assert not opt.state and opt.param_groups[0]['lr']==1e-4

def test_success_and_exception_restore(sample):
    actor,data=sample;opt=torch.optim.Adam(actor.parameters(),lr=1e-4)
    history=bounded_update(actor,opt,*data,retries=2,epochs=1,mean_bound=1,max_bound=1)
    assert history[0]['accepted'] and len(history)==1
    state=deepcopy(actor.state_dict());moments=deepcopy(opt.state_dict())
    with pytest.raises(ValueError,match='current policy'):
        bounded_update(actor,opt,*data,retries=2)
    assert all(torch.equal(v,actor.state_dict()[k]) for k,v in state.items())
    for k,values in moments['state'].items():
        for name,value in values.items():assert torch.equal(value,opt.state_dict()['state'][k][name])

def test_retry_cap(sample):
    actor,data=sample;opt=torch.optim.Adam(actor.parameters())
    with pytest.raises(ValueError):bounded_update(actor,opt,*data,retries=5)
