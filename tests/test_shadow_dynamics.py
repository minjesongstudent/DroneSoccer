import numpy as np
import pytest
import torch
from src.shadow_dynamics import rollout


def inputs(device='cpu',steps=12):
    f=lambda x:torch.tensor(x,dtype=torch.float64,device=device)
    return dict(actions=f(np.zeros((1,steps,4))),dt=f(np.full((1,steps),.1)),
                lengths=torch.tensor([steps],device=device),initial_position=f([[1,1,1]]),
                initial_heading=f([0]),half_box=f([[.0525,.085,.0175]]),
                cage=f([[2.05,2.05,2.05,.5]]),speed=f([.4]),yaw_degrees=f([60]),tau=f([.55]),
                delay=torch.tensor([2],device=device))


@pytest.mark.parametrize('delay',[1,2,3,4])
def test_delay_gradient_and_zero_initial_queue(delay):
    q=inputs();q['delay'][:]=delay;q['actions'].requires_grad_()
    z=rollout(**q)['positions']
    early=torch.autograd.grad(z[0,delay-1,0],q['actions'],retain_graph=True)[0]
    later=torch.autograd.grad(z[0,delay,0],q['actions'])[0]
    assert torch.count_nonzero(early)==0
    assert later[0,0,1]>0 and torch.count_nonzero(later[0,1:])==0


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gradcheck_smooth_region(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    q=inputs(device,8)
    q['actions'][:]=torch.tensor([.1,.2,.05,-.1],device=device)
    actions=q.pop('actions').requires_grad_()
    def f(a):
        z=rollout(a,**q)
        return torch.cat((z['positions'].flatten(),z['headings'].flatten()))
    assert torch.autograd.gradcheck(f,(actions,),eps=1e-5,atol=1e-7,rtol=1e-5)


def test_padding_and_future_do_not_change_past():
    q=inputs();q['lengths'][:]=6;q['actions'][:,:,:]=.2
    a=rollout(**q);q['actions'][:,6:,:]=-.9;b=rollout(**q)
    for k in a:assert torch.equal(a[k],b[k])
    assert torch.equal(a['positions'][:,5],a['positions'][:,-1])


def test_wall_stops_velocity_and_records_contact():
    q=inputs(steps=100);q['actions'][:,:,1]=1
    z=rollout(**q)
    assert z['wall_contacts'].any()
    assert z['positions'][0,-1,0]==2.05-.0525
    assert z['velocities'][0,-1,0]==0


@pytest.mark.parametrize('bad',['delay','dt','shape','nonfinite','command'])
def test_rejects_bad_contract(bad):
    q=inputs()
    if bad=='delay':q['delay'][:]=5
    elif bad=='dt':q['dt'][:]=0
    elif bad=='shape':q['initial_position']=q['initial_position'][:,:2]
    elif bad=='nonfinite':q['actions'][0,0,0]=float('nan')
    else:q['actions'][0,0,0]=1.1
    with pytest.raises(ValueError):rollout(**q)
