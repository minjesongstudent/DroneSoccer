"""Contract, frozen-parameter gradient and final-only gates for analytic trial."""
import numpy as np
import pytest
import torch
from stable_baselines3 import SAC
from tools.run_analytic_actor_trial import (SOURCE, GUARD, freeze_mean_head, commands,
    final_decision, check_budget, batch_tensors, sequence_commands, MAIN_SEEDS, SMOKE_SEEDS)


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_source_conversion_gradient_and_reload(device,tmp_path):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    model=SAC.load(SOURCE,device=device);optimizer=freeze_mean_head(model)
    before={k:v.detach().clone() for k,v in model.actor.state_dict().items()}
    with np.load(GUARD) as z:public=torch.as_tensor(z['features'][:64,:85].copy(),device=device)
    assert {k for k,p in model.actor.named_parameters() if p.requires_grad}=={'mu.weight','mu.bias'}
    for row in public:
        with torch.no_grad():a=commands(model.actor,row[None])[0].cpu().numpy()
        b,_=model.predict(row.cpu().numpy(),deterministic=True)
        assert np.array_equal(a,b)
    # A numeric tensor objective, never applied as a simulator command.
    loss=commands(model.actor,public).square().mean();optimizer.zero_grad();loss.backward()
    assert model.actor.mu.weight.grad.abs().sum()>0
    assert all(p.grad is None for k,p in model.actor.named_parameters() if not k.startswith('mu.'))
    optimizer.step()
    assert any(not torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items() if k.startswith('mu.'))
    assert all(torch.equal(v,model.actor.state_dict()[k]) for k,v in before.items() if not k.startswith('mu.'))
    model.save(tmp_path/'model.zip');other=SAC.load(tmp_path/'model.zip',device=device)
    with torch.no_grad():assert torch.equal(commands(model.actor,public),commands(other.actor,public))


@pytest.mark.parametrize('shape,dtype',[((3,89),torch.float32),((85,),torch.float32),((2,85),torch.float64)])
def test_no_extra_observations(shape,dtype):
    with pytest.raises(ValueError):commands(None,torch.zeros(shape,dtype=dtype))


def metrics(repeat=53,crashes=47):
    return dict(aligned=dict(episodes=100,first_pass=84,clean_repeat=84,crashes=16),
        nominal=dict(episodes=100,first_pass=80,clean_repeat=74,crashes=26),
        randomised=dict(episodes=100,first_pass=73,clean_repeat=repeat,crashes=crashes))


@pytest.mark.parametrize('smoke,complete,accepted,changed,failures',[
    (True,True,1,True,[]),(False,False,1,True,[]),(False,True,0,True,[]),
    (False,True,1,False,[]),(False,True,1,True,['retention'])])
def test_incomplete_or_intermediate_cannot_qualify(smoke,complete,accepted,changed,failures):
    d=final_decision(metrics(48,52),metrics(),smoke=smoke,complete=complete,
        accepted=accepted,changed=changed,failures=failures)
    assert not d['passed'] and not d['policy_promoted'] and not d['acceptance']


def test_final_gate_unchanged():
    kw=dict(smoke=False,complete=True,accepted=40,changed=True,failures=[])
    assert final_decision(metrics(48,52),metrics(),**kw)['passed']
    assert not final_decision(metrics(48,52),metrics(52,48),**kw)['passed']
    bad=metrics();bad['nominal']['clean_repeat']-=3
    assert not final_decision(metrics(48,52),bad,**kw)['passed']
    assert not final_decision(metrics(48,52),None,**kw)['passed']


def test_time_and_stop(tmp_path):
    check_budget(tmp_path,10,3609.9)
    with pytest.raises(RuntimeError):check_budget(tmp_path,10,3610)
    (tmp_path/'STOP').touch()
    with pytest.raises(RuntimeError):check_budget(tmp_path,10,10.1)


def test_episode_padding_is_not_an_actor_truth_input():
    records=[]
    for n in [3,5]:
        records.append(dict(public=np.ones((n,85),np.float32),actions=np.zeros((n,4),np.float32),
            dt=np.ones(n)*.1,initial=dict(initial_position=np.ones(3),initial_heading=0.,
            half_box=np.ones(3)*.1,cage=np.array([2.05,2.05,2.05,.1]),speed=1.,yaw_degrees=90.,tau=.5,delay=2)))
    public,actual,physics=batch_tensors(records,'cpu')
    assert public.shape==(2,5,85) and public.dtype==torch.float32
    assert not public[0,3:].any() and physics['lengths'].tolist()==[3,5]
    assert physics['delay'].dtype==torch.int64 and actual.dtype==torch.float64


def test_unique_separated_fit_schedules():
    assert len(MAIN_SEEDS)==160 and len(SMOKE_SEEDS)==4
    assert not set(MAIN_SEEDS)&set(SMOKE_SEEDS)
    assert all(11240000<=s<11245000 for s in MAIN_SEEDS+SMOKE_SEEDS)


def test_whole_sequence_uses_exact_frame_arithmetic_and_ignores_padding():
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    model=SAC.load(SOURCE,device='cuda');freeze_mean_head(model)
    with np.load(GUARD) as z:rows=torch.as_tensor(z['features'][:120,:85].copy(),device='cuda')
    public=rows.reshape(2,60,85);public[0,45:]=float('nan')
    action=sequence_commands(model.actor,public,torch.tensor([45,60]),lambda:None)
    for j,n in enumerate([45,60]):
        for i in range(n):
            expected,_=model.predict(public[j,i].cpu().numpy(),deterministic=True)
            assert np.array_equal(action[j,i].detach().cpu().numpy(),expected)
    assert not action[0,45:].any()
    action.square().mean().backward()
    assert torch.isfinite(model.actor.mu.weight.grad).all()
