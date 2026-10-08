"""Causal recurrence, true sequence gradients and model identity, without flights."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.recurrent_memory_actor import RecurrentMemoryPolicy
from src.trainable_memory_actor import TrainableMemoryPolicy, checkpoint_manifest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'trainedModels/seenflag-s1-level1.zip'
ESTIMATOR = Path('D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt')


@pytest.fixture
def policy():
    return TrainableMemoryPolicy(SOURCE,ESTIMATOR,device='cpu')


def public_rows():
    with np.load(ROOT.parent/'lab-data/policy-validation-20261006/seenflag-s1-level1/aligned/9410000.npz',allow_pickle=False) as z:
        return z['actor_observations'].copy()


def batch(policy, lengths=(17,8)):
    rows=public_rows()
    x=np.zeros((len(lengths),max(lengths),85),np.float32)
    for i,n in enumerate(lengths):x[i,:n]=rows[i:i+n]
    return torch.tensor(x,device=policy.device),torch.tensor(lengths,dtype=torch.int64)


def test_initial_stream_is_identical_to_frozen_memory_and_actions(policy):
    frozen=RecurrentMemoryPolicy(SOURCE,ESTIMATOR,device='cpu')
    for row in public_rows()[:25]:
        a,b=policy.observe(row),frozen.observe(row)
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(policy.predict(a)[0],frozen.predict(b)[0])
    assert policy.frame_count==25
    for _ in range(3):policy.predict(a)
    assert policy.frame_count==25
    policy.reset();frozen.reset()
    np.testing.assert_array_equal(policy.observe(row),frozen.observe(row))


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_full600_sequence_matches_stream_and_padding_is_inert(device,monkeypatch):
    if device.startswith('cuda') and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    monkeypatch.setattr(torch.backends.cudnn,'allow_tf32',False)
    monkeypatch.setattr(torch.backends.cuda.matmul,'allow_tf32',False)
    p=TrainableMemoryPolicy(SOURCE,ESTIMATOR,device=device)
    x,lengths=batch(p,(600,9))
    with torch.no_grad():
        # Include a real nonzero correction; zero head alone hides memory errors.
        p.actor.residual_weight.copy_(torch.linspace(-.002,.002,256,device=p.device).reshape(4,64))
        seq=p.actor.sequence_features(x,lengths).cpu().numpy()
        seq_actions=p.predict(seq)[0]
    streams=[]
    for i,n in enumerate(lengths):
        p.reset()
        streams.extend(p.observe(row) for row in x[i,:n].cpu().numpy())
    streams=np.asarray(streams)
    np.testing.assert_allclose(seq,streams,rtol=0,atol=1e-6)
    np.testing.assert_allclose(seq_actions,p.predict(streams)[0],rtol=0,atol=1e-6)
    x[1,9:]=float('nan')
    with torch.no_grad():again=p.actor.sequence_features(x,lengths).cpu().numpy()
    np.testing.assert_array_equal(again,seq)


def test_sequence_has_no_future_or_cross_episode_leakage(policy):
    x,lengths=batch(policy)
    first=policy.actor.sequence_features(x,lengths).detach().clone()
    x[0,10:]*=-1; x[1]*=2
    second=policy.actor.sequence_features(x,lengths).detach()
    torch.testing.assert_close(first[:10],second[:10],rtol=0,atol=0)
    one=policy.actor.sequence_features(x[1:,:8],lengths[1:]).detach()
    torch.testing.assert_close(second[17:],one,rtol=0,atol=1e-6)
    assert policy.frame_count==0


def test_control_gradient_reaches_memory_and_earlier_inputs_only_after_nonzero_head(policy):
    x,lengths=batch(policy);x.requires_grad_()
    policy.actor.train()
    mean,_,_=policy.actor.sequence_distribution(x,lengths)
    mean[-1].sum().backward()
    assert all(p.grad is None or not p.grad.any() for p in policy.actor.memory.parameters())
    assert policy.actor.residual_weight.grad.abs().sum()>0
    policy.actor.zero_grad(set_to_none=True);x.grad=None
    with torch.no_grad():policy.actor.residual_weight.fill_(.001)
    mean,_,_=policy.actor.sequence_distribution(x,lengths)
    mean[-1].sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.actor.memory.parameters())
    assert sum(float(p.grad.abs().sum()) for p in policy.actor.memory.parameters())>0
    assert x.grad[1,0].abs().sum()>0  # Earlier actual input influences the final decision.
    assert not x.grad[0].any() and not x.grad[1,8:].any()
    assert all(p.grad is None for p in policy.actor.base_actor.parameters())
    before={k:v.clone() for k,v in policy.actor.memory.state_dict().items()}
    torch.optim.Adam([p for p in policy.actor.parameters() if p.requires_grad],lr=1e-5).step()
    assert any(not torch.equal(before[k],v) for k,v in policy.actor.memory.state_dict().items())
    policy.actor.assert_source_unchanged()
    updated=policy.actor.sequence_features(x.detach(),lengths).detach()
    assert torch.isfinite(updated).all() and policy.frame_count==0


def test_checkpoint_preserves_trained_memory_not_stream_state(policy,tmp_path):
    with torch.no_grad():
        policy.actor.memory.bias_ih_l0.add_(.002)
        policy.actor.residual_weight.fill_(.003)
    rows=public_rows()[:15]
    expected=[policy.predict(policy.observe(row))[0] for row in rows]
    path=tmp_path/'candidate.npz';digest=policy.save(path)
    loaded=TrainableMemoryPolicy.load(path,SOURCE,ESTIMATOR,device='cpu',sha256=digest)
    assert loaded.frame_count==0
    actual=[loaded.predict(loaded.observe(row))[0] for row in rows]
    np.testing.assert_array_equal(actual,expected)
    with pytest.raises(FileExistsError):policy.save(path)
    with pytest.raises(ValueError,match='checksum'):
        TrainableMemoryPolicy.load(path,SOURCE,ESTIMATOR,device='cpu',sha256='0'*64)
    with np.load(path,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    del data['memory.bias_hh_l0']
    wrong=tmp_path/'missing.npz';np.savez(wrong,**data)
    with pytest.raises(ValueError,match='Exact head'):
        TrainableMemoryPolicy.load(wrong,SOURCE,ESTIMATOR,device='cpu',sha256=hashlib.sha256(wrong.read_bytes()).hexdigest())
    assert checkpoint_manifest()['memory_trainable'] is True
    assert checkpoint_manifest()['estimator_role'].startswith('initialization only')


@pytest.mark.parametrize('problem',['float64','width','zero_length','long_length','float_length','nonfinite'])
def test_reject_bad_sequences_without_advancing_stream(policy,problem):
    x,n=batch(policy)
    if problem=='float64':x=x.double()
    elif problem=='width':x=x[:,:,:84]
    elif problem=='zero_length':n[0]=0
    elif problem=='long_length':n[0]=18
    elif problem=='float_length':n=n.float()
    elif problem=='nonfinite':x[0,0,0]=float('nan')
    with pytest.raises(ValueError):policy.actor.sequence_features(x,n)
    assert policy.frame_count==0


def test_normalization_cannot_be_silently_fitted(policy):
    with torch.no_grad():policy.actor.x_mean[0]+=1
    with pytest.raises(ValueError,match='normalizer'):policy.actor.assert_source_unchanged()
