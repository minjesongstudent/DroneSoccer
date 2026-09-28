from copy import deepcopy
from pathlib import Path
import pytest
import torch
from src.balanced_anchor_policy import balanced_gradients, BalancedAnchorSAC, VERSION
from src.geometric_policy import GeometricSAC


@pytest.mark.parametrize('r,a,expected', [(3.,30.,.1),(3.,2.,1.),(0.,2.,0.),(3.,0.,1.)])
def test_anchor_gradient_bound_without_mutation(r,a,expected):
    rl=[torch.tensor([r,0.])];anchor=[torch.tensor([0.,a])]
    result,scale=balanced_gradients(rl,anchor)
    assert scale.item()==pytest.approx(expected)
    assert torch.linalg.vector_norm(result[0]-rl[0])<=torch.linalg.vector_norm(rl[0])+1e-6
    assert rl[0].tolist()==[r,0.] and anchor[0].tolist()==[0.,a]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA proposal/save/load')
def test_cuda_reject_restores_actor_adam_and_archive_contract(tmp_path):
    source=Path('D:/DroneSoccerData/runs/geometric-return-20260928/resume-1/model.zip')
    if not source.exists():pytest.skip('Local diagnostic checkpoint')
    torch.set_num_threads(1)
    model=GeometricSAC.load(source,device='cuda')
    model.__class__=BalancedAnchorSAC;model.balanced_anchor_version=VERSION
    model.load_replay_buffer(source.parent/'replay_buffer.pkl')
    x=model.replay_buffer.sample(256).observations
    before=deepcopy(model.actor.state_dict());optim=deepcopy(model.actor.optimizer.state_dict())
    reference=deepcopy(model.reference_actor.state_dict())
    model.anchor_max_kl=-1 # Force rejection, including any finite zero-KL batch.
    result=model._actor_step(x,model.ent_coef_tensor)
    assert not result['accepted'] and 0<=result['anchor_gradient_scale']<=1
    for k,v in before.items():assert torch.equal(v,model.actor.state_dict()[k])
    for k,v in reference.items():assert torch.equal(v,model.reference_actor.state_dict()[k])
    for k,v in optim['state'].items():
        for name,value in v.items():
            assert torch.equal(value,model.actor.optimizer.state_dict()['state'][k][name])
    model.anchor_max_kl=.1
    model.save(tmp_path/'candidate')
    loaded=BalancedAnchorSAC.load(tmp_path/'candidate',device='cuda')
    assert loaded.balanced_anchor_version==VERSION
    for k,v in before.items():assert torch.equal(v,loaded.actor.state_dict()[k])
    with pytest.raises(ValueError,match='explicit policy'):
        GeometricSAC.load(tmp_path/'candidate',device='cuda')
    with pytest.raises(ValueError,match='version'):
        BalancedAnchorSAC.load(source,device='cuda')


def test_combined_repair_only_accepts_strict_improvement_above_limit():
    from src.balanced_repair_policy import BalancedRepairSAC
    model=object.__new__(BalancedRepairSAC);model.anchor_max_kl=.1
    assert model._accept_anchor_proposal(.5,.49)
    assert not model._accept_anchor_proposal(.5,.499)
    assert not model._accept_anchor_proposal(.5,.51)
    assert not model._accept_anchor_proposal(.09,.101)
    assert model._accept_anchor_proposal(.09,.1)
    assert not model._accept_anchor_proposal(float('nan'),.01)
