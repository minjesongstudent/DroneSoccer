from pathlib import Path
from copy import deepcopy
import zipfile
import pytest,torch
from src.source_anchor_policy import SourceAnchorSAC,start_from_checkpoint
from src.balanced_anchor_policy import BalancedAnchorSAC
from src.return_reset_curriculum import wrap
from src.return_reset_lessons import ReturnResetLesson

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA source transfer')
@pytest.mark.parametrize('standard',[False,True])
def test_distinct_frozen_references_zero_initial_kl_and_strict_archive(tmp_path,standard):
    if standard:
        from src.standard_source_policy import StandardSourceSAC as Policy,start_from_checkpoint as transfer
    else:
        Policy,transfer=SourceAnchorSAC,start_from_checkpoint
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/return-route-40k-20260927/best_model.zip'
    previous=Path('D:/DroneSoccerData/runs/balanced-anchor-20260928/initial_model.zip')
    if not source.exists() or not previous.exists():pytest.skip('Preserved sources required')
    torch.set_num_threads(1)
    env=wrap(ReturnResetLesson(0),metric_ring=True,geometric_return=True)
    model,metadata=transfer(source,env,device='cuda',seed=7,buffer_size=5000,tensorboard_log=tmp_path)
    old=BalancedAnchorSAC.load(previous,device='cuda')
    for k,v in old.actor.state_dict().items():
        assert torch.equal(v,model.actor.state_dict()[k])
        assert torch.equal(v,model.source_anchor_actor.state_dict()[k])
    for k,v in old.reference_actor.state_dict().items():assert torch.equal(v,model.reference_actor.state_dict()[k])
    assert any(not torch.equal(v,model.reference_actor.state_dict()[k]) for k,v in model.source_anchor_actor.state_dict().items())
    assert not any(p.requires_grad for p in model.source_anchor_actor.parameters())
    assert not any(p.requires_grad for p in model.reference_actor.parameters())
    obs,_=env.reset(seed=333)
    x=torch.as_tensor(obs,device='cuda')[None]
    assert model._protected_kl(x,model._protection_mask(x),model._reference_parameters(x)).item()==0
    refs=[deepcopy(a.state_dict()) for a in [model.reference_actor,model.source_anchor_actor]]
    model._actor_step(x,model.ent_coef_tensor)
    for actor,state in zip([model.reference_actor,model.source_anchor_actor],refs):
        for k,v in state.items():assert torch.equal(v,actor.state_dict()[k])
    model.save(tmp_path/'source')
    restored=Policy.load(tmp_path/'source',device='cuda')
    for actor,state in zip([restored.reference_actor,restored.source_anchor_actor],refs):
        for k,v in state.items():assert torch.equal(v,actor.state_dict()[k])
    with pytest.raises(ValueError,match='explicit policy class'):BalancedAnchorSAC.load(tmp_path/'source',device='cuda')
    with pytest.raises(ValueError):Policy.load(previous,device='cuda')
    with zipfile.ZipFile(tmp_path/'source.zip') as archive,zipfile.ZipFile(tmp_path/'missing.zip','w') as dest:
        for name in archive.namelist():
            if name!='source_anchor_actor.pth':dest.writestr(name,archive.read(name))
    with pytest.raises(ValueError,match='BOTH'):Policy.load(tmp_path/'missing.zip',device='cuda')
    assert metadata['original_reference_used_for_penalty'] is False
    env.close()


def test_standard_gradient_reuses_exact_original_update():
    from src.standard_source_policy import StandardSourceSAC
    from src.anchored_policy import AnchoredSAC
    assert StandardSourceSAC._actor_step is AnchoredSAC._actor_step
