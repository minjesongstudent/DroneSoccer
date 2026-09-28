import sys,json
from pathlib import Path
from copy import deepcopy
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'source_snapshot'))
import numpy as np,torch
from src.standard_source_policy import StandardSourceSAC
from src.finite_policy_update import update_actor,gaussian_kl

torch.set_num_threads(1)
model=StandardSourceSAC.load('D:/DroneSoccerData/runs/standard-source-20260928/initial_model.zip',device='cuda')
source=deepcopy(model.actor.state_dict())
saved=torch.load(ROOT/'candidate_0.pt',map_location='cuda',weights_only=False)
assert all(torch.equal(source[k],saved['actor'][k]) for k in source)
meta=json.loads((ROOT/'episodes_0.json').read_text());data=np.load(ROOT/'rollouts_0.npz')
episodes=[dict(m,**{k:data[f'{i}_{k}'] for k in ('obs','actions','gaussian','logp','rewards','returns')}) for i,m in enumerate(meta)]
weights=[];origins=[]
for i,e in enumerate(episodes):
    peers=[p for p in episodes if p['group']==e['group'] and p is not e]
    base=np.asarray([np.mean([p['returns'][t] for p in peers if t<len(p['returns'])]) if any(t<len(p['returns']) for p in peers) else 0 for t in range(len(e['returns']))])
    mass=.4/9 if e['group']<9 else .2
    weights.extend((e['returns']-base)*.995**np.arange(len(base))*mass/8)
    origins.extend((i,t) for t in range(len(base)))
x,a,z=[torch.as_tensor(np.concatenate([e[k] for e in episodes]),device='cuda') for k in ('obs','actions','gaussian')]
w=torch.as_tensor(weights,dtype=torch.float32,device='cuda');w=w/w.abs().mean().clamp_min(1e-8)
lp=torch.as_tensor(np.concatenate([e['logp'] for e in episodes]),device='cuda')
proposal={};restore=model.actor.load_state_dict
def capture_then_restore(state,*args,**kwargs):
    proposal.update(deepcopy(model.actor.state_dict()))
    return restore(state,*args,**kwargs)
model.actor.load_state_dict=capture_then_restore
torch.manual_seed(4000000)
history=update_actor(model.actor,torch.optim.Adam(model.actor.parameters(),lr=3e-6),x,a,z,w,lp)
assert not history[0]['accepted']
assert all(torch.equal(source[k],v) for k,v in model.actor.state_dict().items())
with torch.no_grad():mu0,ls0,_=model.actor.get_action_dist_params(x)
restore(proposal)
with torch.no_grad():
    mu,ls,_=model.actor.get_action_dist_params(x);div=gaussian_kl(mu0,ls0,mu,ls)
    ids=torch.topk(div,10).indices.tolist();rows=[]
    for idx in ids:
        eid,t=origins[idx]
        rows.append(dict(episode=eid,time=t,group=episodes[eid]['group'],seed=episodes[eid]['seed'],
            kl=float(div[idx]),minimum_source_std=float(ls0[idx].exp().min()),
            max_deterministic_command_change=float((mu[idx].tanh()-mu0[idx].tanh()).abs().max()),
            single_task=float(x[idx,99]),visible=float(x[idx,70]),front_visible=float(x[idx,71])))
    groups=[]
    for group in range(12):
        mask=torch.as_tensor([episodes[eid]['group']==group for eid,t in origins],device='cuda')
        groups.append(dict(group=group,steps=int(mask.sum()),mean_kl=float(div[mask].mean()),max_kl=float(div[mask].max()),over_bound=int((div[mask]>.02).sum())))
restore(source)
assert all(torch.equal(source[k],v) for k,v in model.actor.state_dict().items())
result=dict(saved_candidate_equals_source=True,reproduced=history,largest_changes=rows,groups=groups,
            diagnostic_only=True,proposal_not_saved_or_promoted=True,flight_rollouts=0)
(ROOT/'rejection_audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
