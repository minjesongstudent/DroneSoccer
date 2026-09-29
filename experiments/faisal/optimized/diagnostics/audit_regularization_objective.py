"""Fixed-batch objective gradients and saturation; no optimization."""
from pathlib import Path
import sys,json,hashlib
import torch,numpy as np
R=Path(__file__).parent
RUN=R/'command-regularization-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.metric_policy import MetricRingSAC
torch.set_num_threads(1)
protocol=json.loads((RUN/'protocol.json').read_text())
paths=[Path(protocol['source']),RUN/'model_20.zip']
hashes=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
models=[MetricRingSAC.load(p,device='cuda') for p in paths]
bank=np.concatenate([np.load(R/'teacher-neighbours-20260928'/f'original_{s}.npz')['observations'][:-1] for s in protocol['fit_seeds']])
ids=np.load(RUN/'confirmation/derivative_indices.npy')
with np.load('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/data_600000.npz') as d:basic=d['observations'][d['mask'].astype(bool)]
bi=np.random.default_rng(943).choice(len(basic),512,replace=False)
b=torch.tensor(basic[bi],device='cuda')
with torch.no_grad():
    target=models[0].actor(torch.tensor(bank[ids],device='cuda'),deterministic=True)
    basic_target=models[0].actor(b,deterministic=True)
rows=[]
for name,m in zip(['source','update20'],models):
    x=torch.tensor(bank[ids],device='cuda',requires_grad=True)
    action=m.actor(x,deterministic=True)
    jac=torch.stack([torch.autograd.grad(action[:,k].sum(),x,create_graph=True,retain_graph=True)[0][:,72:84] for k in range(4)],dim=1)
    sensitivity=jac.square().sum((1,2)).mean()
    anchor=.5*((action-target).square().mean()+(m.actor(b,deterministic=True)-basic_target).square().mean())
    params=list(m.actor.parameters())
    def gradient(loss):
        grads=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
        return torch.cat([(torch.zeros_like(p) if g is None else g).reshape(-1) for p,g in zip(params,grads)])
    ga=gradient(anchor);gs=gradient(.001*sensitivity)
    total=ga+gs
    rows.append(dict(model=name,anchor=float(anchor.detach()),sensitivity=float(sensitivity.detach()),
        anchor_gradient_norm=float(ga.norm()),weighted_sensitivity_gradient_norm=float(gs.norm()),
        cosine=float(torch.dot(ga,gs)/(ga.norm()*gs.norm()).clamp_min(1e-20)),
        negative_gradient_directional_change_sensitivity=float(-torch.dot(total,gs)),
        saturated_action_fraction=float((action.detach().abs()>.95).float().mean()),
        saturated_derivative_energy=float((jac.square().sum(2)[action.detach().abs()>.95].sum()/jac.square().sum()).detach())))
assert hashes==[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]
result=dict(rows=rows,source_hashes=hashes,training_updates=0,
    caveat='Local gradients on a fixed authentic512lap+512basic batch. Directional change is for an infinitesimal plain-gradient step, not an Adam-step prediction or proof of flight improvement.')
(RUN/'objective_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
