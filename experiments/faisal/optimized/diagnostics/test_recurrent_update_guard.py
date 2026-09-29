from pathlib import Path
from types import SimpleNamespace
from copy import deepcopy
import sys,numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.recurrent_update_guard import distributions,guarded_step
torch.set_num_threads(1)
m=RecurrentPPO.load('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip',device='cuda')
a=np.load(ROOT/'diagnostics/authentic-return-pg-mixed-20260928/rollout_0_0.npz')
segments=[SimpleNamespace(prefix=a['prefix'],observations=a['observations'][:8])]
ref=distributions(m,segments);opt=torch.optim.Adam(m.policy.parameters(),lr=1e-6)
def grads():
    opt.zero_grad()
    m.policy.action_net.bias.sum().backward()
grads();r=guarded_step(m,opt,segments,ref);assert r['accepted'],r
before=deepcopy(m.policy.state_dict());ob=deepcopy(opt.state_dict())
for group in opt.param_groups:group['lr']=1.
ob=deepcopy(opt.state_dict())
grads();r=guarded_step(m,opt,segments,ref);assert not r['accepted'],r
def equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal(x,y)
    else:assert a==b
equal(before,m.policy.state_dict());equal(ob,opt.state_dict())
# Measurement errors must also restore the transaction.
grads()
try:guarded_step(m,opt,[],ref)
except RuntimeError:pass
else:raise AssertionError('Expected failed measurement')
equal(before,m.policy.state_dict());equal(ob,opt.state_dict())
print('PASS: accepted small step; rejected large step restores weights and populated Adam state; measurement exception restores both; no checkpoint saved')
