"""Locate accepted checkpoint drift on preserved first-batch public histories."""
from pathlib import Path
from types import SimpleNamespace
import sys,json
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/independent-baseline-actor-20260929'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_update_guard import distributions
torch.set_num_threads(1)
entries=json.loads((RUN/'protocol.json').read_text())['snapshots'];segments=[]
for i in range(len(entries)):
    with np.load(RUN/f'rollout_0_{i}.npz') as a:
        segments.append(SimpleNamespace(prefix=a['prefix'].copy(),observations=a['observations'].copy(),natural_start=bool(a['natural_start'])))
reference=torch.load(RUN/'guard_reference.pt',map_location='cpu',weights_only=True)
model=RecurrentPPO.load(RUN/'model_2.zip',device='cuda');candidate=distributions(model,segments)
mu,sigma=reference[:,0].double(),reference[:,1].double();nu,tau=candidate[:,0].double(),candidate[:,1].double()
kl=(torch.log(tau/sigma)+(sigma.square()+(mu-nu).square())/(2*tau.square())-.5).sum(1).numpy()
rows=[];offset=0
for i,(e,s) in enumerate(zip(entries,segments)):
    values=kl[offset:offset+len(s.observations)];offset+=len(values);peak=int(values.argmax())
    group=('early_return' if e['group']<8 else 'reapproach') if 'natural' not in e else e['natural']
    rows.append(dict(index=i,group=group,mean=float(values.mean()),max=float(values.max()),peak_step=peak,frames=len(values)))
result=dict(checkpoint=2,mean=float(kl.mean()),maximum=float(kl.max()),rows=rows,
    limitation='Accepted checkpoint only; rejected proposal parameters were rolled back and not saved')
(RUN/'kl_locations.json').write_text(json.dumps(result,indent=2));print(json.dumps(sorted(rows,key=lambda r:r['max'],reverse=True)[:8],indent=2))
