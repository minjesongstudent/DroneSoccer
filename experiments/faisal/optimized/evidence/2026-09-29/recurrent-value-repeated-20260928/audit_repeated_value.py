"""Verify frozen actor, family sampling, and repeated-start FIT predictions."""
from pathlib import Path
import sys,json,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/recurrent-value-repeated-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory
torch.set_num_threads(1)
protocol=json.loads((RUN/'protocol.json').read_text());records=protocol['records']
selection=json.loads((RUN/'selections.json').read_text())
assert len(selection)==16*24*6
counts={}
for s in selection:
    r=records[s['index']]
    assert r['split']=='fit' and r['group']==s['group'] and r['family']==s['family'] and r['history']==s['history']
    key=(s['epoch'],s['group']);counts.setdefault(key,{})[s['family']]=counts.setdefault(key,{}).get(s['family'],0)+1
for (epoch,group),values in counts.items():
    families={r['family'] for r in records if r['group']==group and r['split']=='fit'}
    assert set(values)==families and max(values.values())-min(values.values())<=1
model=RecurrentPPO.load(RUN/'model_16.zip',device='cuda');p=model.policy;p.set_training_mode(False)
source=ROOT/'diagnostics/acquisition-return-mixture-20260928/model_0.zip'
assert hashlib.sha256(source.read_bytes()).hexdigest()==protocol['source_sha256']
original=RecurrentPPO.load(source,device='cuda').policy
critic_names=tuple(protocol['critic_names'])
assert all(torch.equal(v,dict(p.named_parameters())[n]) for n,v in original.named_parameters() if n not in critic_names)
rows=[]
for bank,is_return in [('authentic-return-variance-20260928',True),('basic-return-variance-20260928',False)]:
    base=ROOT/'diagnostics'/bank
    for summary in json.loads((base/'result.json').read_text())['summaries']:
        index=summary['index'];values=[]
        for repeat in range(8):
            with np.load(base/f'case_{index}_repeat_{repeat}.npz') as a:
                values.append(float(a['returns'][0]))
                if repeat==0:
                    with torch.no_grad():
                        state,start=_initial_memory(model,a['prefix'],not is_return)
                        prediction=float(p.predict_values(torch.as_tensor(a['observations'][0:1],device='cuda'),state.vf,start).item()*40)
        mean=float(np.mean(values));group=('early_return' if index<8 else 'reapproach') if is_return else summary['group']
        rows.append(dict(index=index,group=group,prediction=prediction,mean=mean,standard_error=float(np.std(values,ddof=1)/np.sqrt(8))))
metrics={g:dict(rmse=float(np.sqrt(np.mean([(r['prediction']-r['mean'])**2 for r in rows if r['group']==g]))),starts=sum(r['group']==g for r in rows)) for g in sorted({r['group'] for r in rows})}
result=dict(actor_exact=True,sampling_valid=True,selection_count=len(selection),fit_only=True,rows=rows,metrics=metrics,
    caution='Repeated means are now training-data summaries, not out-of-sample evidence')
(RUN/'audit.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(metrics=metrics,actor_exact=True,sampling_valid=True)))
