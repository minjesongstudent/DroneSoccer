from pathlib import Path
import sys,json
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/recurrent-value-calibration-20260928'
DATA=ROOT/'diagnostics/acquisition-return-mixture-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory

def main():
    torch.set_num_threads(1);m=RecurrentPPO.load(RUN/'model_16.zip',device='cuda');m.policy.set_training_mode(False)
    protocol=json.loads((RUN/'protocol.json').read_text());rows=[]
    constant=float(np.concatenate([np.load(DATA/f'rollout_0_{i}.npz')['returns'] for i in protocol['fit']]).mean())
    for i in protocol['heldout']:
        a=np.load(DATA/f'rollout_0_{i}.npz');natural=bool(a['natural_start'])
        state,_=_initial_memory(m,a['prefix'],natural);starts=torch.zeros(len(a['observations']),device='cuda')
        if natural:starts[0]=1
        with torch.no_grad():pred=m.policy.predict_values(torch.as_tensor(a['observations'],device='cuda'),state.vf,starts).flatten().cpu().numpy()*40
        target=a['returns'];group='early_return' if i<8 else 'reapproach' if i<15 else 'natural_full' if i<23 else 'basic'
        rows.append(dict(index=i,group=group,frames=len(target),sse=float(np.sum((pred-target)**2)),
            constant_sse=float(np.sum((constant-target)**2)),bias=float(np.mean(pred-target))))
    summary={}
    for group in ('early_return','reapproach','natural_full','basic'):
        g=[r for r in rows if r['group']==group];n=sum(r['frames'] for r in g)
        summary[group]=dict(episodes=len(g),frames=n,rmse=float(np.sqrt(sum(r['sse'] for r in g)/n)),
            constant_rmse=float(np.sqrt(sum(r['constant_sse'] for r in g)/n)))
    (RUN/'category_audit.json').write_text(json.dumps(dict(summary=summary,rows=rows),indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
