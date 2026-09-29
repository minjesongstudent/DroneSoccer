"""Compare CPU/CUDA sequence values on the worst FIT parity case."""
from pathlib import Path
import sys,json
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'diagnostics/recurrent-value-broad-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory
torch.set_num_threads(1)
records=json.loads((RUN/'protocol.json').read_text())['records']
results={}
for device in ('cpu','cuda'):
    model=RecurrentPPO.load(RUN/'model_16.zip',device=device);p=model.policy;p.set_training_mode(False)
    record=records[12];assert record['split']=='fit'
    with np.load(record['path']) as a,torch.no_grad():
        obs=torch.as_tensor(a['observations'],device=device)
        state,start=_initial_memory(model,a['prefix'],bool(a['natural_start']))
        starts=torch.zeros(len(obs),device=device);starts[0]=start[0]
        batch=p.predict_values(obs,state.vf,starts).flatten().cpu().numpy()*40
        seq=[]
        for frame in obs:
            _,v,_,state=p.forward(frame[None],state,start,deterministic=True)
            seq.append(v.item()*40);start=torch.zeros_like(start)
        results[device]=dict(batch=batch.tolist(),sequential=seq,max_error=float(np.max(np.abs(batch-seq))),
            rmse_batch=float(np.sqrt(np.mean((batch-a['returns'])**2))),rmse_step=float(np.sqrt(np.mean((np.array(seq)-a['returns'])**2))))
results['cpu_cuda_batch_max']=float(np.max(np.abs(np.array(results['cpu']['batch'])-results['cuda']['batch'])))
results['cpu_cuda_step_max']=float(np.max(np.abs(np.array(results['cpu']['sequential'])-results['cuda']['sequential'])))
(RUN/'numerics_audit.json').write_text(json.dumps(results,indent=2))
print(json.dumps({k:{n:v for n,v in val.items() if n not in ('batch','sequential')} if isinstance(val,dict) else val for k,val in results.items()},indent=2))
