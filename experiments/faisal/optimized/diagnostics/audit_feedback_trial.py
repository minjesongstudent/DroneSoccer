"""Local three-command feedback linearization; excludes physical/visual dynamics."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch
ROOT=Path(__file__).parent
RUN=ROOT/'feedback-regularization-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.metric_policy import MetricRingSAC
torch.set_num_threads(1)
paths={'source':Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip'),'update10':RUN/'model_10.zip'}
hashes={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in paths.items()}
events=json.loads((ROOT/'history-guard-rl-20260928/first_divergence/summary.json').read_text())
results=[]
for name,path in paths.items():
    model=MetricRingSAC.load(path,device='cuda')
    for event in events:
        trace=json.loads((ROOT/'history-guard-rl-20260928/first_divergence'/f"original_{event['seed']}.json").read_text())['trace']
        step=event['first_large_observation']['step']
        x=torch.tensor(trace[step]['observation'],device='cuda',requires_grad=True)
        jac=torch.autograd.functional.jacobian(lambda v:model.actor(v[None],deterministic=True)[0],x).detach().cpu().numpy()[:,72:84]
        # Actual command deque is oldest first; append current action after shifting.
        matrix=np.zeros((12,12));matrix[:8,4:]=np.eye(8);matrix[8:]=jac
        eigenvalues=np.linalg.eigvals(matrix)
        results.append(dict(model=name,seed=event['seed'],step=step,
            spectral_radius=float(np.abs(eigenvalues).max()),largest_singular_value=float(np.linalg.svd(matrix,compute_uv=False)[0]),
            eigenvalues=[[float(v.real),float(v.imag)] for v in eigenvalues]))
assert all(hashlib.sha256(v.read_bytes()).hexdigest()==hashes[k] for k,v in paths.items())
summary={name:dict(radius_min=min(r['spectral_radius'] for r in results if r['model']==name),radius_max=max(r['spectral_radius'] for r in results if r['model']==name),above_one=sum(r['spectral_radius']>1 for r in results if r['model']==name)) for name in paths}
report=dict(summary=summary,rows=results,hashes=hashes,training_updates=0,
    caveat='Frozen non-command inputs and local linearization only. Camera, memory, odometry, physical dynamics and future changes in derivative excluded. Neither whole-flight stability proof nor a robust-control guarantee.')
(RUN/'command_feedback_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(summary,indent=2))
