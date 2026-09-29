"""Local derivatives on authentic public observations; no modified-input flights."""
from pathlib import Path
import sys,json
import numpy as np
import torch
R=Path(__file__).parent/'history-guard-rl-20260928'
sys.path.insert(0,str(R/'source_snapshot'))
from src.history_guard_policy import HistoryGuardSAC
torch.set_num_threads(1)
m=HistoryGuardSAC.load(R/'model_0.zip',device='cuda')
summary=json.loads((R/'first_divergence/summary.json').read_text())
groups={'camera':(0,72),'commands':(72,84),'any_memory':(84,99),'front_memory':(100,115),'estimate':(115,119)}
rows=[]
for entry in summary:
    trace=json.loads((R/'first_divergence'/f"original_{entry['seed']}.json").read_text())['trace']
    step=entry['first_large_observation']['step']
    x=torch.tensor(trace[step]['observation'],device='cuda',requires_grad=True)
    def action(value):
        mu,_,_=m.actor.get_action_dist_params(value[None])
        return mu[0].tanh()
    j=torch.autograd.functional.jacobian(action,x).detach()
    rows.append(dict(seed=entry['seed'],step=step,
        frobenius={name:float(torch.linalg.vector_norm(j[:,lo:hi])) for name,(lo,hi) in groups.items()},
        command_spectral_norm=float(torch.linalg.matrix_norm(j[:,72:84],ord=2)),
        caveat='Local derivative only; groups have different dimensions/scales. Not proof of closed-loop instability.'))
(R/'command_sensitivity.json').write_text(json.dumps(rows,indent=2))
print(json.dumps(rows,indent=2))
