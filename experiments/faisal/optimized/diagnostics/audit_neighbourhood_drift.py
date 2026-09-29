"""Measure update drift on identical authentic teacher histories, FIT only."""
from pathlib import Path
import sys,json
import torch
import numpy as np
R=Path(__file__).parent
sys.path.insert(0,str(R/'neighbourhood-rl-20260928/source_snapshot'))
from src.metric_policy import MetricRingSAC
from src.geometric_policy import GeometricSAC
from src.anchored_policy import gaussian_kl
torch.set_num_threads(1)
old=MetricRingSAC.load('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip',device='cuda')
new=GeometricSAC.load(R/'neighbourhood-rl-20260928/model_16000.zip',device='cuda')
seeds=json.loads((R/'neighbourhood-rl-20260928/protocol.json').read_text())['fit_seeds']
obs=np.concatenate([np.load(R/'teacher-neighbours-20260928'/f'original_{seed}.npz')['observations'][:-1] for seed in seeds])
with torch.inference_mode():
    x=torch.as_tensor(obs,device='cuda')
    mask=new._protection_mask(x)
    mu,sd,_=new.actor.get_action_dist_params(x)
    ref,rsd,_=old.actor.get_action_dist_params(x)
    kl=gaussian_kl(mu,sd,ref,rsd)
    delta=(mu.tanh()-ref.tanh()).square().mean(1)
    result={name:dict(count=int(m.sum()),mean_kl=float(kl[m].mean()),mean_action_mse=float(delta[m].mean())) for name,m in [('protected',mask),('unprotected',~mask)]}
(R/'neighbourhood-rl-20260928/drift_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
