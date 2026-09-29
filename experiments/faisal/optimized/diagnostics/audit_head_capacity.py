"""Offline least-squares diagnostic of the frozen actor representation. No flights."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch

RUN=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO

def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    model=RecurrentPPO.load(RUN/'model_update_500.zip',device='cuda')
    model.policy.set_training_mode(False)
    seeds=json.loads((RUN/'protocol.json').read_text())['fit_lap_seeds']
    assert len(seeds)==9 and all(s%4!=3 for s in seeds)
    all_features=[];all_actions=[]
    def capture(_module,inputs):all_features.append(inputs[0].detach().double())
    hook=model.policy.action_net.register_forward_pre_hook(capture)
    with torch.inference_mode():
        for seed in seeds:
            with np.load(f'D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories/{seed}.npz') as data:
                obs=torch.as_tensor(data['observations'][:-1],device='cuda')
                all_actions.append(torch.as_tensor(data['actions'],device='cuda',dtype=torch.float64))
            z=torch.zeros(1,1,128,device='cuda')
            model.policy.get_distribution(obs,(z,z.clone()),torch.zeros(600,device='cuda'))
    hook.remove()
    X=torch.cat(all_features);Y=torch.cat(all_actions)
    X=torch.cat((X,torch.ones((len(X),1),device='cuda',dtype=torch.float64)),dim=1)
    # Full-SVD pseudoinverse avoids assuming independent/sufficiently excited features.
    U,S,Vh=torch.linalg.svd(X,full_matrices=False)
    cutoff=float(S.max())*1e-10
    inverse=torch.where(S>cutoff,1/S,torch.zeros_like(S))
    W=Vh.T@(inverse[:,None]*(U.T@Y))
    fitted=X@W
    baseline=X[:,:-1]@model.policy.action_net.weight.detach().double().T+model.policy.action_net.bias.detach().double()
    report=dict(kind='offline optimal linear readout on frozen features, not a deployed/trained policy',
        model_sha256=hashlib.sha256((RUN/'model_update_500.zip').read_bytes()).hexdigest(),
        device=torch.cuda.get_device_name(),fit_seeds=seeds,reserved_or_acceptance_used=False,
        samples=len(X),features=X.shape[1],effective_rank=int((S>cutoff).sum()),svd_cutoff=cutoff,
        baseline_unclipped_mse=float((baseline-Y).square().mean()),
        optimal_unclipped_mse=float((fitted-Y).square().mean()),
        optimal_clipped_mse=float((fitted.clip(-1,1)-Y).square().mean()),
        optimal_per_action_mse=(fitted-Y).square().mean(0).tolist(),
        per_episode_optimal_mse=(fitted-Y).square().reshape(9,600,4).mean((1,2)).tolist(),
        fitted_action_min=float(fitted.min()),fitted_action_max=float(fitted.max()),
        explanation='Best unregularized squared-error affine readout for these frozen features. No basic-retention constraint. Does not bound nonlinear models or clipped-output optimum. No weights written or environment steps taken.')
    (Path(__file__).parent/'head_capacity_audit.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
