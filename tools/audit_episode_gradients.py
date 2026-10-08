"""Diagnostic only: gradient consistency of original-policy mission rollouts."""
import argparse
from pathlib import Path
import shutil

import numpy as np
import torch
from stable_baselines3 import SAC

from tools.run_onpolicy_trial import action_log_prob
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();out=args.out
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(1)
        assert torch.cuda.is_available()
        source=ROOT/'trainedModels/seenflag-s1-level1.zip'
        rollout=Path('D:/DroneSoccerData/runs/mission-objective-20261007/rollout_01.npz')
        with np.load(rollout) as z:
            observations=z['observations'].copy();actions=z['actions'].copy()
            targets=z['returns'].copy();lengths=z['episode_lengths'].copy();old=z['old_log_prob'].copy()
        model=SAC.load(source,device='cuda')
        x=torch.tensor(observations,device='cuda');a=torch.tensor(actions,device='cuda')
        with torch.no_grad():
            actual=action_log_prob(model.actor,x,a).cpu().numpy()
        np.testing.assert_allclose(actual,old,rtol=1e-5,atol=1e-5)
        starts=np.r_[0,np.cumsum(lengths)]
        success=np.array([targets[start] for start in starts[:-1]])
        gradients=[]
        for i,(start,end) in enumerate(zip(starts[:-1],starts[1:])):
            assert np.all(targets[start:end]==success[i])
            # Leave-this-episode-out constant baseline: no action-dependent value approximation.
            baseline=float((success.sum()-success[i])/(len(success)-1))
            model.actor.zero_grad()
            objective=(float(success[i])-baseline)*action_log_prob(model.actor,x[start:end],a[start:end]).sum()
            objective.backward()
            gradients.append(torch.cat([p.grad.flatten() for p in model.actor.parameters()]).cpu().numpy())
        g=np.stack(gradients).astype(np.float64);mean=g.mean(axis=0)
        signal=float(np.linalg.norm(mean));noise=float(np.sqrt(g.var(axis=0,ddof=1).sum()/len(g)))
        rng=np.random.default_rng(11560000);cosines=[]
        for _ in range(200):
            order=rng.permutation(len(g));left=g[order[:len(g)//2]].mean(axis=0);right=g[order[len(g)//2:]].mean(axis=0)
            cosines.append(float(left@right/(np.linalg.norm(left)*np.linalg.norm(right)+1e-20)))
        np.savez_compressed(out/'episode_gradients.npz',gradients=g.astype(np.float32),success=success,lengths=lengths,split_cosines=cosines)
        result=dict(episodes=len(g),successes=int(success.sum()),source_sha256=sha(source),rollout_sha256=sha(rollout),
            source_likelihood_matches_saved=True,mean_gradient_norm=signal,estimated_mean_noise_norm=noise,
            signal_to_noise_norm_ratio=signal/noise,median_split_cosine=float(np.median(cosines)),
            split_cosine_10_90=np.quantile(cosines,[.1,.9]).tolist(),fraction_negative=float(np.mean(np.array(cosines)<0)),
            caveat='Only sixteen source episodes; split diagnostics reuse data and are not independent confidence intervals. No policy updated.',
            actor_updates=0,acceptance=False)
        write_json(out/'report.json',result);shutil.copy2(__file__,out/'audit_source.py');print(result)


if __name__=='__main__':main()
