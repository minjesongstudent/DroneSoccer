"""Offline parameter ablation on saved FIT inputs; no rollout or optimizer step."""
import argparse
import copy
from pathlib import Path
import pickle
import shutil
import numpy as np
import torch
from stable_baselines3 import SAC
from src.twelve_action_history import extend_twelve_replay,extend_twelve_model
from src.constrained_sac import episode_partition,gaussian_kl
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(1)
    assert torch.cuda.is_available()
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        args.out.mkdir(parents=True,exist_ok=False)
        source=ROOT/'trainedModels/seenflag-s1-level1.zip'
        candidate=Path('D:/DroneSoccerData/runs/twelve-action-rl-20261007/online_005000/model.zip')
        replay=Path('D:/DroneSoccerData/runs/clearance-reward-20261007/clearance/step_200000/replay_buffer.pkl')
        with replay.open('rb') as stream: data=pickle.load(stream)
        obs,_=extend_twelve_replay(data.observations[:data.pos,0],data.next_observations[:data.pos,0],data.dones[:data.pos])
        fit,held,_=episode_partition(data.dones[:data.pos])
        rng=np.random.default_rng(11570000)
        ids={name:rng.choice(indices,min(len(indices),8192),replace=False) for name,indices in [('fit',fit),('heldout',held)]}
        np.savez_compressed(args.out/'indices.npz',**ids)
        original=extend_twelve_model(SAC.load(source,device='cuda')).actor
        trained=SAC.load(candidate,device='cuda').actor
        # Parameter ablations are offline diagnostics, never flight policies.
        original_plus_new=copy.deepcopy(original)
        trained_without_new=copy.deepcopy(trained)
        with torch.no_grad():
            original_plus_new.latent_pi[0].weight[:,72:108].copy_(trained.latent_pi[0].weight[:,72:108])
            trained_without_new.latent_pi[0].weight[:,72:108].zero_()
        report=dict(actor_updates=0,rollouts=0,acceptance=False,
            hashes={str(path):sha(path) for path in [source,candidate,replay]},
            description='Offline weight ablations; all observation/action history values remain recorded values',groups={})
        with torch.no_grad():
            for group,indices in ids.items():
                x=torch.tensor(obs[indices],device='cuda')
                ref=original.get_action_dist_params(x)[:2]
                actual=trained.get_action_dist_params(x)[:2]
                metrics={}
                for name,actor in [('trained',trained),('source_plus_new_columns',original_plus_new),('trained_without_new_columns',trained_without_new)]:
                    dist=actor.get_action_dist_params(x)[:2]
                    kl=gaussian_kl(*ref,*dist)
                    delta=dist[0].tanh()-ref[0].tanh()
                    from_trained=dist[0].tanh()-actual[0].tanh()
                    metrics[name]=dict(mean_source_kl=float(kl.mean()),max_source_kl=float(kl.max()),
                        mean_abs_action_change_from_source=float(delta.abs().mean()),
                        mean_abs_action_change_from_trained=float(from_trained.abs().mean()))
                first=trained.latent_pi[0]
                new_activation=torch.nn.functional.linear(x[:,72:108],first.weight[:,72:108])
                old_activation=torch.nn.functional.linear(torch.cat([x[:,:72],x[:,108:]],1),
                    torch.cat([first.weight[:,:72],first.weight[:,108:]],1),first.bias)
                metrics['first_layer']=dict(new_rms=float(new_activation.square().mean().sqrt()),
                    existing_rms=float(old_activation.square().mean().sqrt()),
                    added_weight_norm=float(first.weight[:,72:108].norm()))
                report['groups'][group]=metrics
        write_json(args.out/'report.json',report)
        shutil.copy2(__file__,args.out/'audit_source.py')
        print(report)


if __name__=='__main__':main()
