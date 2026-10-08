"""Bounded source-constrained imitation of successful FIT learned trajectories."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch

from src.clearance_training import PROFILE
from src.constrained_sac import ConstrainedSAC, gaussian_kl, boundary_penalty, guarded_optimizer_step
from tools.run_constrained_clearance import PARENT
from tools.run_clearance_experiment import ROOT, evaluate, retention_failures, sha, single_trainer, write_json


def success_indices(rows, capacity):
    selected=[]; cursor=0
    for episode,row in enumerate(rows):
        end=cursor+row['steps']
        if end>capacity: break
        if (episode%5!=4 and row['steps']==600 and row['status']=='complete' and row['outcome']=='timeout'
                and row['full_body_passes']>=2 and row['wall_steps']==0):
            selected.extend(range(cursor,end))
        cursor=end
    return np.asarray(selected,dtype=np.int64)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--four-actions',action='store_true')
    args=parser.parse_args(); out=args.out.resolve()
    assert torch.cuda.is_available()
    torch.set_num_threads(1)
    torch.manual_seed(11530000); np.random.seed(11530000)
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        source=ROOT/'trainedModels/seenflag-s1-level1.zip'
        assert sha(source)==PROFILE.sha256
        replay=PARENT/'clearance/step_200000/replay_buffer.pkl'
        with replay.open('rb') as stream: data=pickle.load(stream)
        assert data.pos==200000 and not data.full
        if args.four_actions:
            from src.action_history_extension import extend_replay_observations
            obs,nxt=extend_replay_observations(data.observations[:data.pos,0],
                data.next_observations[:data.pos,0],data.dones[:data.pos])
            data.observations=obs[:,None,:];data.next_observations=nxt[:,None,:]
        rows=[json.loads(line) for line in (PARENT/'clearance/training_trajectories/episodes.jsonl').read_text().splitlines()]
        # Verify logged episode boundaries against the actual replay before selection.
        cursor=0
        for row in rows:
            end=cursor+row['steps']
            if end>data.pos: break
            assert data.dones[end-1,0] and not data.dones[cursor:end-1].any()
            cursor=end
        indices=success_indices(rows,data.pos)
        assert len(indices)>256
        np.save(out/'successful_fit_indices.npy',indices)
        hashes={}
        for file in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tests/test_success_imitation.py',ROOT/'tests/test_action_history_extension.py',ROOT/'tools/evaluate_four_actions.py',ROOT/'tools/run_clearance_experiment.py',ROOT/'tools/validate_flight_policies.py',source]:
            rel=file.relative_to(ROOT); dest=out/'source'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);hashes[str(rel)]=sha(dest)
        protocol=dict(kind='FIT success-filtered learned-action imitation',source_hashes=hashes,
            action_history=4 if args.four_actions else 3,inputs=89 if args.four_actions else 85,
            replay_sha256=sha(replay),selected_transitions=len(indices),heldout='episode index mod5 ==4 excluded',
            selection='complete600-step timeout; >=2 full clearances; no wall contact; FIT only',
            loss='negative log likelihood of actual source actions + source KL + boundary penalty .1',
            action_clamp='1e-6 inside tanh bounds for stable likelihood only; no altered flight actions',
            actor_lr=1e-6,mean_kl_limit=.01,max_kl_limit=.1,checks=[2] if args.smoke else [100,500,1000,2000],
            wall_seconds=120 if args.smoke else 1800,training_device='cuda',acceptance=False,physical_flight=False)
        write_json(out/'protocol.json',protocol)
        model=ConstrainedSAC.load(source,device='cuda')
        evaluator=evaluate
        if args.four_actions:
            from src.action_history_extension import extend_model
            from tools.evaluate_four_actions import evaluate_four_actions
            model=extend_model(model)
            evaluator=evaluate_four_actions
        # Guard covers all FIT behaviors, including failures, not just imitation examples.
        from src.constrained_sac import episode_partition
        fit,_,_=episode_partition(data.dones[:data.pos])
        guard=np.random.default_rng(11520000).choice(fit,size=1024,replace=False)
        model.configure(data.observations[guard,0],boundary_weight=.1)
        baseline=json.loads((PARENT/'reference_guard/summary.json').read_text())
        if args.four_actions:
            initial=out/'initial_89.zip';model.save(initial)
            initial_metrics=evaluator(initial,out/'initial_parity_guard',2 if args.smoke else 25)
            if not args.smoke and initial_metrics!=baseline:
                raise RuntimeError('Expanded initial policy guard differs from original; inspect before training')
        result=dict(status='running',checks=[],acceptance=False)
        deadline=time.monotonic()+protocol['wall_seconds'];accepted=rejected=consecutive=0
        for update in range(1,max(protocol['checks'])+1):
            ids=np.random.choice(indices,256,replace=True)
            obs=torch.as_tensor(data.observations[ids,0],device='cuda')
            actions=torch.as_tensor(data.actions[ids,0],device='cuda').clamp(-1+1e-6,1-1e-6)
            mean,std,kwargs=model.actor.get_action_dist_params(obs)
            distribution=model.actor.action_dist.proba_distribution(mean,std,**kwargs)
            nll=-distribution.log_prob(actions).mean()
            with torch.no_grad(): rm,rs,_=model.reference_actor.get_action_dist_params(obs)
            gm,gs,_=model.actor.get_action_dist_params(model.trust_observations)
            loss=nll+gaussian_kl(rm,rs,mean,std).mean()+.1*boundary_penalty(
                gaussian_kl(*model.trust_distribution,gm,gs),.01,.1)
            assert torch.isfinite(loss)
            ok,drift=guarded_optimizer_step(model.actor,model.actor.optimizer,loss,model.trust_check)
            accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
            stop=consecutive>=100 or time.monotonic()>=deadline or (out/'STOP').exists()
            if update in protocol['checks'] or stop:
                checkpoint=out/f'actor_{update:06d}.zip';model.save(checkpoint)
                torch.save(dict(torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state()),out/f'rng_{update:06d}.pt')
                metrics=evaluator(checkpoint,out/f'guard_{update:06d}',2 if args.smoke else 25)
                failures=[] if args.smoke else retention_failures(baseline,metrics)
                record=dict(update=update,accepted=accepted,rejected=rejected,drift=drift,metrics=metrics,failures=failures)
                result['checks'].append(record);write_json(out/'result.json',result);print(json.dumps(record),flush=True)
                if failures or stop:
                    result['status']='retention_regression' if failures else 'bounded_stop';break
        else: result['status']='bounded_trial_complete'
        result['final_development']=evaluator(checkpoint,out/'final_development',2 if args.smoke else 100)
        result['source_unchanged']=sha(source)==PROFILE.sha256
        write_json(out/'result.json',result)


if __name__=='__main__': main()
