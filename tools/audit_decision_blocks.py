"""Compare learned 12-step prefixes with repeated closed-loop learned continuations."""
import argparse
import json
from pathlib import Path
import pickle
import shutil
import time
import numpy as np
import torch
from stable_baselines3 import SAC
from tools.audit_critic_rankings import sample,soft_return
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--smoke',action='store_true');args=p.parse_args();out=args.out
    parent=Path('D:/DroneSoccerData/runs/critic-independent-families-20261007')
    torch.set_num_threads(1)
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        assert (parent/'DONE').exists()
        out.mkdir(parents=True,exist_ok=False);shutil.copy2(parent/'model.zip',out/'model.zip')
        shutil.copy2(__file__,out/'collector_source.py')
        model=SAC.load(out/'model.zip',device='cpu');alpha=float(model.log_ent_coef.detach().exp());gamma=model.gamma
        hashes={str(parent/'model.zip'):sha(parent/'model.zip')}
        states=1 if args.smoke else 64;replicas=2 if args.smoke else 6;records=[]
        write_json(out/'protocol.json',dict(kind='12-step learned prefix diagnostic',states=states,candidates=4,
            replicas=replicas,block_steps=12,actor_updates=0,acceptance=False,
            model_sha256=sha(out/'model.zip'),source_run=str(parent),wall_seconds=1800,
            actor_inputs='Unchanged public121. Every movement sampled at actual observations.',
            interpretation='Conditioned on four realized learned prefixes, not expected open-loop action-sequence values',
            split='Same state-index mod5 heldout as source; whole reset families stay together',
            terminal_prefixes='Logged once, repeated rows represent a fixed terminal prefix, not independent flights; analyze separately'))
        deadline=time.monotonic()+1800;actual_tail_rollouts=0;terminal_blocks=0
        for state in range(states):
            folder=out/f'state_{state:03d}';folder.mkdir()
            source_folder=parent/f'state_{state:03d}'
            snapshot_path=source_folder/'environment.pkl';hashes[str(snapshot_path)]=sha(snapshot_path)
            for candidate in range(4):
                saved_path=source_folder/f'action_{candidate}_replica_0.npz';hashes[str(saved_path)]=sha(saved_path)
                with np.load(saved_path) as saved:
                    obs=saved['observations'][0].copy();first=saved['actions'][0].copy()
                env=pickle.loads(snapshot_path.read_bytes())
                env.unwrapped.np_random=np.random.default_rng(11640000+state)
                torch.manual_seed(11650000+state*4+candidate)
                xs=[];actions=[];rewards=[];logps=[];positions=[]
                done=cut=False
                for step in range(12):
                    if step==0:
                        with torch.no_grad():
                            x=torch.tensor(obs[None],dtype=torch.float32)
                            mean,std,kwargs=model.actor.get_action_dist_params(x)
                            distribution=model.actor.action_dist.proba_distribution(mean,std,**kwargs)
                            logp=float(distribution.log_prob(torch.tensor(first[None]).clamp(-1+1e-6,1-1e-6))[0])
                        action=first
                    else:action,logp=sample(model,obs)
                    xs.append(obs.copy());actions.append(action.copy());logps.append(logp)
                    obs,reward,done,cut,info=env.step(action);rewards.append(reward);positions.append(env.unwrapped.pos.copy())
                    if done or cut:break
                np.savez_compressed(folder/f'block_{candidate}.npz',observations=xs,actions=actions,
                    rewards=rewards,logps=logps,positions=positions,next_observation=obs)
                after=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
                (folder/f'block_{candidate}_state.pkl').write_bytes(after)
                prefix_terminal=bool(done or cut);terminal_blocks+=int(prefix_terminal)
                for replica in range(replicas):
                    if time.monotonic()>deadline:raise RuntimeError('Bounded block diagnostic time cap')
                    branch=pickle.loads(after);current=obs.copy();tail_x=[];tail_a=[];tail_r=[];tail_l=[];tail_p=[]
                    end_done,end_cut=done,cut;last_info=info
                    seed=11660000+state*100+replica
                    branch.unwrapped.np_random=np.random.default_rng(seed);torch.manual_seed(seed)
                    if not prefix_terminal:actual_tail_rollouts+=1
                    while not(end_done or end_cut):
                        action,logp=sample(model,current);tail_x.append(current.copy());tail_a.append(action.copy());tail_l.append(logp)
                        current,reward,end_done,end_cut,last_info=branch.step(action)
                        tail_r.append(reward);tail_p.append(branch.unwrapped.pos.copy())
                    tail=0.
                    if end_cut and not end_done:
                        with torch.no_grad():
                            x=torch.tensor(current[None],dtype=torch.float32);a,l=model.actor.action_log_prob(x)
                            tail=float(torch.cat(model.critic_target(x,a),1).min()-alpha*l[0])
                    all_r=rewards+tail_r;all_l=logps+tail_l
                    row=dict(state=state,candidate=candidate,replica=replica,rng_seed=seed,
                        prefix_terminal=prefix_terminal,prefix_steps=len(actions),tail_steps=len(tail_a),
                        finite_soft_return=soft_return(all_r,all_l,gamma,alpha),
                        bootstrapped_soft_return=soft_return(all_r,all_l,gamma,alpha,tail),
                        mission=bool(end_cut and not end_done and branch.env.audit.count>=2 and last_info['wall_steps']==0),
                        outcome=last_info['outcome'],full_passes=branch.env.audit.count,returns=branch.env.audit.returns,
                        wall_steps=last_info['wall_steps'])
                    np.savez_compressed(folder/f'block_{candidate}_tail_{replica}.npz',observations=tail_x,
                        actions=tail_a,rewards=tail_r,logps=tail_l,positions=tail_p,next_observation=current)
                    records.append(row)
                    with (out/'continuations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    branch.close()
                env.close()
            write_json(out/'status.json',dict(states_completed=state+1,actual_tail_rollouts=actual_tail_rollouts,
                terminal_prefixes=terminal_blocks,actor_updates=0))
            print('STATE',state+1,flush=True)
        write_json(out/'input_hashes.json',hashes)
        (out/'DONE').write_text('Bounded learned block diagnostic complete. No actor trained.\n')


if __name__=='__main__':main()
