"""Bounded paired learned-policy continuations to audit critic action rankings."""
import argparse
import copy
import json
from pathlib import Path
import pickle
import shutil
import time

import numpy as np
import torch
from stable_baselines3 import SAC
from src.clearance_training import ClearanceTrainingEnv
from src.twelve_action_history import TwelveActionObservation
from src.sim import HoopEnv
from tools.run_clearance_experiment import ROOT,sha,write_json,single_trainer


def sample(model,obs):
    with torch.no_grad():
        action,logp=model.actor.action_log_prob(torch.tensor(obs[None],dtype=torch.float32,device=model.device))
    return action[0].cpu().numpy(),float(logp[0])


def soft_return(rewards,logps,gamma,alpha,tail=0.):
    # Q(s,a) excludes entropy of the fixed first action; subsequent entropy is included.
    total=0.
    for i,(reward,logp) in enumerate(zip(rewards,logps)):
        total += gamma**i*(reward-(alpha*logp if i else 0.))
    return total+gamma**len(rewards)*tail


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--model',type=Path,default=Path('D:/DroneSoccerData/runs/twelve-adapter-20261007/online_005000/model.zip'))
    parser.add_argument('--seed-base',type=int,choices=[11200000,11220000,11230000],default=11200000)
    parser.add_argument('--states',type=int,choices=[12,64],default=12)
    parser.add_argument('--replicas',type=int,choices=[6,18],default=6)
    parser.add_argument('--replica-start',type=int,choices=[0,6],default=0)
    parser.add_argument('--reference-run',type=Path)
    args=parser.parse_args();out=args.out
    if args.replica_start and (args.reference_run is None or args.smoke):
        parser.error('Additional replicas require a reference run and cannot be a smoke test')
    torch.set_num_threads(1)
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        path=args.model
        shutil.copy2(path,out/'model.zip')
        if args.reference_run:
            assert (args.reference_run/'DONE').exists()
            assert sha(path)==sha(args.reference_run/'model.zip')
        model=SAC.load(out/'model.zip',device='cpu')
        alpha=float(model.log_ent_coef.detach().exp());gamma=model.gamma
        states=1 if args.smoke else args.states;replicas=2 if args.smoke else args.replicas;candidates=4
        max_attempts=64 if states<=12 else 384
        hashes={}
        for file in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/run_clearance_experiment.py']:
            rel=file.resolve().relative_to(ROOT);dest=out/'source'/rel
            dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,dest);hashes[str(rel)]=sha(dest)
        write_json(out/'protocol.json',dict(model_sha256=sha(out/'model.zip'),source_hashes=hashes,
            states=states,candidates=candidates,replicas=replicas,max_attempts=max_attempts,max_episode_steps=600,
            replica_start=args.replica_start,reference_run=str(args.reference_run) if args.reference_run else None,
            actor_updates=0,training=False,acceptance=False,seed_base=args.seed_base,
            state_selection='First surviving stochastic learned prefixes at20/60/120 steps, rotating by attempt',
            actions='Four sampled from current learned distribution; all continuations stochastic same learned actor',
            common_random_numbers='Each replica uses shared future environment RNG and actor RNG across candidate actions',
            outcomes='Finite soft return, timeout-bootstrapped soft return and full-episode mission separately',
            caveat='Finite replicas; bootstrapped target remains critic-dependent, not independent ground truth',
            gamma=gamma,alpha=alpha,wall_seconds=1800))
        deadline=time.monotonic()+1800;records=[];selected=0
        for attempt in range(max_attempts):
            if selected==states:break
            env=TwelveActionObservation(ClearanceTrainingEnv(HoopEnv(level=1,randomise=True,hanger=True),'clearance'))
            obs,_=env.reset(seed=args.seed_base+attempt)
            torch.manual_seed(11580000+attempt)
            prefix=[];prefix_length=[20,60,120][attempt%3]
            alive=True
            for _ in range(prefix_length):
                action,logp=sample(model,obs)
                prefix.append(dict(observation=obs.copy(),action=action.copy(),logp=logp))
                obs,_,done,cut,_=env.step(action)
                if done or cut:alive=False;break
            with (out/'selection.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(attempt=attempt,seed=args.seed_base+attempt,prefix_length=len(prefix),selected=alive))+'\n')
            if not alive:env.close();continue
            folder=out/f'state_{selected:03d}';folder.mkdir()
            snapshot=pickle.dumps(env,protocol=pickle.HIGHEST_PROTOCOL)
            (folder/'environment.pkl').write_bytes(snapshot)
            torch.save(torch.get_rng_state(),folder/'sampling_rng.pt')
            np.savez_compressed(folder/'prefix.npz',observations=np.array([p['observation'] for p in prefix]),
                actions=np.array([p['action'] for p in prefix]),logps=np.array([p['logp'] for p in prefix]))
            choices=[sample(model,obs) for _ in range(candidates)]
            if args.reference_run:
                prior=args.reference_run/f'state_{selected:03d}'
                with np.load(prior/'prefix.npz') as saved:
                    np.testing.assert_array_equal(saved['observations'],np.array([p['observation'] for p in prefix]))
                    np.testing.assert_array_equal(saved['actions'],np.array([p['action'] for p in prefix]))
                for candidate,(action,_) in enumerate(choices):
                    with np.load(prior/f'action_{candidate}_replica_0.npz') as saved:
                        np.testing.assert_array_equal(saved['observations'][0],obs)
                        np.testing.assert_array_equal(saved['actions'][0],action)
            with torch.no_grad():
                x=torch.tensor(np.tile(obs,(candidates,1)),dtype=torch.float32)
                a=torch.tensor(np.array([v[0] for v in choices]),dtype=torch.float32)
                q=torch.cat(model.critic(x,a),1).min(1).values.numpy()
            outcomes=[]
            for candidate,(first,first_logp) in enumerate(choices):
                for replica in range(args.replica_start,args.replica_start+replicas):
                    if time.monotonic()>deadline:raise RuntimeError('Bounded diagnostic wall cap reached')
                    branch=pickle.loads(snapshot)
                    random_seed=11600000+selected*100+replica
                    branch.unwrapped.np_random=np.random.default_rng(random_seed)
                    torch.manual_seed(random_seed)
                    current=obs.copy();observations=[];actions=[];rewards=[];logps=[];positions=[]
                    step=0
                    while True:
                        action,logp=(first,first_logp) if step==0 else sample(model,current)
                        observations.append(current.copy());actions.append(action.copy());logps.append(logp)
                        current,reward,done,cut,info=branch.step(action)
                        rewards.append(reward);positions.append(branch.unwrapped.pos.copy());step+=1
                        if done or cut:break
                    tail=0.
                    if cut and not done:
                        with torch.no_grad():
                            last=torch.tensor(current[None],dtype=torch.float32)
                            next_action,next_logp=model.actor.action_log_prob(last)
                            tail=float(torch.cat(model.critic_target(last,next_action),1).min()-alpha*next_logp[0])
                    mission=bool(cut and not done and branch.env.audit.count>=2 and info['wall_steps']==0)
                    row=dict(state=selected,seed=args.seed_base+attempt,candidate=candidate,replica=replica,
                        rng_seed=random_seed,q=float(q[candidate]),finite_soft_return=soft_return(rewards,logps,gamma,alpha),
                        bootstrapped_soft_return=soft_return(rewards,logps,gamma,alpha,tail),tail=tail,
                        terminated=bool(done),truncated=bool(cut),outcome=info['outcome'],mission=mission,
                        full_passes=branch.env.audit.count,returns=branch.env.audit.returns,wall_steps=info['wall_steps'],steps=step)
                    np.savez_compressed(folder/f'action_{candidate}_replica_{replica}.npz',
                        observations=observations,actions=actions,rewards=rewards,logps=logps,positions=positions,next_observation=current)
                    outcomes.append(row)
                    with (out/'continuations.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
                    branch.close()
            aggregate=[]
            for candidate in range(candidates):
                rows=[r for r in outcomes if r['candidate']==candidate]
                aggregate.append(dict(candidate=candidate,q=float(q[candidate]),
                    finite_mean=float(np.mean([r['finite_soft_return'] for r in rows])),
                    bootstrapped_mean=float(np.mean([r['bootstrapped_soft_return'] for r in rows])),
                    mission_rate=float(np.mean([r['mission'] for r in rows]))))
            record=dict(state=selected,seed=args.seed_base+attempt,prefix_steps=prefix_length,actions=aggregate)
            records.append(record);write_json(out/'results.json',records);print(json.dumps(record),flush=True)
            selected+=1;env.close()
        write_json(out/'status.json',dict(status='complete' if selected==states else 'insufficient_surviving_states',states=selected,actor_updates=0))
        (out/'DONE').write_text('Bounded critic diagnostic completed. No actor trained or promoted.\n')


if __name__=='__main__':main()
