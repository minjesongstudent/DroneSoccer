"""Bounded on-policy clipped actor updates from complete learned-policy flights."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import pickle
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

from src.clearance_training import ClearanceTrainingEnv,PROFILE
from src.constrained_sac import ConstrainedSAC,episode_partition,gaussian_kl,boundary_penalty,guarded_optimizer_step
from src.sim import HoopEnv
from tools.run_clearance_experiment import ROOT,sha,write_json,evaluate,retention_failures,single_trainer
from tools.validate_flight_policies import SUITES


def counted_development_episodes(path,seed_base,count=100):
    """Recount saved episode outcomes, rejecting missing/duplicated layouts."""
    rows=[json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(rows)!=count or sorted(row['seed'] for row in rows)!=list(range(seed_base,seed_base+count)):
        raise ValueError(f'{path}: expected exactly {count} unique fixed-suite seeds')
    for row in rows:
        for key in ('crashed','clean_first_pass','clean_repeated_episode'):
            if type(row[key]) is not bool:raise ValueError(f'{path}: nonboolean {key}')
        for key in ('steps','wall_steps','full_body_passes','returns_to_front'):
            if type(row[key]) is not int or row[key]<0:raise ValueError(f'{path}: invalid {key}')
        if not 1<=row['steps']<=600:raise ValueError(f'{path}: invalid mission length')
        if row['crashed']!=(row['outcome'] in ('hit','wrong_way','net')):
            raise ValueError(f'{path}: inconsistent crash outcome')
        repeated=(row['full_body_passes']>=2 and row['steps']==600
                  and not row['crashed'] and row['wall_steps']==0)
        if row['clean_repeated_episode']!=repeated:
            raise ValueError(f'{path}: inconsistent clean-repeat outcome')
        if row['clean_first_pass'] and row['full_body_passes']==0:
            raise ValueError(f'{path}: first passage without a full-body passage')
    return dict(episodes=count,first_pass=sum(row['clean_first_pass'] for row in rows),
                clean_repeat=sum(row['clean_repeated_episode'] for row in rows),
                crashes=sum(row['crashed'] for row in rows))


def prepare_gae_screen_reference(reference_run,destination,source):
    """Freeze and verify the actual original-policy evidence before training."""
    reference_run=Path(reference_run);destination=Path(destination)
    report=json.loads((reference_run/'results.json').read_text())
    protocol=json.loads((reference_run/'protocol.json').read_text())
    source_hash=sha(source)
    if (source_hash!=PROFILE.sha256 or protocol['episodes']!=100 or protocol['max_steps']!=600
            or protocol['suites']!=SUITES
            or protocol['profiles'][PROFILE.name]['sha256']!=source_hash
            or tuple(protocol['profiles'][PROFILE.name]['lags'])!=tuple(PROFILE.lags)):
        raise ValueError('Baseline protocol, source policy or suite mismatch')
    metrics={};inputs=[Path('results.json'),Path('protocol.json')]
    for suite,seed in SUITES.items():
        relative=Path(PROFILE.name)/suite/'episodes.jsonl'
        counted=counted_development_episodes(reference_run/relative,seed)
        old=report[f'{PROFILE.name}/{suite}']
        declared=dict(episodes=old['episodes'],first_pass=old['clean_first_pass'],
                      clean_repeat=old['clean_repeated_episodes'],crashes=old['crashes'])
        if counted!=declared:raise ValueError(f'Baseline {suite} report disagrees with episodes')
        metrics[suite]=counted;inputs.append(relative)
    destination.mkdir(parents=True,exist_ok=False)
    evidence={}
    for relative in inputs:
        original=reference_run/relative;snapshot=destination/relative
        snapshot.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(original,snapshot)
        digest=sha(snapshot)
        if digest!=sha(original):raise ValueError('Baseline changed while snapshotting')
        evidence[relative.as_posix()]=dict(original_path=str(original.resolve()),
                                          path=str(snapshot.resolve()),sha256=digest)
    frozen=dict(profile=PROFILE.name,source_sha256=source_hash,metrics=metrics,evidence=evidence)
    write_json(destination/'reference.json',frozen)
    return dict(path=str((destination/'reference.json').resolve()),
                sha256=sha(destination/'reference.json'))


def gae_screen_reasons(reference,candidate):
    """Fixed development screen only; never mission acceptance or promotion."""
    reasons=[]
    for suite in SUITES:
        old,new=reference[suite],candidate[suite]
        if old['episodes']!=100 or new['episodes']!=100:
            reasons.append(f'{suite}: screen requires exactly 100 episodes per policy')
        if new['first_pass']<old['first_pass']-2:reasons.append(f'{suite}: first-pass drop exceeds 2')
        if new['clean_repeat']<old['clean_repeat']-2:reasons.append(f'{suite}: repeat drop exceeds 2')
        if new['crashes']>old['crashes']+2:reasons.append(f'{suite}: crash rise exceeds 2')
    if candidate['randomised']['clean_repeat']-reference['randomised']['clean_repeat']<5:
        reasons.append('randomised: clean-repeat gain is below 5/100')
    if reference['randomised']['crashes']-candidate['randomised']['crashes']<5:
        reasons.append('randomised: crash reduction is below 5/100')
    return reasons


def write_gae_final_screen(out,result,checkpoint,source,reference_record,smoke=False):
    out=Path(out);checkpoint=Path(checkpoint);source=Path(source)
    reasons=[];reference={};candidate={};evidence={}
    if smoke:reasons.append('Software-only smoke: never eligible for the development screen')
    if result['status']=='retention_regression' or any(r.get('failures') for r in result['iterations']):
        reasons.append('Training hit a retention-regression stop')
    if result['status'] not in ('bounded_trial_complete','bounded_stop'):
        reasons.append('Training did not reach a valid bounded completion')
    if not result.get('source_unchanged') or sha(source)!=PROFILE.sha256:
        reasons.append('Original source policy changed')
    for label,path in [('source_policy',source),('final_candidate',checkpoint),
                       ('run_result',out/'result.json'),('run_protocol',out/'protocol.json'),
                       ('final_summary',out/'final_development/summary.json'),
                       ('final_model_identity',out/'final_development/model.json')]:
        evidence[label]=dict(path=str(path.resolve()),sha256=sha(path))
    try:
        if sha(reference_record['path'])!=reference_record['sha256']:
            raise ValueError('Frozen baseline manifest changed')
        frozen=json.loads(Path(reference_record['path']).read_text())
        evidence['baseline_manifest']=reference_record
        for label,item in frozen['evidence'].items():
            if sha(item['path'])!=item['sha256']:raise ValueError(f'Frozen baseline {label} changed')
            evidence['baseline/'+label]=item
        if frozen['source_sha256']!=PROFILE.sha256:raise ValueError('Frozen baseline source mismatch')
        reference=frozen['metrics']
        identity=json.loads((out/'final_development/model.json').read_text())
        if identity['sha256']!=sha(checkpoint):raise ValueError('Final evaluation used another candidate')
        if not result['iterations'] or checkpoint.name!=f"policy_{result['iterations'][-1]['iteration']:02d}.zip":
            raise ValueError('Screen requires the final saved checkpoint')
        final=json.loads((out/'final_development/summary.json').read_text())
        if final!=result['final_development']:raise ValueError('Final summary differs from run result')
        for suite,seed in SUITES.items():
            path=out/'final_development'/suite/'episodes.jsonl'
            evidence['final_episodes/'+suite]=dict(path=str(path.resolve()),sha256=sha(path))
            counted=counted_development_episodes(path,seed,2 if smoke else 100)
            if any(final[suite][key]!=value for key,value in counted.items()):
                raise ValueError(f'Final {suite} summary disagrees with saved episodes')
            candidate[suite]=counted
        if not smoke:reasons.extend(gae_screen_reasons(reference,candidate))
    except (ValueError,KeyError,TypeError,OSError) as exc:
        reasons.append(f'Evidence validation failed: {exc}')
    passed=not reasons
    decision=dict(kind='Final fixed development go/no-go; not acceptance',
        decision='software_only' if smoke else 'eligible_for_one_fresh_development_screen' if passed else 'reject_close_variant',
        passed=passed,reasons=reasons,reference=reference,candidate=candidate,evidence=evidence,
        acceptance=False,policy_promoted=False,physical_flight=False)
    write_json(out/'final_screen_decision.json',decision)
    return decision


def discounted_returns(rewards,gamma):
    result=np.empty(len(rewards),np.float32);value=0.
    for i in range(len(rewards)-1,-1,-1):
        value=float(rewards[i])+gamma*value;result[i]=value
    return result


def finite_horizon_gae(rewards,values,episode_lengths,gamma=.995,lam=.95):
    """GAE over complete episodes, with zero tails at crash AND 600-step cutoff.

    Values are frozen predictions from the existing MC-fitted baseline. No
    bootstrap crosses a reset or extends the declared finite mission horizon.
    """
    rewards=np.asarray(rewards,dtype=np.float64)
    values=np.asarray(values,dtype=np.float64)
    lengths=np.asarray(episode_lengths)
    if (rewards.ndim!=1 or values.shape!=rewards.shape or lengths.ndim!=1
            or not np.issubdtype(lengths.dtype,np.integer) or len(lengths)==0
            or np.any(lengths<=0) or int(lengths.sum())!=len(rewards)):
        raise ValueError('Complete positive episode lengths must partition rewards and values')
    if (not np.isfinite(rewards).all() or not np.isfinite(values).all()
            or not 0<=gamma<=1 or not 0<=lam<=1):
        raise ValueError('Finite rewards/values and gamma/lambda in [0,1] required')
    advantages=np.empty(len(rewards),np.float32)
    start=0
    for length in lengths:
        end=start+int(length)
        carry=next_value=0.0
        for index in range(end-1,start-1,-1):
            delta=rewards[index]+gamma*next_value-values[index]
            carry=delta+gamma*lam*carry
            advantages[index]=carry
            next_value=values[index]
        start=end
    return advantages


def mission_returns(length,passes,terminated,wall_steps):
    success=length==600 and passes>=2 and not terminated and wall_steps==0
    return np.full(length,float(success),dtype=np.float32)


def grouped_advantages(returns,lengths,group_size):
    starts=np.r_[0,np.cumsum(lengths)];outcomes=np.asarray(returns)[starts[:-1]]
    per_episode=np.zeros(len(lengths),np.float32);mixed=0
    for start in range(0,len(lengths),group_size):
        group=outcomes[start:start+group_size]
        if len(group)<2:continue
        mixed+=int(group.min()!=group.max())
        per_episode[start:start+len(group)]=group-(group.sum()-group)/(len(group)-1)
    return np.repeat(per_episode,lengths),mixed


def action_log_prob(actor,obs,actions):
    mean,std,kwargs=actor.get_action_dist_params(obs)
    return actor.action_dist.proba_distribution(mean,std,**kwargs).log_prob(actions.clamp(-1+1e-6,1-1e-6))


def clipped_loss(log_prob,old_log_prob,advantage):
    ratio=(log_prob-old_log_prob).exp()
    return -torch.minimum(ratio*advantage,ratio.clamp(.9,1.1)*advantage).mean()


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--actor-lr',type=float,choices=[1e-6,1e-5],default=1e-6)
    parser.add_argument('--objective',choices=['discounted','mission'],default='discounted')
    parser.add_argument('--episodes-per-iteration',type=int,choices=[16,48],default=16)
    parser.add_argument('--actor-epochs',type=int,choices=[1,3],default=3)
    parser.add_argument('--group-size',type=int,choices=[1,4],default=1)
    parser.add_argument('--approach-curriculum',action='store_true')
    parser.add_argument('--backtracking',action='store_true')
    parser.add_argument('--advantage-estimator',choices=['monte-carlo','gae95'],default='monte-carlo')
    args=parser.parse_args(argv)
    if args.group_size>1 and args.objective!='mission':parser.error('Grouped baseline requires mission objective')
    if args.approach_curriculum and args.group_size!=1:parser.error('Curriculum requires independent complete episodes')
    if args.advantage_estimator=='gae95' and not (
            args.approach_curriculum and args.backtracking and args.objective=='discounted'
            and args.group_size==1 and args.actor_lr==1e-5
            and args.episodes_per_iteration==48 and args.actor_epochs==1):
        parser.error('GAE95 requires the paired approach-curriculum/backtracking configuration: '
                     'discounted objective, 48 independent episodes, actor lr 1e-5, one epoch')
    return args


def main():
    args=parse_args();out=args.out.resolve()
    reset_base=(11241000 if args.advantage_estimator=='gae95' else
                11240000 if args.approach_curriculum else 11150000)
    assert torch.cuda.is_available();torch.set_num_threads(1)
    np.random.seed(11540000);torch.manual_seed(11540000)
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        source=ROOT/'trainedModels/seenflag-s1-level1.zip';assert sha(source)==PROFILE.sha256
        parent=Path('D:/DroneSoccerData/runs/clearance-reward-20261007')
        replay_path=parent/'clearance/step_200000/replay_buffer.pkl'
        with replay_path.open('rb') as stream:replay=pickle.load(stream)
        fit,_,_=episode_partition(replay.dones[:replay.pos])
        ids=np.random.default_rng(11520000).choice(fit,8192 if args.approach_curriculum else 1024,replace=False)
        model=ConstrainedSAC.load(source,device='cuda')
        model.set_random_seed(11540000)
        model.configure(replay.observations[ids,0],boundary_weight=.1,actor_lr=args.actor_lr)
        del replay
        value=nn.Sequential(nn.Linear(85,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1)).cuda()
        value_optimizer=torch.optim.Adam(value.parameters(),lr=3e-4)
        assert all(parameter.device.type=='cuda' for parameter in model.actor.parameters())
        assert all(parameter.device.type=='cuda' for parameter in value.parameters())
        hashes={}
        for file in list((ROOT/'src').glob('*.py'))+[Path(__file__),ROOT/'tools/run_clearance_experiment.py',ROOT/'tools/validate_flight_policies.py',ROOT/'tests/test_onpolicy_trial.py',source]:
            rel=file.relative_to(ROOT);dest=out/'source'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(file,dest);hashes[str(rel)]=sha(dest)
        protocol=dict(kind='on-policy finite-horizon clipped actor trial',sources=hashes,
            iterations=1 if args.smoke else 5,episodes_per_iteration=(16 if args.group_size>1 else 2) if args.smoke else args.episodes_per_iteration,
            group_size=args.group_size,
            reset_seed_base=reset_base,algorithm_seed=11540000,max_episode_steps=600,
            approach_curriculum=args.approach_curriculum,
            max_backtracks=4 if args.backtracking else 0,
            curriculum='Even FIT seeds: front45-65cm, lateral+-18cm,vertical+-6cm,heading+-15deg; odd seeds unchanged full-task resets. All episodes continue to native crash/600step cutoff.',
            actor_lr=args.actor_lr,value_lr=3e-4,clip_ratio=.1,source_mean_kl=.01,source_max_kl=.1,
            source_boundary_weight=.1,actor_epochs=1 if args.smoke else args.actor_epochs,
            value_batches=2 if args.smoke else 100,wall_seconds=180 if args.smoke else 3600,
            objective=args.objective,
            advantage_estimator=args.advantage_estimator,
            returns=('Binary exact clean-repeat600-step survival outcome, undiscounted score-function target'
                     if args.objective=='mission' else 'Complete finite600-step discounted reward-to-go, gamma .995, no timeout bootstrap'),
            baseline=('Leave-one-attempt-out outcome mean within same reset layout; no fitted value'
                      if args.group_size>1 else 'Separate fitted state value; detached advantages; SAC Q gradients unused'),
            rewards='Existing full-clearance training reward',actions='Stochastic learned actor only',
            acceptance=False,physical_flight=False,guard_replay_sha256=sha(replay_path),
            runtime=dict(python=sys.version,packages={name:importlib.metadata.version(name)
                         for name in ('torch','numpy','stable-baselines3','gymnasium')},
                         torch_cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(model.device),
                         model_device=str(model.device)))
        if args.advantage_estimator=='gae95':
            reference_record=prepare_gae_screen_reference(
                ROOT.parent/'lab-data/policy-validation-20261006',out/'screen_reference',source)
            protocol.update(gae_gamma=.995,gae_lambda=.95,
                final_screen_reference=reference_record,
                gae_baseline='Unchanged MC-fitted value; frozen before GAE; zero tails at both crash and 600-step cutoff',
                final_screen=dict(final_checkpoint_only=True,episodes_per_suite=100,
                    randomized_min_repeat_gain=5,randomized_min_crash_reduction=5,
                    any_suite_max_first_pass_drop=2,any_suite_max_repeat_drop=2,
                    any_suite_max_crash_increase=2,
                    reference='Original 85-input policy on unchanged fixed development suites',
                    on_fail='Close variant; no lambda sweep, unchanged extension, or fresh cohort',
                    on_pass='One independent paired development screen; not simulation acceptance'))
        write_json(out/'protocol.json',protocol);print('PROTOCOL',str(out),flush=True)
        if args.approach_curriculum:
            from src.approach_curriculum import ApproachCurriculumEnv
            simulator=ApproachCurriculumEnv(level=1,randomise=True,hanger=True)
        else:simulator=HoopEnv(level=1,randomise=True,hanger=True)
        env=ClearanceTrainingEnv(simulator,'clearance',seed_base=reset_base,trace_dir=out/'trajectories')
        baseline=json.loads((parent/'reference_guard/summary.json').read_text())
        deadline=time.monotonic()+protocol['wall_seconds'];result=dict(status='running',iterations=[],acceptance=False)
        accepted=rejected=consecutive=0
        for iteration in range(1,protocol['iterations']+1):
            observations=[];actions=[];returns=[];episode_lengths=[];rollout_rewards=[]
            for episode_index in range(protocol['episodes_per_iteration']):
                if args.group_size>1:
                    seed=11150000+(iteration-1)*(protocol['episodes_per_iteration']//args.group_size)+episode_index//args.group_size
                    obs,_=env.reset(seed=seed)
                else:obs,_=env.reset()
                rewards=[]
                while True:
                    action,_=model.predict(obs,deterministic=False)
                    observations.append(obs.copy());actions.append(action.copy())
                    obs,reward,done,cut,_=env.step(action);rewards.append(reward)
                    if done or cut:break
                values=(mission_returns(len(rewards),env.audit.count,done,env.unwrapped.wall_steps)
                        if args.objective=='mission' else discounted_returns(rewards,.995))
                returns.extend(values);episode_lengths.append(len(rewards))
                rollout_rewards.extend(rewards)
                if time.monotonic()>deadline or (out/'STOP').exists():break
            x=torch.tensor(np.asarray(observations),device='cuda');a=torch.tensor(np.asarray(actions),device='cuda')
            target=torch.tensor(np.asarray(returns),device='cuda')
            with torch.no_grad():old=action_log_prob(model.actor,x,a).detach()
            np.savez_compressed(out/f'rollout_{iteration:02d}.npz',observations=x.cpu().numpy(),actions=a.cpu().numpy(),
                                returns=target.cpu().numpy(),old_log_prob=old.cpu().numpy(),episode_lengths=episode_lengths)
            # Value only estimates a baseline. Its action derivatives never enter the actor loss.
            for _ in range(0 if args.group_size>1 else protocol['value_batches']):
                batch=np.random.randint(0,len(x),min(256,len(x)))
                loss=(value(x[batch]).flatten()-target[batch]).square().mean()
                value_optimizer.zero_grad();loss.backward();nn.utils.clip_grad_norm_(value.parameters(),10);value_optimizer.step()
            with torch.no_grad():
                mixed=None
                if args.group_size>1:
                    advantages,mixed=grouped_advantages(returns,episode_lengths,args.group_size)
                    adv=torch.tensor(advantages,device='cuda')
                else:
                    frozen_values=value(x).flatten()
                    if args.advantage_estimator=='gae95':
                        raw_adv=finite_horizon_gae(rollout_rewards,frozen_values.cpu().numpy(),episode_lengths)
                        np.savez_compressed(out/f'gae_inputs_{iteration:02d}.npz',
                            rewards=rollout_rewards,values=frozen_values.cpu().numpy(),
                            episode_lengths=episode_lengths,raw_advantages=raw_adv)
                        adv=torch.tensor(raw_adv,device='cuda')
                    else:adv=target-frozen_values
                    adv=(adv-adv.mean())/(adv.std(unbiased=False)+1e-8)
            np.save(out/f'advantages_{iteration:02d}.npy',adv.cpu().numpy())
            stop=mixed==0
            for epoch in range(0 if stop else protocol['actor_epochs']):
                for batch in np.array_split(np.random.permutation(len(x)),max(1,int(np.ceil(len(x)/256)))):
                    log=action_log_prob(model.actor,x[batch],a[batch])
                    mean,std,_=model.actor.get_action_dist_params(x[batch])
                    with torch.no_grad():rm,rs,_=model.reference_actor.get_action_dist_params(x[batch])
                    gm,gs,_=model.actor.get_action_dist_params(model.trust_observations)
                    loss=clipped_loss(log,old[batch],adv[batch])+gaussian_kl(rm,rs,mean,std).mean()+.1*boundary_penalty(
                        gaussian_kl(*model.trust_distribution,gm,gs),.01,.1)
                    assert torch.isfinite(loss)
                    ok,drift=guarded_optimizer_step(model.actor,model.actor.optimizer,loss,model.trust_check,
                                                    max_backtracks=protocol['max_backtracks'])
                    accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
                    if consecutive>=100 or time.monotonic()>deadline or (out/'STOP').exists():stop=True;break
                if stop:break
            checkpoint=out/f'policy_{iteration:02d}.zip';model.save(checkpoint)
            allowed,drift=model.trust_check();assert allowed
            with torch.no_grad():
                ratio=(action_log_prob(model.actor,x,a)-old).exp()
                explained=1-(target-value(x).flatten()).var(unbiased=False)/(target.var(unbiased=False)+1e-8)
                diagnostics=dict(**drift,mean_ratio=float(ratio.mean()),
                    clip_fraction=float(((ratio<.9)|(ratio>1.1)).float().mean()),
                    fitted_value_explained_variance=None if args.group_size>1 else float(explained),mixed_groups=mixed,return_mean=float(target.mean()),
                    return_std=float(target.std(unbiased=False)))
            torch.save(dict(value=value.state_dict(),optimizer=value_optimizer.state_dict(),torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state()),out/f'state_{iteration:02d}.pt')
            metrics=evaluate(checkpoint,out/f'guard_{iteration:02d}',2 if args.smoke else 25)
            failures=[] if args.smoke else retention_failures(baseline,metrics)
            record=dict(iteration=iteration,transitions=len(x),episodes=len(episode_lengths),accepted=accepted,rejected=rejected,
                        metrics=metrics,failures=failures,diagnostics=diagnostics)
            result['iterations'].append(record);write_json(out/'result.json',result);print(json.dumps(record),flush=True)
            if failures or stop:
                result['status']='retention_regression' if failures else 'bounded_stop';break
        else:result['status']='bounded_trial_complete'
        env.close();result['final_development']=evaluate(checkpoint,out/'final_development',2 if args.smoke else 100)
        result['source_unchanged']=sha(source)==PROFILE.sha256
        if args.advantage_estimator=='gae95':result['final_screen_decision_file']='final_screen_decision.json'
        write_json(out/'result.json',result)
        if args.advantage_estimator=='gae95':
            decision=write_gae_final_screen(out,result,checkpoint,source,reference_record,smoke=args.smoke)
            print('FINAL_SCREEN',json.dumps(decision),flush=True)


if __name__=='__main__':main()
