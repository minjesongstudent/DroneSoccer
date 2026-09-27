"""Same frozen data, public100D one-step SAC versus soft Retrace(n8,lambda.95).

No collection, teacher, actor update or production launch. --run explicitly
starts a capped 5000-update-per-critic CUDA diagnostic after review/tests.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import traceback

sys.dont_write_bytecode = True
import numpy as np

VERSION = 'paired-public-soft-retrace-calibration-v1'
DEFAULT_DATA = Path('D:/DroneSoccerData/diagnostics/asymmetric-comparison-v1')
DEFAULT_OUTPUT = Path('D:/DroneSoccerData/diagnostics/retrace-comparison-v1')
UPDATES, CHUNK, N_STEPS, TRACE_LAMBDA = 5000, 250, 8, .95


def load_module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sequence_table(arrays,flights,n_steps=N_STEPS):
    """Global chronological rows, never flattened-replay neighbors across gaps."""
    size=len(arrays['rewards'])
    if size<=0 or n_steps<=0:
        raise ValueError('Nonempty data and positive trace length are required.')
    indices=np.zeros((size,n_steps),np.int64)
    valid=np.zeros((size,n_steps),bool)
    covered=np.zeros(size,bool)
    for flight in flights:
        start,end=flight['start'],flight['end']
        if not 0<=start<end<=size or covered[start:end].any():
            raise ValueError('Invalid or overlapping flight bounds.')
        if not np.all(arrays['flight_id'][start:end]==flight['flight_id']):
            raise ValueError('Flight identity differs from its stored bounds.')
        if not np.all(arrays['train'][start:end]==(flight['split']=='train')):
            raise ValueError('A flight crosses training/development splits.')
        term,trunc=arrays['terminated'][start:end],arrays['truncated'][start:end]
        if (term[:-1]|trunc[:-1]).any() or not (term[-1] or trunc[-1]):
            raise ValueError('Stored flight does not end exactly at its first boundary.')
        if bool(term[-1])!=flight['terminal'] or bool(trunc[-1])!=flight['truncated']:
            raise ValueError('Flight metadata and terminal flags disagree.')
        if not np.array_equal(arrays['next_observations'][start:end-1],arrays['observations'][start+1:end]):
            raise ValueError('Stored trajectory is not chronologically continuous.')
        steps=np.arange(start,end)[:,None]+np.arange(n_steps)
        indices[start:end]=np.minimum(steps,end-1)
        valid[start:end]=steps<end
        covered[start:end]=True
    if not covered.all():
        raise ValueError('Every stored row must belong to exactly one flight.')
    return indices,valid


def renamed_gate(value):
    """Reuse frozen decision arithmetic while naming the actual treatments."""
    def rename(text):
        return text.replace('asymmetric','soft_retrace').replace('public','one_step')
    if isinstance(value,dict):
        return {rename(key):renamed_gate(item) for key,item in value.items()}
    if isinstance(value,list):
        return [renamed_gate(item) for item in value]
    return rename(value) if isinstance(value,str) else value


def evaluate(models,arrays,flights,output,updates,common,progress):
    """Frozen evaluator definitions, both critics using public input only."""
    import torch
    results={}
    with torch.no_grad():
        for name,model in models.items():
            model.policy.set_training_mode(False)
            alpha=float(model.ent_coef_tensor.item())
            q_values=[]
            for begin in range(0,len(arrays['rewards']),512):
                chunk=slice(begin,begin+512)
                obs=torch.as_tensor(arrays['observations'][chunk],device='cuda')
                action=torch.as_tensor(arrays['actions'][chunk],device='cuda')
                q_values.append(torch.cat(model.critic(obs,action),1).cpu().numpy())
            q_values=np.concatenate(q_values)
            records=[]
            for flight in flights:
                first,end=flight['start'],flight['end']
                torch.manual_seed(9_600_000+flight['flight_id'])
                obs=torch.as_tensor(arrays['next_observations'][[first,end-1]],device='cuda').repeat_interleave(64,0)
                action,logp=model.actor.action_log_prob(obs)
                v=(torch.cat(model.critic_target(obs,action),1).min(1).values-alpha*logp).reshape(2,64).mean(1).cpu().numpy()
                finite=common.finite_soft_returns(arrays['rewards'][first:end],arrays['log_prob'][first:end],model.gamma,alpha)
                terminal=flight['terminal']
                tail=0. if terminal else model.gamma**(end-first)*float(v[1])
                q1,q2=map(float,q_values[first]); q=min(q1,q2)
                target=float(arrays['rewards'][first])+(0. if arrays['terminated'][first] else model.gamma*float(v[0]))
                records.append({**flight,'q1':q1,'q2':q2,'q_min':q,
                    'terminal_error':q-float(finite[0]) if terminal else None,
                    'terminal_q1_error':q1-float(finite[0]) if terminal else None,
                    'terminal_q2_error':q2-float(finite[0]) if terminal else None,
                    'realized_finite_soft_return':float(finite[0]),
                    'exact_complete_sampled_return':float(finite[0]) if terminal else None,
                    'own_target_bootstrap_tail':tail,'own_bootstrapped_return':float(finite[0])+tail,
                    'q_minus_own_bootstrapped_return':q-float(finite[0])-tail,
                    'one_step_td_residual':q-target})
            groups={}
            for split in ('train','development'):
                for phase in ('approach',*common.PHASES):
                    rows=[r for r in records if r['split']==split and r['source_phase']==phase]
                    if not rows:
                        continue
                    terminals=[r for r in rows if r['terminal']]
                    crashes=[r for r in terminals if r['crash']]
                    groups[split+'|'+phase]={
                        'episodes':len(rows),'terminal_episodes':len(terminals),'excluded_timeouts':len(rows)-len(terminals),
                        'crash_terminal_episodes':len(crashes),
                        'terminal_error':common.summary_values([r['terminal_error'] for r in terminals]),
                        'crash_terminal_error':common.summary_values([r['terminal_error'] for r in crashes]),
                        'own_bootstrap_residual':common.summary_values([r['q_minus_own_bootstrapped_return'] for r in rows]),
                        'one_step_td_residual':common.summary_values([r['one_step_td_residual'] for r in rows])}
            results[name]={'groups':groups,'initial_records':records}
            progress.update('evaluating_critics',updates=updates,critic=name)
    clusters=defaultdict(list)
    for before,after in zip(results['one_step']['initial_records'],results['soft_retrace']['initial_records']):
        if before['flight_id']!=after['flight_id']:
            raise AssertionError('Unpaired flight evaluation.')
        if before['split']=='development' and before['terminal'] and before['crash']:
            clusters[before['source_seed']].append(abs(before['terminal_error'])-abs(after['terminal_error']))
    values=np.array([np.mean(v) for v in clusters.values()])
    interval=None
    if len(values)>1:
        bootstrap=np.random.default_rng(9_800_027).choice(values,(2000,len(values)),replace=True).mean(1)
        interval=np.quantile(bootstrap,[.025,.975]).tolist()
    gate=renamed_gate(common.evidence_gate(results['one_step']['initial_records'],results['soft_retrace']['initial_records']))
    report={'version':VERSION,'updates_per_critic':updates,'models':results,'gate':gate,
        'paired_terminal_crash_comparison':{'positive_favors_soft_retrace':True,
            'distinct_source_flights':len(values),'source_cluster_mean_mae_reduction':float(values.mean()) if len(values) else None,
            'descriptive_95pct_source_cluster_bootstrap_interval':interval},
        'definitions':{'terminal_error':'min(Q1,Q2) minus complete sampled soft return, excluding queried action entropy.',
            'timeout_comparison':'Observed soft prefix plus own target-critic tail; never ground truth.',
            'actor_observation_dim':100,'critic_observation_dim':100,'privileged_context_used':False,
            'selection_limit':'Crash-terminal subset, shared source states, no independent 7M evaluation.'}}
    common.write_json(output/f'evaluation-{updates:05d}.json',report)
    return report


def run(args):
    import torch
    from torch.nn import functional as F
    from stable_baselines3.common.utils import polyak_update
    if not torch.cuda.is_available():
        raise RuntimeError('This bounded diagnostic requires the local CUDA GPU.')
    torch.set_num_threads(1)
    data_dir,output=args.data.resolve(),args.output.resolve()
    if output.drive.lower()!='d:' or output.exists():
        raise ValueError('Use a new, empty output directory on D:.')
    project=Path(__file__).resolve().parents[1]
    original_snapshot=data_dir/'source_snapshot'
    common=load_module('frozen_comparison_tools',original_snapshot/'diagnostics/asymmetric_critic_comparison.py')
    provenance=json.loads((data_dir/'provenance.json').read_text())
    for relative,expected in provenance['source_hashes'].items():
        if common.sha_file(original_snapshot/relative)!=expected:
            raise ValueError('Prior frozen source/checkpoint has changed.')
    immutable={str(data_dir/name):common.sha_file(data_dir/name) for name in
               ('dataset.npz','dataset-summary.json','flights.json','protocol.json',
                'replay-sample-indices.npy','pairing-verification.json')}
    if immutable[str(data_dir/'dataset.npz')]!=json.loads((data_dir/'dataset-summary.json').read_text())['sha256']:
        raise ValueError('Frozen dataset checksum mismatch.')
    output.mkdir(parents=True)
    snapshot=output/'source_snapshot'
    shutil.copytree(original_snapshot,snapshot)
    additions={project/'src/retrace_targets.py':snapshot/'src/retrace_targets.py',
               Path(__file__).resolve():snapshot/'diagnostics/retrace_critic_comparison.py',
               project/'tests/test_retrace_targets.py':snapshot/'tests/test_retrace_targets.py'}
    for source,destination in additions.items():
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,destination)
    source_hashes={str(path.relative_to(snapshot)):common.sha_file(path) for path in snapshot.rglob('*') if path.is_file()}
    common.write_json(output/'provenance.json',{'version':VERSION,'frozen_dataset':str(data_dir),
        'immutable_inputs_sha256':immutable,'source_sha256':source_hashes,
        'base_source_hashes':provenance['source_hashes'],'new_files':list(map(str,additions.values()))})
    if any(name=='src' or name.startswith('src.') for name in sys.modules):
        raise RuntimeError('Use a fresh process for frozen project imports.')
    sys.path.insert(0,str(snapshot))
    from src.anchored_policy import AnchoredSAC
    from src.retrace_targets import soft_retrace_targets,VERSION as TARGET_VERSION
    from src.curriculum import LessonEnv
    from src.mission_observation import MissionObservation
    from stable_baselines3.common.logger import configure
    progress=common.Progress(output)
    progress.update('initializing',treatment='public100D soft Retrace',device=torch.cuda.get_device_name(0))
    try:
        # Deliberately do not load the stored privileged context arrays.
        with np.load(data_dir/'dataset.npz',allow_pickle=False) as saved:
            arrays={key:saved[key] for key in ('observations','next_observations','actions','rewards','log_prob',
                                              'terminated','truncated','flight_id','train')}
        if arrays['observations'].shape[1:]!=(100,) or any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Dataset must contain finite public100D trajectories.')
        flights=json.loads((data_dir/'flights.json').read_text())
        if any(7_000_000<=flight['source_seed']<8_000_000 for flight in flights):
            raise ValueError('Independent final-test data must not be used.')
        sequence,valid=sequence_table(arrays,flights)
        train=np.flatnonzero(arrays['train'])
        schedule=np.load(data_dir/'replay-sample-indices.npy',allow_pickle=False)
        source=AnchoredSAC.load(snapshot/'diagnostic_actor_20000.zip',device='cuda')
        source.policy.set_training_mode(False)
        source.actor.requires_grad_(False); source.reference_actor.requires_grad_(False)
        source_before=common.frozen_fingerprint(source)
        previous_protocol=json.loads((data_dir/'protocol.json').read_text())
        if source_before!=previous_protocol['source_fingerprint']:
            raise ValueError('The target actor must exactly match stored behavior; no inverse-tanh ratios are used.')
        if schedule.shape!=(UPDATES,source.batch_size) or (schedule<0).any() or (schedule>=len(train)).any():
            raise ValueError('Frozen replay start schedule has incompatible dimensions/indices.')
        if not np.all((arrays['observations'][train[schedule],99]>.5).sum(1)==source.batch_size//2):
            raise ValueError('Anchor schedule must retain 50/50 single-pass/lap balance.')
        models={}
        for name in ('one_step','soft_retrace'):
            model=AnchoredSAC('MlpPolicy',MissionObservation(LessonEnv(0),single_pass=True),
                device='cuda',seed=common.INITIALIZATION_SEED,learning_rate=.0003,
                buffer_size=8,learning_starts=0,batch_size=source.batch_size,tau=source.tau,gamma=source.gamma,
                ent_coef=float(source.ent_coef_tensor.item()),target_update_interval=source.target_update_interval,
                policy_kwargs={'net_arch':source.actor.net_arch,'activation_fn':source.actor.activation_fn})
            model.actor.load_state_dict(source.actor.state_dict())
            model.reference_actor.load_state_dict(source.reference_actor.state_dict())
            model.actor.requires_grad_(False); model.reference_actor.requires_grad_(False)
            model.policy.set_training_mode(False)
            model.set_logger(configure(str(output/name),['csv']))
            models[name]=model
        if common.stable_hash(models['one_step'].critic.state_dict())!=common.stable_hash(models['soft_retrace'].critic.state_dict()):
            raise AssertionError('Fresh public critics must start byte-identical.')
        frozen={name:common.frozen_fingerprint(model) for name,model in models.items()}
        protocol={'version':VERSION,'target_version':TARGET_VERSION,'gate_predeclared':common.GATE,
            'baseline':'public100D one-step SAC','treatment':'public100D sampled soft Bellman residual Retrace',
            'n_steps':N_STEPS,'lambda':TRACE_LAMBDA,'updates_each':UPDATES,
            'actor_updates':0,'new_trajectories':0,'privileged_context_used':False,
            'actor_observation_dim':100,'critic_observation_dim':100,
            'gamma':source.gamma,'alpha':float(source.ent_coef_tensor.item()),'tau':source.tau,
            'batch_size':source.batch_size,'learning_rate':.0003,'source_fingerprint':source_before,
            'dataset_sha256':immutable[str(data_dir/'dataset.npz')],
            'importance_ratio':'Exactly1 because source actor hash equals stored behavior actor; c_i=.95.',
            'soft_target':'B0 + sum_k gamma^k*(product_i=1..k c_i)*(r_k+gamma*(1-terminal_k)*V_next_k-B_k)',
            'target_baseline':'Same scalar minimum of target twin critics in leading term, subtraction and next soft value.',
            'entropy':'No entropy of queried action; V_next includes entropy from the next policy action.',
            'boundaries':'Trace stops at terminal or timeout; timeout bootstraps actual stored final next observation.',
            'pairing':'Same frozen start schedule, same fresh critics, identical first successor actor samples; later Retrace samples use an isolated RNG stream.',
            'same_data_limit':previous_protocol['horizon'],
            'approval':'No automatic production promotion; failed20k actor is diagnostic only.',
            'theory_limit':'Sampled expected-soft-Bellman residual adaptation; no convergence guarantee for aliased100D neural twin-min critics.'}
        common.write_json(output/'protocol.json',protocol)
        evaluate(models,arrays,flights,output,0,common,progress)
        tensors={key:torch.as_tensor(value,device='cuda') for key,value in arrays.items() if key not in ('train','flight_id')}
        original_pairing=json.loads((data_dir/'pairing-verification.json').read_text())
        pairing=[]
        for begin in range(0,UPDATES,CHUNK):
            torch.manual_seed(9_400_027+begin)
            sample_hash=hashlib.sha256()
            for update in range(begin,begin+CHUNK):
                anchors=train[schedule[update]]
                ids=sequence[anchors]
                mask=valid[anchors]
                if not arrays['train'][ids[mask]].all() or not np.all(arrays['flight_id'][ids]==arrays['flight_id'][anchors,None]):
                    raise AssertionError('Sequence escaped its original training flight.')
                ix=torch.as_tensor(ids,device='cuda')
                mask_t=torch.as_tensor(mask,device='cuda')
                obs=tensors['observations'][ix]
                next_obs=tensors['next_observations'][ix]
                actions=tensors['actions'][ix]
                reward=tensors['rewards'][ix]
                terminated=tensors['terminated'][ix]
                truncated=tensors['truncated'][ix]
                behavior=tensors['log_prob'][ix]
                with torch.no_grad():
                    first_action,first_logp=source.actor.action_log_prob(next_obs[:,0].contiguous())
                    sample_hash.update(first_action.cpu().numpy().tobytes())
                    sample_hash.update(first_logp.cpu().numpy().tobytes())
                    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                        torch.manual_seed(9_700_027+update)
                        later_action,later_logp=source.actor.action_log_prob(next_obs[:,1:].reshape(-1,100))
                    next_actions=torch.cat((first_action[:,None],later_action.reshape(-1,N_STEPS-1,4)),1)
                    next_logp=torch.cat((first_logp[:,None],later_logp.reshape(-1,N_STEPS-1)),1)
                for name,model in models.items():
                    with torch.no_grad():
                        if name=='one_step':
                            nq=torch.cat(model.critic_target(next_obs[:,0],first_action),1).min(1).values
                            target=reward[:,0]+source.gamma*(~terminated[:,0])*(nq-model.ent_coef_tensor*first_logp)
                        else:
                            baseline=torch.cat(model.critic_target(obs.reshape(-1,100),actions.reshape(-1,4)),1).min(1).values.reshape(-1,N_STEPS)
                            nq=torch.cat(model.critic_target(next_obs.reshape(-1,100),next_actions.reshape(-1,4)),1).min(1).values.reshape(-1,N_STEPS)
                            values=nq-model.ent_coef_tensor*next_logp
                            target=soft_retrace_targets(reward,terminated,truncated,mask_t,baseline,values,
                                behavior,behavior,gamma=source.gamma,trace_lambda=TRACE_LAMBDA)
                    predicted=model.critic(obs[:,0],actions[:,0])
                    loss=.5*sum(F.mse_loss(q,target[:,None]) for q in predicted)
                    model.critic.optimizer.zero_grad(); loss.backward(); model.critic.optimizer.step()
                    if update%model.target_update_interval==0:
                        polyak_update(model.critic.parameters(),model.critic_target.parameters(),model.tau)
                        polyak_update(model.batch_norm_stats,model.batch_norm_stats_target,1.)
                    model._n_updates+=1; model.critic_calibration_updates+=1
                    model.logger.record('train/critic_loss',float(loss.item()))
            reference=original_pairing[begin//CHUNK]['next_action_and_logprob_sha256']
            if sample_hash.hexdigest()!=reference:
                raise AssertionError('One-step actor RNG differs from the frozen original comparison.')
            for name,model in models.items():
                if common.frozen_fingerprint(model)!=frozen[name]:
                    raise AssertionError('Actor/reference/actor optimizer or counters changed.')
                model.logger.dump(begin+CHUNK)
            pairing.append({'first_update':begin+1,'last_update':begin+CHUNK,
                'next_action_and_logprob_sha256':sample_hash.hexdigest(),'matches_original_one_step_rng':True,
                'start_indices_sha256':common.stable_hash(schedule[begin:begin+CHUNK]),'actors_unchanged':True})
            common.write_json(output/'pairing-verification.json',pairing)
            progress.update('calibrating',updates_per_critic=begin+CHUNK,update_cap=UPDATES,treatment='soft_retrace')
            if begin+CHUNK in (1000,2500,UPDATES):
                final=evaluate(models,arrays,flights,output,begin+CHUNK,common,progress)
        for name,model in models.items():
            model.save(output/name/f'critic-calibrated-{UPDATES}.zip')
        if common.frozen_fingerprint(source)!=source_before:
            raise AssertionError('Source actor changed.')
        for path,expected in immutable.items():
            if common.sha_file(path)!=expected:
                raise AssertionError('Frozen input data changed.')
        for relative,expected in source_hashes.items():
            if common.sha_file(snapshot/relative)!=expected:
                raise AssertionError('New frozen source changed.')
        common.write_json(output/'frozen-actor-verification.json',frozen)
        progress.update('complete',updates_per_critic=UPDATES,actor_updates=0,
            supports_bounded_diagnostic_behavioral_trial=final['gate']['supports_bounded_diagnostic_behavioral_trial'],
            gate_reasons=final['gate']['reasons'])
    except BaseException as error:
        common.write_json(output/'error.json',{'type':type(error).__name__,'message':str(error),'traceback':traceback.format_exc()})
        progress.update('failed',error=str(error))
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--data',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    if not args.run:
        parser.error('Only an explicitly reviewed --run starts this bounded diagnostic.')
    run(args)


if __name__=='__main__':
    main()
