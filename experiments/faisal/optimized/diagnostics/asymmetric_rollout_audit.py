"""Read-only asymmetric-critic calibration on authentic return snapshots.

Example (no learning):
  python diagnostics/asymmetric_rollout_audit.py --run D:/DroneSoccerData/runs/NAME \
      --checkpoint checkpoints/model_10000_steps.zip --device cuda

The default task is the exact 200-step ReturnPracticeEnv used for practice.
``--task full-flight`` instead continues the original restored physical flight
to its original cap, using that task's different critic clock/context. Neither
mode is an independent full-mission test. Actor input always remains 100D.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib
import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_fingerprint(model):
    hash_value = hashlib.sha256()
    for prefix, module in [('policy', model.policy), ('reference', model.reference_actor)]:
        for name, value in module.state_dict().items():
            hash_value.update((prefix+'.'+name).encode())
            hash_value.update(value.detach().cpu().contiguous().numpy().tobytes())
    counters = {name: getattr(model, name) for name in
                ('num_timesteps', '_n_updates', 'actor_attempts', 'actor_updates',
                 'anchor_rejected', 'critic_calibration_updates')}
    return {'parameters_sha256':hash_value.hexdigest(), 'counters':counters,
            'entropy_coefficient':float(model.ent_coef_tensor.item())}


def stochastic_seed(base, bank_hash, snapshot_id, repetition, deterministic):
    # Deterministic derivation does not depend on Python's randomized hash().
    key = f'{base}|{bank_hash}|{snapshot_id}|{repetition}|{int(deterministic)}'
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % (2**31)


def describe(values):
    array = np.asarray(values, dtype=np.float64)
    if not array.size:
        return {'count':0, 'mean':None, 'median':None, 'min':None, 'max':None}
    return {'count':len(array), 'mean':float(array.mean()), 'median':float(np.median(array)),
            'min':float(array.min()), 'max':float(array.max())}


def summaries(records):
    result = {}
    for split in sorted({r['split'] for r in records}):
        for deterministic in (False, True):
            for phase in ('all', 'return_side', 'return_front'):
                rows = [r for r in records if r['split']==split and r['deterministic']==deterministic
                        and (phase=='all' or r['source_phase']==phase)]
                if not rows:
                    continue
                terminal = [r for r in rows if r['crashed'] and r['terminated']]
                key = '|'.join((split, 'deterministic' if deterministic else 'stochastic', phase))
                result[key] = {
                    'episodes':len(rows),
                    'unique_snapshots':len({r['snapshot_id'] for r in rows}),
                    'unique_source_flights':len({r['source_seed'] for r in rows}),
                    'outcomes':dict(Counter(r['outcome'] for r in rows)),
                    'new_clean_passes':sum(r['new_clean_passes'] for r in rows),
                    'episodes_with_new_pass':sum(r['new_clean_passes']>=1 for r in rows),
                    'episodes_with_two_new_passes':sum(r['new_clean_passes']>=2 for r in rows),
                    'survived_with_new_pass':sum(r['survived_with_new_pass'] for r in rows),
                    'outside_front_crossing_episodes':sum(bool(r['outside_front_crossing_steps']) for r in rows),
                    'whole_body_front_return_episodes':sum(r['front_reached_step'] is not None for r in rows),
                    'safe_front_return_episodes':sum(r['safe_front_return'] for r in rows),
                    'crash_terminal_q':describe([r['initial_q'] for r in terminal]),
                    'crash_terminal_soft_return':describe([r['soft_return_observed'] for r in terminal]),
                    'crash_terminal_q_excess':describe([r['initial_q']-r['soft_return_observed'] for r in terminal]),
                    'crash_terminal_absolute_error':describe([abs(r['initial_q']-r['soft_return_observed']) for r in terminal]),
                    'timeout_count_excluded_from_terminal_calibration':sum(r['truncated'] for r in rows),
                    'all_initial_q':describe([r['initial_q'] for r in rows]),
                    'observed_soft_return':describe([r['soft_return_observed'] for r in rows]),
                    'target_bootstrapped_soft_return':describe([r['target_bootstrapped_soft_return'] for r in rows]),
                    'protected_fraction':sum(r['protected_steps'] for r in rows)/sum(r['steps'] for r in rows),
                }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True,
                        help='Absolute path or path relative to --run')
    parser.add_argument('--source-snapshot', type=Path,
                        help='Defaults to RUN/source_snapshot; must contain the saved src package')
    parser.add_argument('--output', type=Path,
                        help='New report path; defaults under D:/DroneSoccerData/diagnostics')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cuda')
    parser.add_argument('--task', choices=('practice','full-flight'), default='practice')
    parser.add_argument('--episodes-per-bank', type=int, default=16)
    parser.add_argument('--rollouts-per-snapshot', type=int, default=2,
                        help='Independent action-RNG repetitions for stochastic mode')
    parser.add_argument('--selection-seed', type=int, default=9_100_000)
    parser.add_argument('--tail-samples', type=int, default=128)
    parser.add_argument('--trace-every', type=int, default=20)
    args = parser.parse_args()
    if min(args.episodes_per_bank,args.rollouts_per_snapshot,args.tail_samples,args.trace_every)<=0:
        parser.error('Episode, repetition, tail-sample and trace intervals must be positive')
    if args.selection_seed < 0 or args.selection_seed >= 2**31:
        parser.error('Selection seed must be in [0, 2**31)')
    run = args.run.resolve()
    checkpoint = args.checkpoint if args.checkpoint.is_absolute() else run/args.checkpoint
    checkpoint = checkpoint.resolve()
    frozen = (args.source_snapshot or run/'source_snapshot').resolve()
    if not (frozen/'src/asymmetric_policy.py').exists():
        raise ValueError('The specified frozen source snapshot has no asymmetric policy implementation.')
    # This is a standalone process: refusing preloaded src avoids mixed-code audits.
    if any(name=='src' or name.startswith('src.') for name in sys.modules):
        raise RuntimeError('Start this audit in a fresh process without an imported src package.')
    sys.path.insert(0,str(frozen))
    asymmetric = importlib.import_module('src.asymmetric_policy')
    context_module = importlib.import_module('src.critic_context')
    return_module = importlib.import_module('src.return_practice')
    evaluation = importlib.import_module('src.return_evaluation')
    anchored = importlib.import_module('src.anchored_policy')
    sim = importlib.import_module('src.sim')
    for module in (asymmetric,context_module,return_module,evaluation,anchored,sim):
        if Path(module.__file__).resolve().parent != frozen/'src':
            raise RuntimeError(f'Mixed simulator implementation: {module.__file__}')
    config_path = run/'config.json'
    config = json.loads(config_path.read_text())
    if (config.get('unified_version') != asymmetric.VERSION or not config.get('asymmetric_critic')
            or config.get('critic_context_version') != context_module.VERSION
            or config.get('critic_context_dim') != context_module.CONTEXT_DIM
            or config.get('reward_version') != return_module.REWARD_VERSION
            or config.get('observation_version') != return_module.OBSERVATION_VERSION):
        raise ValueError('Run configuration does not match the saved asymmetric critic contract.')
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no CPU fallback.')
    torch.set_num_threads(1)
    source_hashes = {str(path.relative_to(frozen)):digest(path)
                     for path in sorted((frozen/'src').rglob('*.py'))}
    immutable = {str(checkpoint):digest(checkpoint), str(config_path):digest(config_path)}
    libraries = {}
    for split, key in [('train','return_library'),('development','return_validation')]:
        library = return_module.ReturnSnapshotLibrary(config[key])
        if library.manifest['split'] != split or library.bank_sha256 != config[key+'_sha256']:
            raise ValueError('Frozen return bank differs from the saved run.')
        libraries[split] = library
        immutable[str(library.directory/'manifest.json')] = library.bank_sha256
    model = asymmetric.AsymmetricSAC.load(checkpoint,device=args.device)
    model.policy.set_training_mode(False)
    if model.observation_space.shape != (100,):
        raise ValueError('The audited actor must retain its 100-value input contract.')
    initial_fingerprint = model_fingerprint(model)
    output = args.output or Path('D:/DroneSoccerData/diagnostics')/(
        f'asymmetric-{run.name}-{model.num_timesteps}-{args.task}-{time.time_ns()}.json')
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f'Preserve the previous report; choose a new output path: {output}')
    output.parent.mkdir(parents=True,exist_ok=True)
    report = {
        'audit_version':'separate-critic-context-rollout-audit-v1',
        'run':str(run),'checkpoint':str(checkpoint),'source_snapshot':str(frozen),
        'audit_script':str(Path(__file__).resolve()),'audit_script_sha256':digest(__file__),
        'source_sha256':source_hashes,'immutable_inputs_sha256':immutable,
        'model_before':initial_fingerprint,'device':str(model.device),
        'versions':{'policy':asymmetric.VERSION,'context':context_module.VERSION,
                    'reward':config['reward_version'],'observation':config['observation_version']},
        'protocol':{
            'task':args.task,'actor_observation_dim':100,'critic_context_dim':context_module.CONTEXT_DIM,
            'episodes_per_bank':args.episodes_per_bank,'stochastic_repetitions':args.rollouts_per_snapshot,
            'selection_seed':args.selection_seed,'tail_samples':args.tail_samples,
            'snapshot_selection':'phase-balanced distinct source flights, without replacement',
            'action_seed_derivation':'SHA256(selection seed, bank SHA256, snapshot ID, repetition, deterministic flag)',
            'environment_rng':'exact stored snapshot RNG; no reseeding after restore',
            'soft_return':'r0 + sum(t>=1) gamma^t * (r_t - alpha*log_pi(a_t|o_t)); queried action entropy excluded',
            'terminal_calibration':'stochastic crash-terminal continuations only; no unknown tail',
            'timeout_tail':'gamma^T * E_a[Q_target(o_T,context_T,a) - alpha*log_pi(a|o_T)]',
            'timeout_tail_is_ground_truth':False,'full_mission_test':False,'historical_passes_counted':False,
            'new_demonstrations':0,'training_updates':0,
            'limitations':['Repeated action seeds from one snapshot are not independent layouts.',
                           'Terminal-subset errors can have survival selection bias.',
                           'Finite Monte Carlo samples do not prove exact expected-Q bias.',
                           'Practice and full-flight contexts differ in their local clock/cap.']},
        'schedules':{},'records':[]}
    alpha = float(model.ent_coef_tensor.item())
    def tensor(observation):
        if np.asarray(observation).shape != (100,):
            raise RuntimeError('Private state entered the actor observation.')
        return torch.as_tensor(observation,dtype=torch.float32,device=model.device).reshape(1,100)
    def tail_value(observation, context):
        x = tensor(observation).repeat(args.tail_samples,1)
        c = torch.as_tensor(context,dtype=torch.float32,device=model.device).reshape(1,-1).repeat(args.tail_samples,1)
        a, logp = model.actor.action_log_prob(x)
        q = torch.cat(model.critic_target(asymmetric.critic_inputs(x,c),a),dim=1).min(dim=1).values
        values = (q-alpha*logp).detach().cpu().numpy()
        return {'target_soft_value_mean':float(values.mean()),
                'target_soft_value_sample_std':float(values.std()),'samples':len(values)}
    with torch.inference_mode():
        for split, library in libraries.items():
            schedule = evaluation._snapshot_schedule(library.records,args.episodes_per_bank,args.selection_seed)
            report['schedules'][split] = schedule
            for index, selected in enumerate(schedule):
                for deterministic in (True,False):
                    repetitions = 1 if deterministic else args.rollouts_per_snapshot
                    for repetition in range(repetitions):
                        seed = stochastic_seed(args.selection_seed,library.bank_sha256,selected['id'],repetition,deterministic)
                        torch.manual_seed(seed)
                        if args.task=='practice':
                            env = return_module.ReturnPracticeEnv(library,horizon=200)
                            observation, _ = env.reset(seed=seed,options={'snapshot_id':selected['id']})
                            horizon = 200
                        else:
                            restored = library.restore(selected['id'])
                            env, observation = restored['environment'],restored['observation'].copy()
                            horizon = sim.LAPS_MAX_STEPS-env.unwrapped.steps
                        base = env.unwrapped
                        original_passes = int(base.passes)
                        if original_passes<1 or float((base.pos-base.hoop_centre)@base.hoop_normal)<=0:
                            raise ValueError('A return snapshot must start behind the ring after a real prior pass.')
                        initial_rng = json.dumps(base.np_random.bit_generator.state,sort_keys=True)
                        context = context_module.critic_context(env)
                        initial_obs_hash = hashlib.sha256(observation.tobytes()).hexdigest()
                        initial_context_hash = hashlib.sha256(context.tobytes()).hexdigest()
                        pending_front = False
                        front_reached = None
                        crossing_steps, pass_steps, trace = [], [], []
                        phase_steps, protected_steps = Counter(),0
                        reward_sum, soft_return, discounted_reward = 0.,0.,0.
                        initial_q, initial_action, initial_std = None,None,None
                        initial_position = base.pos.copy()
                        initial_distance = float(base._progress_distance(base.pos))
                        for elapsed in range(1,horizon+1):
                            x = tensor(observation)
                            c = torch.as_tensor(context,dtype=torch.float32,device=model.device).reshape(1,-1)
                            mu, log_std, _ = model.actor.get_action_dist_params(x)
                            distribution = model.actor.action_dist.proba_distribution(mu,log_std)
                            action_tensor = distribution.get_actions(deterministic=deterministic)
                            log_prob = float(distribution.log_prob(action_tensor).item())
                            action = action_tensor[0].detach().cpu().numpy()
                            if not np.isfinite(action).all() or not np.isfinite(log_prob):
                                raise RuntimeError('Saved policy produced a nonfinite action or log probability.')
                            if initial_q is None:
                                q1,q2 = [float(q.item()) for q in model.critic(asymmetric.critic_inputs(x,c),action_tensor)]
                                initial_q = min(q1,q2)
                                initial_q_pair = [q1,q2]
                                initial_action = action.tolist()
                                initial_std = log_std.exp()[0].detach().cpu().tolist()
                            protected = bool(anchored.protected_observations(x).item())
                            protected_steps += int(protected)
                            phase_steps[base._lap_phase] += 1
                            if elapsed==1 or elapsed%args.trace_every==0:
                                trace.append({'elapsed':elapsed-1,'phase':base._lap_phase,
                                    'position':base.pos.tolist(),'action':action.tolist(),
                                    'normal_std':log_std.exp()[0].detach().cpu().tolist(),
                                    'protected':protected,'speed':float(np.linalg.norm(base.vel)),
                                    'remaining_route_distance':float(base._progress_distance(base.pos))})
                            previous_position, previous_passes = base.pos.copy(),int(base.passes)
                            observation,reward,terminated,truncated,info = env.step(action)
                            context = context_module.critic_context(env)
                            discount = model.gamma**(elapsed-1)
                            reward_sum += float(reward)
                            discounted_reward += discount*float(reward)
                            soft_return += discount*(float(reward)-(alpha*log_prob if elapsed>1 else 0.))
                            crashed = info['outcome'] in sim.CRASHES
                            crossing = base._crossing(previous_position)
                            if crossing is not None:
                                if crossing[0]>0:
                                    pending_front=False
                                elif not crossing[1] and not crashed:
                                    crossing_steps.append(elapsed)
                                    pending_front=True
                            front_clear = float((base.pos-base.hoop_centre)@base.hoop_normal)<-base._clearance_reach()
                            if pending_front and front_clear and not crashed:
                                front_reached = front_reached or elapsed
                                pending_front=False
                            if int(base.passes)>previous_passes:
                                pass_steps.append(elapsed)
                            if terminated or truncated:
                                break
                        else:
                            raise RuntimeError('Expected authentic environment truncation at its declared horizon.')
                        tail = None if terminated else tail_value(observation,context)
                        tail_discount = 0. if terminated else model.gamma**elapsed
                        tail_term = 0. if terminated else tail_discount*tail['target_soft_value_mean']
                        new_passes = int(base.passes)-original_passes
                        survived = bool(elapsed==horizon and truncated and not terminated and not crashed)
                        record = {
                            'split':split,'snapshot_id':selected['id'],'snapshot_sha256':selected['sha256'],
                            'source_seed':selected['seed'],'source_step':selected['source_step'],
                            'source_phase':selected['phase'],'historical_passes':original_passes,
                            'initial_actor_observation_sha256':initial_obs_hash,'initial_context_sha256':initial_context_hash,
                            'initial_environment_rng_sha256':hashlib.sha256(initial_rng.encode()).hexdigest(),
                            'torch_seed':seed,'deterministic':deterministic,'repetition':repetition,
                            'steps':elapsed,'horizon':horizon,'outcome':info['outcome'],
                            'terminated':bool(terminated),'truncated':bool(truncated),'crashed':bool(crashed),
                            'initial_q':initial_q,'initial_q_pair':initial_q_pair,
                            'initial_action':initial_action,'initial_normal_std':initial_std,
                            'reward_sum':reward_sum,'discounted_reward':discounted_reward,
                            'soft_return_observed':soft_return,'queried_action_entropy_included':False,
                            'truncation_tail_discount':tail_discount,'truncation_tail_estimate':tail,
                            'truncation_tail_contribution':tail_term,'target_bootstrapped_soft_return':soft_return+tail_term,
                            'crash_terminal_q_error':initial_q-soft_return if terminated and crashed else None,
                            'new_clean_passes':new_passes,'new_pass_steps':pass_steps,
                            'survived_with_new_pass':bool(new_passes>=1 and survived),
                            'outside_front_crossing_steps':crossing_steps,'front_reached_step':front_reached,
                            'safe_front_return':bool(front_reached is not None and survived),
                            'full_mission_success':False,'phase_steps':dict(phase_steps),
                            'protected_steps':protected_steps,'initial_position':initial_position.tolist(),
                            'final_position':base.pos.tolist(),'initial_route_distance':initial_distance,
                            'final_route_distance':float(base._progress_distance(base.pos)),'trace':trace}
                        report['records'].append(record)
                        env.close()
                print(json.dumps({'split':split,'snapshots_completed':index+1,
                                  'snapshots_total':len(schedule),'rollouts':len(report['records'])}),flush=True)
    report['model_after'] = model_fingerprint(model)
    if report['model_after'] != initial_fingerprint:
        raise RuntimeError('Read-only audit changed model parameters/counters.')
    if any(digest(path)!=expected for path,expected in immutable.items()):
        raise RuntimeError('A checkpoint, config or bank manifest changed during the audit.')
    if any(digest(frozen/path)!=expected for path,expected in source_hashes.items()):
        raise RuntimeError('Frozen simulator source changed during the audit.')
    report['summary'] = summaries(report['records'])
    report['verified_no_learning'] = True
    # Exclusive creation preserves any report produced by another process.
    with output.open('x',encoding='utf-8') as destination:
        json.dump(report,destination,indent=2)
    print(json.dumps({'report':str(output),'summary':report['summary']}),flush=True)


if __name__=='__main__':
    main()
