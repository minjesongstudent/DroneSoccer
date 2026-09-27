"""Bounded, same-data critic calibration; never optimize or deploy an actor.

Run --self-test first. Root coordinates the actual --run after component tests.
The diagnostic deliberately studies the failed 20k actor, not a production
transfer source. Its development bank is not the independent 7M mission test.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

sys.dont_write_bytecode = True
import numpy as np


VERSION = 'paired-frozen-actor-critic-calibration-v1'
DEFAULT_RUN = Path('D:/DroneSoccerData/runs/return-practice-overnight-20260927')
DEFAULT_OUTPUT = Path('D:/DroneSoccerData/diagnostics/asymmetric-comparison-v1')
INITIALIZATION_SEED = 20260927
SAMPLE_SEED = 9_300_027
ACTION_SEED = 9_100_027
UPDATES = 5000
CHUNK = 250
PHASES = ('return_side', 'return_front')
GATE = {
    'relative_terminal_mae_reduction': .20,
    'minimum_terminal_source_flights_overall': 4,
    'minimum_terminal_episodes_for_phase_assessment': 4,
    'maximum_phase_positive_bias_increase': 2.,
    'maximum_phase_mae_increase': 2.,
    'maximum_abs_mean_timeout_min_q': 100.,
    'unsupported_phases_are_unassessed_not_automatically_failed': True,
    'supports_only_a_bounded_diagnostic_behavioral_trial_from_good_source': True,
    'root_review_required_no_automatic_promotion': True,
    'does_not_approve_mission_completion': True,
}


def sha_file(path):
    with Path(path).open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def stable_hash(value):
    """Hash weights and optimizer values without pickle container metadata."""
    digest = hashlib.sha256()

    def add(item):
        if hasattr(item, 'detach'):
            array = item.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode())
            digest.update(str(array.shape).encode())
            digest.update(array.tobytes())
        elif isinstance(item, np.ndarray):
            digest.update(str(item.dtype).encode())
            digest.update(str(item.shape).encode())
            digest.update(np.ascontiguousarray(item).tobytes())
        elif isinstance(item, dict):
            for key in sorted(item, key=str):
                add(key)
                add(item[key])
        elif isinstance(item, (tuple, list)):
            digest.update(str(type(item)).encode())
            for element in item:
                add(element)
        else:
            digest.update((type(item).__name__+':'+repr(item)+';').encode())
    add(value)
    return digest.hexdigest()


def frozen_fingerprint(model):
    names = ('num_timesteps', 'actor_updates', 'actor_attempts',
             'anchor_accepted', 'anchor_rejected')
    return {
        'actor': stable_hash(model.actor.state_dict()),
        'reference_actor': stable_hash(model.reference_actor.state_dict()),
        'actor_optimizer': stable_hash(model.actor.optimizer.state_dict()),
        'counters': {name: getattr(model, name) for name in names},
    }


def finite_soft_returns(rewards, log_prob, gamma, alpha):
    """Realized Q returns: exclude queried action entropy, include later ones.

    For a true terminal episode these are complete sampled returns. For a
    timeout they are only the observed prefix; zero is not a terminal target.
    """
    rewards, log_prob = np.asarray(rewards), np.asarray(log_prob)
    if rewards.ndim != 1 or rewards.shape != log_prob.shape or not len(rewards):
        raise ValueError('A nonempty aligned reward/log-probability sequence is required.')
    result = np.zeros(len(rewards), np.float64)
    result[-1] = rewards[-1]
    for index in range(len(rewards)-2, -1, -1):
        result[index] = rewards[index]+gamma*(result[index+1]-alpha*log_prob[index+1])
    return result


def choose_snapshots(records, count, seed):
    """Phase balance distinct source flights, not merely distinct reset seeds."""
    if count <= 0 or count % 2:
        raise ValueError('The snapshot count must be positive and even.')
    rng = np.random.default_rng(seed)
    candidates = {phase: [row for row in records if row['phase'] == phase] for phase in PHASES}
    for rows in candidates.values():
        rng.shuffle(rows)
    used, chosen = set(), []
    # Alternate phases so a flight represented in both phases cannot dominate.
    for _ in range(count//2):
        for phase in PHASES:
            row = next((r for r in candidates[phase] if r['seed'] not in used), None)
            if row is None:
                raise ValueError('Not enough distinct source flights for phase-balanced selection.')
            chosen.append(dict(row))
            used.add(row['seed'])
    return chosen


def sample_schedule(single_pass, updates, batch_size, seed):
    single_pass = np.asarray(single_pass, dtype=bool)
    single, laps = np.flatnonzero(single_pass), np.flatnonzero(~single_pass)
    if not len(single) or not len(laps) or batch_size % 2:
        raise ValueError('Paired calibration requires both tasks and an even batch size.')
    rng = np.random.default_rng(seed)
    result = np.empty((updates, batch_size), np.int32)
    for index in range(updates):
        result[index] = np.r_[rng.choice(single, batch_size//2), rng.choice(laps, batch_size//2)]
        rng.shuffle(result[index])
    return result


def summary_values(values):
    array = np.asarray(values, dtype=np.float64)
    if not len(array):
        return None
    if not np.isfinite(array).all():
        raise ValueError('Nonfinite diagnostic value.')
    return {'count': len(array), 'mean': float(array.mean()),
            'mean_absolute': float(np.abs(array).mean()),
            'p10': float(np.quantile(array, .1)), 'p90': float(np.quantile(array, .9))}


def evidence_gate(public_records, asymmetric_records):
    """Predeclared conservative evidence gate, not a policy success criterion."""
    p = {row['flight_id']: row for row in public_records}
    a = {row['flight_id']: row for row in asymmetric_records}
    if p.keys() != a.keys():
        raise ValueError('Comparison must use the identical flights.')
    selected = [i for i, row in p.items() if row['split'] == 'development'
                and row['task'] == 'return' and row['terminal'] and row['crash']]
    reasons, phases, uncertainty = [], {}, []
    overall_sources = len({p[i]['source_seed'] for i in selected})
    if overall_sources < GATE['minimum_terminal_source_flights_overall']:
        reasons.append('Fewer than four distinct held-out source flights have exact crash-terminal returns.')
    if not selected:
        reasons.append('No exact terminal crash returns in the held-out development sample.')
        overall = None
    else:
        before = float(np.mean([abs(p[i]['terminal_error']) for i in selected]))
        after = float(np.mean([abs(a[i]['terminal_error']) for i in selected]))
        improvement = 1-after/before if before > 0 else 0.
        overall = {'public_mae': before, 'asymmetric_mae': after, 'relative_reduction': improvement}
        if improvement < GATE['relative_terminal_mae_reduction']:
            reasons.append('Held-out terminal MAE did not fall by at least 20%.')
    for phase in PHASES:
        ids = [i for i in selected if p[i]['source_phase'] == phase]
        sources = len({p[i]['source_seed'] for i in ids})
        metrics = {'terminal_episodes': len(ids), 'distinct_source_flights': sources}
        assessed = len(ids) >= GATE['minimum_terminal_episodes_for_phase_assessment']
        metrics['assessed_for_phase_regression'] = assessed
        if not assessed:
            uncertainty.append(f'{phase}: unassessed for accuracy; fewer than four crash-terminal episodes. '
                               'Timeout prefixes and own bootstraps are not ground truth.')
        if ids:
            pb = float(np.mean([p[i]['terminal_error'] for i in ids]))
            ab = float(np.mean([a[i]['terminal_error'] for i in ids]))
            pm = float(np.mean([abs(p[i]['terminal_error']) for i in ids]))
            am = float(np.mean([abs(a[i]['terminal_error']) for i in ids]))
            metrics.update(public_bias=pb, asymmetric_bias=ab, public_mae=pm, asymmetric_mae=am)
            if assessed and max(0., ab)-max(0., pb) > GATE['maximum_phase_positive_bias_increase']:
                reasons.append(f'{phase}: more than two reward units of new mean optimism.')
            if assessed and am-pm > GATE['maximum_phase_mae_increase']:
                reasons.append(f'{phase}: MAE worsened by more than two reward units.')
        timeouts = [row['q_min'] for row in a.values() if row['split'] == 'development'
                    and row['source_phase'] == phase and not row['terminal']]
        if timeouts and abs(float(np.mean(timeouts))) > GATE['maximum_abs_mean_timeout_min_q']:
            reasons.append(f'{phase}: timeout Q magnitude exceeded the numerical sanity limit.')
        phases[phase] = metrics
    return {'supports_bounded_diagnostic_behavioral_trial': not reasons, 'reasons': reasons,
            'missing_phase_uncertainty': uncertainty, 'terminal_source_flights_overall': overall_sources,
            'overall_terminal_crashes': overall, 'phase_support': phases,
            'interpretation': 'Passing supports a separately reviewed bounded diagnostic behavioral trial '
                              'from the good source actor. Root still decides whether to launch. '
                              'This failed diagnostic actor is never promoted. Unsupported phases remain '
                              'unassessed: this is not evidence of broad critic accuracy or mission success.'}


class Progress:
    def __init__(self, directory):
        self.directory, self.started = directory, time.monotonic()
        self.started_utc = datetime.now(timezone.utc).isoformat()

    def update(self, phase, **details):
        state = {'version': VERSION, 'phase': phase, 'started_utc': self.started_utc,
                 'updated_utc': datetime.now(timezone.utc).isoformat(),
                 'elapsed_seconds': round(time.monotonic()-self.started, 3), **details}
        write_json(self.directory/'status.json', state)
        print(json.dumps(state), flush=True)


def prepare_snapshot(project, source_run, output):
    if output.exists():
        raise FileExistsError('Use a new empty output directory; existing evidence is never overwritten.')
    output.mkdir(parents=True)
    snapshot = output/'source_snapshot'
    (snapshot/'src').mkdir(parents=True)
    (snapshot/'diagnostics').mkdir()
    hashes = {}
    for path in sorted((project/'src').glob('*.py')):
        target = snapshot/'src'/path.name
        shutil.copy2(path, target)
        hashes['src/'+path.name] = sha_file(target)
        if hashes['src/'+path.name] != sha_file(path):
            raise RuntimeError('Source changed during snapshot creation.')
    script = Path(__file__).resolve()
    shutil.copy2(script, snapshot/'diagnostics'/script.name)
    hashes['diagnostics/'+script.name] = sha_file(script)
    config = json.loads((source_run/'config.json').read_text(encoding='utf-8'))
    shutil.copy2(source_run/'config.json', snapshot/'source_run_config.json')
    checkpoint = source_run/'checkpoints'/'model_20000_steps.zip'
    shutil.copy2(checkpoint, snapshot/'diagnostic_actor_20000.zip')
    hashes['diagnostic_actor_20000.zip'] = sha_file(checkpoint)
    if hashes['diagnostic_actor_20000.zip'] != sha_file(snapshot/'diagnostic_actor_20000.zip'):
        raise RuntimeError('Checkpoint changed during copying.')
    banks = {}
    for split, key in [('train', 'return_library'), ('development', 'return_validation')]:
        bank = Path(config[key])
        manifest = json.loads((bank/'manifest.json').read_text(encoding='utf-8'))
        destination = output/'libraries'/split
        destination.mkdir(parents=True)
        shutil.copy2(bank/'manifest.json', destination/'manifest.json')
        for record in manifest['snapshots']:
            if Path(record['file']).name != record['file']:
                raise ValueError('Unsafe snapshot filename.')
            shutil.copy2(bank/record['file'], destination/record['file'])
        banks[split] = {'original_path': str(bank), 'frozen_path': str(destination),
                        'manifest_sha256': sha_file(destination/'manifest.json')}
    write_json(output/'provenance.json', {'source_run': str(source_run), 'source_hashes': hashes,
                                         'banks': banks, 'script_version': VERSION})
    return snapshot, hashes, banks


def collect_dataset(source, libraries, output, progress):
    import torch
    from src.critic_context import critic_context
    from src.curriculum import LessonEnv
    from src.mission_observation import MissionObservation
    from src.sim import CRASHES

    plans = []
    for split, count in [('train', 24), ('development', 16)]:
        selected = choose_snapshots(libraries[split].records, count, INITIALIZATION_SEED)
        for record in selected:
            for trial in range(2):
                plans.append({'split': split, 'task': 'return', 'source_phase': record['phase'],
                              'source_seed': record['seed'], 'snapshot_id': record['id'], 'trial': trial})
    for index in range(20):
        plans.append({'split': 'train', 'task': 'single_pass', 'stage': index % 2,
                      'source_phase': 'approach', 'source_seed': 2_500_000+index, 'trial': 0})
    if len(plans) != 100 or any(7_000_000 <= p['source_seed'] < 8_000_000 for p in plans):
        raise AssertionError('Unexpected flight count or reserved independent-test seed.')
    train_seeds = {p['source_seed'] for p in plans if p['split'] == 'train'}
    development_seeds = {p['source_seed'] for p in plans if p['split'] == 'development'}
    if train_seeds & development_seeds:
        raise AssertionError('Train/development source overlap.')
    fields = ('observations', 'next_observations', 'context', 'next_context',
              'actions', 'rewards', 'log_prob', 'terminated', 'truncated', 'flight_id', 'train')
    rows, flights = {name: [] for name in fields}, []
    alpha = float(source.ent_coef_tensor.item())
    for flight_id, plan in enumerate(plans):
        torch.manual_seed(ACTION_SEED+flight_id*7919)
        if plan['task'] == 'return':
            restored = libraries[plan['split']].restore(plan['snapshot_id'])
            env, observation = restored['environment'], restored['observation'].copy()
        else:
            env = MissionObservation(LessonEnv(stage=plan['stage']), single_pass=True)
            observation, _ = env.reset(seed=plan['source_seed'])
        start = len(rows['rewards'])
        historical_passes = env.unwrapped.passes
        source_step = env.unwrapped.steps
        phase_counts = Counter()
        try:
            for local_step in range(600):
                context = critic_context(env)
                phase_counts[env.unwrapped._lap_phase] += 1
                with torch.no_grad():
                    action, log_prob = source.actor.action_log_prob(
                        torch.as_tensor(observation, device='cuda').reshape(1, -1))
                action_array = action.detach().cpu().numpy()[0]
                next_observation, reward, terminated, truncated, info = env.step(action_array)
                next_context = critic_context(env)
                values = (observation.copy(), next_observation.copy(), context, next_context,
                          action_array.copy(), float(reward), float(log_prob.item()), bool(terminated),
                          bool(truncated), flight_id, plan['split'] == 'train')
                for field, value in zip(fields, values):
                    rows[field].append(value)
                observation = next_observation
                if terminated or truncated:
                    break
            else:
                raise RuntimeError('The original simulator failed to stop by its physical horizon.')
            end = len(rows['rewards'])
            returns = finite_soft_returns(rows['rewards'][start:end], rows['log_prob'][start:end], source.gamma, alpha)
            flights.append({**plan, 'flight_id': flight_id, 'action_seed': ACTION_SEED+flight_id*7919,
                            'start': start, 'end': end, 'steps': end-start, 'source_step': source_step,
                            'terminal': bool(terminated), 'truncated': bool(truncated),
                            'crash': info['outcome'] in CRASHES, 'outcome': info['outcome'],
                            'new_passes': env.unwrapped.passes-historical_passes,
                            'initial_finite_soft_return': float(returns[0]),
                            'truncation_tail_discount': 0. if terminated else source.gamma**(end-start),
                            'phase_counts': dict(phase_counts)})
        finally:
            env.close()
        write_json(output/'flights.json', flights)
        progress.update('collecting', flights=len(flights), flight_cap=100,
                        transitions=len(rows['rewards']), last_outcome=info['outcome'])
    arrays = {}
    for name, values in rows.items():
        dtype = bool if name in ('terminated', 'truncated', 'train') else np.int32 if name == 'flight_id' else np.float32
        arrays[name] = np.asarray(values, dtype=dtype)
    np.savez_compressed(output/'dataset.npz', **arrays)
    write_json(output/'dataset-summary.json', {
        'flights': len(flights), 'transitions': len(arrays['rewards']),
        'train_transitions': int(arrays['train'].sum()),
        'development_transitions': int((~arrays['train']).sum()),
        'sha256': sha_file(output/'dataset.npz'), 'plan': plans,
        'outcomes': dict(Counter(f['split']+'|'+f['source_phase']+'|'+f['outcome'] for f in flights)),
        'new_passes': sum(f['new_passes'] for f in flights),
        'terminal_selection_limit': 'Exact complete returns exist only for true terminal flights. '
            'Here crash-conditioned return statistics can be selection-biased; successful single-pass '
            'terminals are reported separately. Timeouts retain an unknown continuation value.',
    })
    return arrays, flights


def match_critic_initialization(public, asymmetric, context_dim):
    """Same random public critic, zero new columns, identical hidden layers."""
    import torch
    before, after = public.critic.state_dict(), asymmetric.critic.state_dict()
    for name, tensor in before.items():
        target = after[name]
        if tensor.shape == target.shape:
            target.copy_(tensor)
        elif (tensor.ndim == 2 and target.shape[0] == tensor.shape[0]
              and tensor.shape[1] == 104 and target.shape[1] == 104+context_dim):
            target.zero_()
            target[:, :100].copy_(tensor[:, :100])
            target[:, 100+context_dim:].copy_(tensor[:, 100:])
        else:
            raise ValueError(f'Unexpected critic architecture at {name}.')
    asymmetric.critic.load_state_dict(after)
    public.critic_target.load_state_dict(public.critic.state_dict())
    asymmetric.critic_target.load_state_dict(asymmetric.critic.state_dict())
    with torch.no_grad():
        obs = torch.linspace(-1, 1, 1600, device='cuda').reshape(16, 100)
        ctx = torch.ones((16, context_dim), device='cuda')
        act = torch.linspace(-1, .9, 64, device='cuda').reshape(16, 4)
        maximum_error = 0.
        for p, a in zip(public.critic(obs, act), asymmetric.critic(torch.cat((obs, ctx), 1), act)):
            maximum_error = max(maximum_error, float((p-a).abs().max().item()))
            if not torch.allclose(p, a, atol=3e-5, rtol=1e-5):
                raise RuntimeError('Initial critics are not functionally matched.')
    return maximum_error


def make_models(source, arrays, output):
    import torch
    from stable_baselines3.common.logger import configure
    from src.anchored_policy import AnchoredSAC
    from src.asymmetric_policy import AsymmetricSAC
    from src.critic_context import CONTEXT_DIM
    from src.curriculum import LessonEnv
    from src.mission_observation import MissionObservation

    train = np.flatnonzero(arrays['train'])
    models = {}
    for name, cls in [('public', AnchoredSAC), ('asymmetric', AsymmetricSAC)]:
        torch.manual_seed(INITIALIZATION_SEED)
        env = MissionObservation(LessonEnv(stage=0), single_pass=True)
        model = cls('MlpPolicy', env, device='cuda', seed=INITIALIZATION_SEED,
            learning_rate=.0003, buffer_size=max(5000, len(train)+1), learning_starts=0,
            batch_size=source.batch_size, tau=source.tau, gamma=source.gamma,
            train_freq=(1, 'step'), gradient_steps=1, ent_coef=float(source.ent_coef_tensor.item()),
            target_update_interval=source.target_update_interval,
            policy_kwargs={'net_arch': source.actor.net_arch, 'activation_fn': source.actor.activation_fn})
        model.set_logger(configure(str(output/name), ['csv']))
        model.actor.load_state_dict(source.actor.state_dict())
        model.reference_actor.load_state_dict(source.reference_actor.state_dict())
        model.actor.requires_grad_(False)
        model.reference_actor.requires_grad_(False)
        model.policy.set_training_mode(False)
        for index in train:
            info = {'TimeLimit.truncated': bool(arrays['truncated'][index] and not arrays['terminated'][index]),
                    'critic_context': arrays['context'][index], 'next_critic_context': arrays['next_context'][index]}
            model.replay_buffer.add(arrays['observations'][index:index+1], arrays['next_observations'][index:index+1],
                arrays['actions'][index:index+1], arrays['rewards'][index:index+1],
                np.array([arrays['terminated'][index] or arrays['truncated'][index]]), [info])
        if model.device.type != 'cuda' or model.ent_coef_optimizer is not None:
            raise RuntimeError('Local CUDA and fixed source entropy are required.')
        # A time-limit transition must survive the buffer's terminal mask.
        timeouts = np.flatnonzero(arrays['truncated'][train] & ~arrays['terminated'][train])
        if len(timeouts) and model.replay_buffer._get_samples(timeouts).dones.any().item():
            raise AssertionError('Timeouts were incorrectly made terminal in replay.')
        models[name] = model
    error = match_critic_initialization(models['public'], models['asymmetric'], CONTEXT_DIM)
    write_json(output/'initial-critic-equivalence.json', {
        'maximum_absolute_error': error, 'probes': 16, 'nonzero_context_and_actions': True,
        'absolute_tolerance': 3e-5, 'relative_tolerance': 1e-5})
    return models, train


def evaluate_critics(models, arrays, flights, output, updates, progress):
    import torch
    from src.asymmetric_policy import critic_inputs

    results = {}
    for name, model in models.items():
        model.policy.set_training_mode(False)
        alpha = float(model.ent_coef_tensor.item())
        q_values = []
        with torch.no_grad():
            for begin in range(0, len(arrays['rewards']), 512):
                chunk = slice(begin, begin+512)
                obs = torch.as_tensor(arrays['observations'][chunk], device='cuda')
                action = torch.as_tensor(arrays['actions'][chunk], device='cuda')
                if name == 'asymmetric':
                    obs = critic_inputs(obs, torch.as_tensor(arrays['context'][chunk], device='cuda'))
                q_values.append(torch.cat(model.critic(obs, action), 1).cpu().numpy())
            q_values = np.concatenate(q_values)
            records, probe_records = [], []
            for flight in flights:
                first, end = flight['start'], flight['end']
                # Identical actor samples for each critic and each evaluation.
                torch.manual_seed(9_600_000+flight['flight_id'])
                obs = torch.as_tensor(arrays['next_observations'][[first, end-1]], device='cuda').repeat_interleave(64, 0)
                action, log_prob = model.actor.action_log_prob(obs)
                critic_obs = (critic_inputs(obs, torch.as_tensor(arrays['next_context'][[first, end-1]],
                    device='cuda').repeat_interleave(64, 0)) if name == 'asymmetric' else obs)
                soft_value = (torch.cat(model.critic_target(critic_obs, action), 1).min(1).values
                              -alpha*log_prob).reshape(2, 64).mean(1).cpu().numpy()
                finite = finite_soft_returns(arrays['rewards'][first:end], arrays['log_prob'][first:end], model.gamma, alpha)
                terminal = flight['terminal']
                tail = 0. if terminal else model.gamma**(end-first)*float(soft_value[1])
                q1, q2 = map(float, q_values[first])
                qmin = min(q1, q2)
                bootstrap = 0. if arrays['terminated'][first] else model.gamma*float(soft_value[0])
                one_step_target = float(arrays['rewards'][first])+bootstrap
                records.append({**flight, 'q1': q1, 'q2': q2, 'q_min': qmin,
                    'realized_finite_soft_return': float(finite[0]),
                    'exact_complete_sampled_return': float(finite[0]) if terminal else None,
                    'terminal_error': qmin-float(finite[0]) if terminal else None,
                    'terminal_q1_error': q1-float(finite[0]) if terminal else None,
                    'terminal_q2_error': q2-float(finite[0]) if terminal else None,
                    'own_target_bootstrap_tail': tail,
                    'own_bootstrapped_return': float(finite[0])+tail,
                    'q_minus_own_bootstrapped_return': qmin-float(finite[0])-tail,
                    'one_step_reward': float(arrays['rewards'][first]),
                    'one_step_bootstrap': bootstrap, 'one_step_target': one_step_target,
                    'one_step_td_residual': qmin-one_step_target})
                for local in range(0, end-first, 20):
                    q = float(q_values[first+local].min())
                    probe_records.append({'flight_id': flight['flight_id'], 'local_step': local,
                        'split': flight['split'], 'source_phase': flight['source_phase'],
                        'terminal': terminal, 'crash': flight['crash'], 'q_min': q,
                        'exact_complete_sampled_return': float(finite[local]) if terminal else None,
                        'terminal_error': q-float(finite[local]) if terminal else None})
        groups = {}
        for split in ('train', 'development'):
            for phase in ('approach', *PHASES):
                selected = [r for r in records if r['split'] == split and r['source_phase'] == phase]
                if not selected:
                    continue
                terminals = [r for r in selected if r['terminal']]
                crashes = [r for r in terminals if r['crash']]
                groups[split+'|'+phase] = {
                    'episodes': len(selected), 'terminal_episodes': len(terminals),
                    'crash_terminal_episodes': len(crashes), 'excluded_timeouts': len(selected)-len(terminals),
                    'terminal_error': summary_values([r['terminal_error'] for r in terminals]),
                    'terminal_q1_error': summary_values([r['terminal_q1_error'] for r in terminals]),
                    'terminal_q2_error': summary_values([r['terminal_q2_error'] for r in terminals]),
                    'crash_terminal_error': summary_values([r['terminal_error'] for r in crashes]),
                    'own_bootstrap_residual': summary_values([r['q_minus_own_bootstrapped_return'] for r in selected]),
                    'one_step_td_residual': summary_values([r['one_step_td_residual'] for r in selected]),
                    'own_timeout_tail': summary_values([r['own_target_bootstrap_tail'] for r in selected if not r['terminal']]),
                }
        results[name] = {'groups': groups, 'initial_records': records, 'trajectory_probes': probe_records}
        progress.update('evaluating_critics', updates=updates, critic=name)
    paired = []
    for p, a in zip(results['public']['initial_records'], results['asymmetric']['initial_records']):
        assert p['flight_id'] == a['flight_id']
        if p['split'] == 'development' and p['terminal'] and p['crash']:
            paired.append({'source_seed': p['source_seed'], 'source_phase': p['source_phase'],
                           'flight_id': p['flight_id'],
                           'mae_reduction': abs(p['terminal_error'])-abs(a['terminal_error'])})
    clusters = defaultdict(list)
    for row in paired:
        clusters[row['source_seed']].append(row['mae_reduction'])
    cluster_means = np.array([np.mean(values) for values in clusters.values()])
    interval = None
    if len(cluster_means) > 1:
        rng = np.random.default_rng(9_800_027)
        bootstrap = rng.choice(cluster_means, (2000, len(cluster_means)), replace=True).mean(1)
        interval = np.quantile(bootstrap, [.025, .975]).tolist()
    report = {'updates_per_critic': updates, 'models': results, 'paired_terminal_crash_comparison': {
        'positive_favors_asymmetric': True, 'episodes': len(paired), 'distinct_source_flights': len(clusters),
        'source_cluster_mean_mae_reduction': float(cluster_means.mean()) if len(cluster_means) else None,
        'descriptive_95pct_source_cluster_bootstrap_interval': interval,
        'small_sample_caveat': 'Repeated stochastic trials share source states. This descriptive interval '
            'conditions on crash termination and is not a mission-success or timeout-calibration interval.'},
        'gate': evidence_gate(results['public']['initial_records'], results['asymmetric']['initial_records']),
        'definitions': {'terminal_error': 'min(Q1,Q2) minus complete realized soft return; current action entropy excluded.',
            'timeout_comparison': 'Observed soft prefix plus gamma**T times own target critic soft value. '
                'This is self-consistency, not independent ground truth; timeouts never get zero terminal tails.',
            'stochastic_sampling': 'Each realized complete return is exact for its sampled trajectory, '
                'not the noise-free expected Q. Repeated trials estimate variability.'}}
    write_json(output/f'evaluation-{updates:05d}.json', report)
    return report


def calibrate(models, arrays, train, flights, output, progress):
    import torch

    schedule = sample_schedule(arrays['observations'][train, -1] > .5, UPDATES,
                               models['public'].batch_size, SAMPLE_SEED)
    np.save(output/'replay-sample-indices.npy', schedule)
    frozen = {name: frozen_fingerprint(model) for name, model in models.items()}
    original_sample = {name: model.replay_buffer.sample for name, model in models.items()}
    original_action = {name: model.actor.action_log_prob for name, model in models.items()}
    pairing = []
    evaluate_critics(models, arrays, flights, output, 0, progress)
    try:
        for start in range(0, UPDATES, CHUNK):
            fingerprints = {}
            for name, model in models.items():
                cursor = [start]
                action_hash = hashlib.sha256()

                def fixed_sample(batch_size, env=None, *, current=model, offset=cursor):
                    if batch_size != schedule.shape[1] or offset[0] >= start+CHUNK:
                        raise AssertionError('Unexpected replay sampling during frozen-actor calibration.')
                    result = current.replay_buffer._get_samples(schedule[offset[0]], env=env)
                    offset[0] += 1
                    return result

                def tracked_action(obs, *, original=original_action[name], digest=action_hash):
                    actions, log_prob = original(obs)
                    digest.update(actions.detach().cpu().numpy().tobytes())
                    digest.update(log_prob.detach().cpu().numpy().tobytes())
                    return actions, log_prob

                model.replay_buffer.sample = fixed_sample
                model.actor.action_log_prob = tracked_action
                torch.manual_seed(9_400_027+start)
                np.random.seed(9_400_027+start)
                model._calibrate_critics(CHUNK, model.batch_size)
                if cursor[0] != start+CHUNK:
                    raise AssertionError('Unexpected calibration update count.')
                if frozen_fingerprint(model) != frozen[name]:
                    raise AssertionError('Actor, reference, actor optimizer, or actor counters changed.')
                fingerprints[name] = action_hash.hexdigest()
                model.actor.action_log_prob = original_action[name]
                model.replay_buffer.sample = original_sample[name]
                model.logger.dump(start+CHUNK)
            if fingerprints['public'] != fingerprints['asymmetric']:
                raise AssertionError('Critics did not receive byte-identical next actions/log probabilities.')
            pairing.append({'first_update': start+1, 'last_update': start+CHUNK,
                            'sample_indices_sha256': stable_hash(schedule[start:start+CHUNK]),
                            'next_action_and_logprob_sha256': fingerprints['public'],
                            'actors_unchanged': True})
            write_json(output/'pairing-verification.json', pairing)
            progress.update('calibrating', updates_per_critic=start+CHUNK, update_cap=UPDATES)
            if start+CHUNK in (1000, 2500, UPDATES):
                evaluate_critics(models, arrays, flights, output, start+CHUNK, progress)
    finally:
        for name, model in models.items():
            model.actor.action_log_prob = original_action[name]
            model.replay_buffer.sample = original_sample[name]
    for name, model in models.items():
        if frozen_fingerprint(model) != frozen[name]:
            raise AssertionError('Frozen weights changed before saving evidence.')
        model.save(output/name/f'critic-calibrated-{UPDATES}.zip')
    write_json(output/'frozen-actor-verification.json', frozen)


def self_test():
    # Direct expanded Q sum confirms current-action entropy is excluded.
    gamma, alpha = .5, .2
    actual = finite_soft_returns([1., 2., 3.], [900., -2., -4.], gamma, alpha)
    np.testing.assert_allclose(actual, [1+.5*(2+.4)+.25*(3+.8), 2+.5*(3+.8), 3])
    np.testing.assert_allclose(actual, finite_soft_returns([1., 2., 3.], [-900., -2., -4.], gamma, alpha))
    assert actual[0]+gamma**3*10 == actual[0]+1.25  # timeout tail is retained
    schedule = sample_schedule([True, True, False, False, False], 12, 8, 31)
    assert np.all(np.sum(schedule < 2, axis=1) == 4)
    assert np.array_equal(schedule, sample_schedule([True, True, False, False, False], 12, 8, 31))
    records = [{'id': f'{phase}-{seed}', 'phase': phase, 'seed': seed}
               for phase in PHASES for seed in range(12)]
    chosen = choose_snapshots(records, 8, 12)
    assert len({r['seed'] for r in chosen}) == 8
    assert Counter(r['phase'] for r in chosen) == dict.fromkeys(PHASES, 4)
    public, asymmetric = [], []
    for index in range(8):
        row = {'flight_id': index, 'split': 'development', 'task': 'return',
               'terminal': True, 'crash': True, 'source_seed': index,
               'source_phase': PHASES[index//4], 'terminal_error': 10., 'q_min': 1.}
        public.append(row)
        asymmetric.append(dict(row, terminal_error=7.))
    assert evidence_gate(public, asymmetric)['supports_bounded_diagnostic_behavioral_trial']
    asymmetric[0]['terminal_error'] = 100.
    assert not evidence_gate(public, asymmetric)['supports_bounded_diagnostic_behavioral_trial']
    asymmetric[0]['terminal_error'] = 7.
    for index in range(4, 8):
        public[index]['terminal'] = asymmetric[index]['terminal'] = False
        public[index]['terminal_error'] = asymmetric[index]['terminal_error'] = None
    scoped = evidence_gate(public, asymmetric)
    assert scoped['supports_bounded_diagnostic_behavioral_trial']
    assert len(scoped['missing_phase_uncertainty']) == 1
    public[3]['source_seed'] = asymmetric[3]['source_seed'] = 2
    assert not evidence_gate(public, asymmetric)['supports_bounded_diagnostic_behavioral_trial']
    print('Self-tests passed: soft-return entropy index, timeout tail, paired task sampling, '
          'distinct-source selection, scoped evidence gates and explicit unassessed phases.')


def cuda_self_test():
    """No rollouts or optimizer steps: check the real critic input-column map."""
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA self-test cannot fall back to CPU.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.anchored_policy import AnchoredSAC
    from src.asymmetric_policy import AsymmetricSAC
    from src.critic_context import CONTEXT_DIM
    from src.curriculum import LessonEnv
    from src.mission_observation import MissionObservation
    models = []
    for cls in (AnchoredSAC, AsymmetricSAC):
        model = cls('MlpPolicy', MissionObservation(LessonEnv(0), single_pass=True),
                    device='cuda', seed=INITIALIZATION_SEED, buffer_size=8, batch_size=8)
        models.append(model)
    error = match_critic_initialization(*models, CONTEXT_DIM)
    for model in models:
        model.get_env().close()
    print(json.dumps({'cuda_initial_critic_equivalence_passed': True,
                      'maximum_absolute_error': error, 'context_dim': CONTEXT_DIM,
                      'actor_updates': 0, 'critic_updates': 0}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', help='Start the bounded diagnostic after component tests.')
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--cuda-self-test', action='store_true')
    parser.add_argument('--source-run', type=Path, default=DEFAULT_RUN)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.self_test or args.cuda_self_test:
        self_test()
        if args.cuda_self_test:
            cuda_self_test()
        return
    if not args.run:
        parser.error('Use --self-test or explicitly --run; import/default execution performs no training.')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('This diagnostic requires the PC CUDA GPU. CPU fallback is disabled.')
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    project = Path(__file__).resolve().parents[1]
    output, source_run = args.output.resolve(), args.source_run.resolve()
    if output.drive.lower() != 'd:':
        raise ValueError('Diagnostic artifacts must remain on D: because C: has insufficient free space.')
    snapshot, hashes, banks = prepare_snapshot(project, source_run, output)
    progress = Progress(output)
    progress.update('initializing', device=torch.cuda.get_device_name(0))
    try:
        if 'src' in sys.modules:
            raise RuntimeError('Run in a fresh process so all project imports use the frozen snapshot.')
        sys.path.insert(0, str(snapshot))
        from src.anchored_policy import AnchoredSAC
        from src.critic_context import CONTEXT_DIM, VERSION as CONTEXT_VERSION
        from src.return_practice import ReturnSnapshotLibrary
        from src import sim
        assert Path(sim.__file__).resolve().parent == (snapshot/'src').resolve()
        source = AnchoredSAC.load(snapshot/'diagnostic_actor_20000.zip', device='cuda')
        if source.num_timesteps != 20000 or source.observation_space.shape != (100,) or source.use_sde:
            raise ValueError('Expected the failed 20k 100D Gaussian actor diagnostic checkpoint.')
        source.policy.set_training_mode(False)
        source.actor.requires_grad_(False)
        source.reference_actor.requires_grad_(False)
        before = frozen_fingerprint(source)
        libraries = {split: ReturnSnapshotLibrary(bank['frozen_path']) for split, bank in banks.items()}
        protocol = {'version': VERSION, 'gate_predeclared': GATE, 'flights_cap': 100,
            'train_return': {'distinct_source_flights': 24, 'stochastic_trials_each': 2},
            'train_single_pass': {'centered': 10, 'varied': 10},
            'development_return': {'distinct_source_flights': 16, 'stochastic_trials_each': 2},
            'new_teacher_actions': 0, 'new_actor_updates': 0, 'critic_updates_each': UPDATES,
            'critic_learning_rate': .0003, 'batch_size': source.batch_size,
            'gamma': source.gamma, 'alpha': float(source.ent_coef_tensor.item()), 'tau': source.tau,
            'target_update_interval': source.target_update_interval,
            'context_version': CONTEXT_VERSION, 'context_dim': CONTEXT_DIM, 'actor_dim': 100,
            'device': str(source.device), 'gpu': torch.cuda.get_device_name(0),
            'initialization_seed': INITIALIZATION_SEED, 'sample_seed': SAMPLE_SEED,
            'initialization': 'Fresh same-seed public critic, copied common/action columns and hidden '
                'layers into asymmetric critic; added context columns zero. Initial Q functions matched.',
            'replay': 'Identical train transitions, explicit identical 50/50 single/lap sampled indices; '
                'identical per-chunk actor RNG with byte hashes of next actions and log probabilities.',
            'horizon': 'Original simulator clock, full remaining horizon; no artificial 200-step stop. '
                'This calibration dataset differs from production 200-step return practice: it includes '
                'the additional physical transitions and later crashes. Both critics receive the same '
                'extended data; this compares their context use, not a replication of production learning.',
            'diagnostic_actor': 'Failed return-practice 20k actor; never a production transfer source.',
            'final_test': 'Independent 7M mission test is neither loaded nor consumed.',
            'extension': 'Stops at 5000. Any 10000-update extension requires a separate evidence-based decision.',
            'source_fingerprint': before, 'banks': banks}
        write_json(output/'protocol.json', protocol)
        arrays, flights = collect_dataset(source, libraries, output, progress)
        if frozen_fingerprint(source) != before:
            raise AssertionError('Collection changed its frozen diagnostic actor.')
        models, train = make_models(source, arrays, output)
        for model in models.values():
            if stable_hash(model.actor.state_dict()) != before['actor'] or stable_hash(model.reference_actor.state_dict()) != before['reference_actor']:
                raise AssertionError('Model transfer changed public actor or reference weights.')
        protocol['critic_parameter_counts'] = {name: sum(p.numel() for p in m.critic.parameters()) for name, m in models.items()}
        write_json(output/'protocol.json', protocol)
        calibrate(models, arrays, train, flights, output, progress)
        for relative, expected in hashes.items():
            if sha_file(snapshot/relative) != expected:
                raise AssertionError('Frozen source or diagnostic checkpoint changed.')
        if sha_file(source_run/'checkpoints'/'model_20000_steps.zip') != hashes['diagnostic_actor_20000.zip']:
            raise AssertionError('Original diagnostic checkpoint changed during this run.')
        if frozen_fingerprint(source) != before:
            raise AssertionError('Original source actor or optimizer changed.')
        final = json.loads((output/f'evaluation-{UPDATES:05d}.json').read_text(encoding='utf-8'))
        progress.update('complete', updates_per_critic=UPDATES, flights=100,
                        actor_updates=0,
                        supports_bounded_diagnostic_behavioral_trial=final['gate']['supports_bounded_diagnostic_behavioral_trial'],
                        gate_reasons=final['gate']['reasons'],
                        missing_phase_uncertainty=final['gate']['missing_phase_uncertainty'])
    except BaseException as error:
        write_json(output/'error.json', {'type': type(error).__name__, 'message': str(error),
                                         'traceback': traceback.format_exc()})
        progress.update('failed', error=str(error))
        raise


if __name__ == '__main__':
    main()
