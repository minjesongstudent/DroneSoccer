"""One finite authentic second-passage reset experiment; simulation development only.

The source prefix reconstructs memory, never a training target. Every restored
episode retains its original simulator clock, delay queue, RNG and passage audit.
There is no execution on import, alternative seed CLI, or automatic retry.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from src.authentic_checkpoints import capture_checkpoint, read_checkpoint, restore_checkpoint, write_checkpoint
from src.sim import HoopEnv
from tools import run_estimated_motion_trial as shared

ROOT, RUNS = shared.ROOT, shared.RUNS
PLAN = 'experiments/faisal/docs/AUTHENTIC_LATE_RESET_TRIAL_20261008.md'
PARITY_KEYS = ('features89', 'public85', 'actions', 'rewards', 'positions', 'headings',
               'terminated', 'truncated', 'absolute_steps', 'full_body_passes')


def status(out, phase, **progress):
    target = Path(out)/'status.json'; temporary = target.with_suffix('.tmp')
    shared.write_json(temporary,dict(phase=phase,**progress))
    temporary.replace(target)


def configuration(smoke):
    bank = list(range(11244810, 11244818) if smoke else range(11245000, 11245064))
    normal = [11244800] if smoke else list(range(11244600, 11244720))
    return dict(smoke=bool(smoke), bank_seeds=bank, normal_seeds=normal,
        training_seeds=sorted(bank+normal), rounds=1 if smoke else 5,
        normal_per_round=1 if smoke else 24, tails_per_round=1 if smoke else 24,
        minimum_parents=1 if smoke else 16, algorithm_seed=11594900 if smoke else 11594000,
        max_total_seconds=600 if smoke else 3600, value_batches=2 if smoke else 100,
        checkpoint_pass=2, checkpoint_latest_step=450, max_original_steps=600,
        target_mode='full_horizon', gamma=1.0, memory_fraction=.15)


def declared_seeds(data):
    """Recognize both historical and this trial's explicit allocation formats."""
    if not isinstance(data, dict): return set()
    result = set()
    for key in ('training_seeds', 'bank_seeds', 'normal_seeds', 'seeds'):
        value = data.get(key, [])
        if isinstance(value, list): result.update(s for s in value if type(s) is int)
    if type(data.get('reset_seed_base')) is int:
        n = data.get('iterations', 0)*data.get('episodes_per_iteration', 0)
        if type(n) is int: result.update(range(data['reset_seed_base'], data['reset_seed_base']+n))
    for key in ('config', 'configuration', 'allocation', 'cohort_config'):
        result.update(declared_seeds(data.get(key)))
    arms = data.get('arms', {})
    for arm in arms.values() if isinstance(arms, dict) else arms if isinstance(arms, list) else ():
        result.update(declared_seeds(arm))
    return result


def scan_allocation(roots, seeds, check=lambda: None):
    wanted = set(seeds)
    if len(wanted) != len(seeds) or any(type(s) is not int or not 11000000 <= s < 11250000 for s in seeds):
        raise ValueError('Unique FIT seed allocation required')
    paths, conflicts, logs, rows, declarations = set(), [], 0, 0, 0
    for root in roots:
        check()
        if not Path(root).is_dir(): continue
        for name in ('episodes.jsonl', 'protocol.json', 'cohort_manifest.json', 'seed_allocation.json'):
            for path in Path(root).rglob(name):
                check(); paths.add(path.resolve())
    for path in sorted(paths):
        check()
        if path.name == 'episodes.jsonl':
            logs += 1
            for line in path.read_text(encoding='utf-8').splitlines():
                check()
                if not line.strip(): continue
                row = json.loads(line); rows += 1
                found = wanted & {row.get('seed'), row.get('parent_seed')}
                if found: conflicts.append(dict(path=str(path), actual_seeds=sorted(found)))
        else:
            declarations += 1
            found = wanted & declared_seeds(json.loads(path.read_text(encoding='utf-8')))
            if found: conflicts.append(dict(path=str(path), allocated_seeds=sorted(found)))
    result = dict(roots=[str(Path(r).resolve()) for r in roots], seeds=list(seeds),
                  episode_logs=logs, episode_rows=rows, declarations=declarations, conflicts=conflicts)
    if conflicts: raise ValueError('FIT allocation overlaps saved or reserved evidence: '+json.dumps(conflicts))
    return result


def eligible_checkpoint(env):
    """Only the current causal clearance event, never the future endpoint."""
    return bool(env._active and env.audit.count == 2 and env.unwrapped.steps <= 450
                and env.unwrapped.wall_steps == 0 and env._trace['clearance_events'][-1] == 1)


def replay_prefix(policy, prefix, budget):
    prefix = np.asarray(prefix)
    if prefix.dtype != np.float32 or prefix.ndim != 2 or prefix.shape[1] != 85 or not np.isfinite(prefix).all():
        raise ValueError('Authentic float32 public85 prefix required')
    policy.reset()
    for observation in prefix:
        budget.check(); features = policy.observe(observation)
        if not np.array_equal(features[:85], observation): raise ValueError('Prefix changed during memory replay')


def endpoint_record(env, prefix_steps, inherited_passes, done, cut):
    return dict(absolute_steps=int(env.unwrapped.steps), prefix_steps=int(prefix_steps),
        tail_length=int(env.unwrapped.steps-prefix_steps), historical_full_body_passes=int(inherited_passes),
        full_body_passes=int(env.audit.count), new_full_body_passes=int(env.audit.count-inherited_passes),
        terminated=bool(done), truncated=bool(cut), wall_steps=int(env.unwrapped.wall_steps),
        outcome=str(env._last_info['outcome']))


def tail_returns(rewards, endpoint):
    """Dense gamma1 returns only over issued suffix commands, with zero end tail."""
    rewards = np.asarray(rewards, np.float32)
    if (rewards.ndim != 1 or not 1 <= len(rewards) <= 600 or not np.isfinite(rewards).all()
            or endpoint['tail_length'] != len(rewards)
            or endpoint['absolute_steps'] != endpoint['prefix_steps']+len(rewards)
            or not 0 <= endpoint['prefix_steps'] < endpoint['absolute_steps'] <= 600
            or type(endpoint['terminated']) is not bool or type(endpoint['truncated']) is not bool
            or endpoint['terminated'] == endpoint['truncated']
            or (endpoint['truncated'] and endpoint['absolute_steps'] != 600)
            or endpoint['new_full_body_passes'] != endpoint['full_body_passes']-endpoint['historical_full_body_passes']
            or endpoint['new_full_body_passes'] < 0):
        raise ValueError('Complete original-horizon suffix endpoint required')
    return shared.discounted_returns(rewards, 1.0)


def run_suffix(policy, env, public, folder, trajectory_id, ancestry, budget, *, deterministic, capture=None):
    """Policy is already reset/replayed. Save actual rows even on an interrupted flight."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    path = folder/(trajectory_id+'.npz')
    if path.exists(): raise FileExistsError('Trajectory identity must be unique')
    prefix_steps, inherited = int(env.unwrapped.steps), int(env.audit.count)
    data = {key: [] for key in PARITY_KEYS}; complete = False; endpoint = None
    try:
        while True:
            budget.check(); features = policy.observe(public)
            if not np.array_equal(features[:85], public): raise ValueError('Actor public prefix changed')
            action, _ = policy.predict(features, deterministic=deterministic)
            next_public, reward, done, cut, _ = env.step(action)
            if not np.isfinite(reward): raise ValueError('Non-finite suffix reward')
            row = dict(features89=features.copy(), public85=public.copy(), actions=action.copy(), rewards=reward,
                positions=env.unwrapped.pos.copy(), headings=env.unwrapped.heading, terminated=bool(done),
                truncated=bool(cut), absolute_steps=env.unwrapped.steps, full_body_passes=env.audit.count)
            for key, val in row.items(): data[key].append(val)
            public = next_public
            if capture is not None: capture(env)
            if done or cut:
                endpoint = endpoint_record(env, prefix_steps, inherited, done, cut)
                complete = True; break
    finally:
        arrays = {key: np.asarray(vals, dtype=np.float32 if key in ('features89','public85','actions') else None)
                  for key, vals in data.items()}
        # Full raw trace is preserved; only PARITY_KEYS rows enter training.
        raw = {'raw_'+key: np.asarray(vals) for key, vals in env._trace.items()}
        meta = dict(ancestry, trajectory_id=trajectory_id, prefix_steps=prefix_steps,
            inherited_full_body_passes=inherited, complete=complete, endpoint=endpoint,
            trace=path.name, suffix_rows=len(data['rewards']))
        np.savez_compressed(path, **arrays, **raw, complete=np.bool_(complete),
            metadata=np.asarray(json.dumps(meta, sort_keys=True)),
            hoop_centre=env.unwrapped.hoop_centre, hoop_normal=env.unwrapped.hoop_normal,
            drone_half_box=env.unwrapped._half_box)
        with (folder/'episodes.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(meta, sort_keys=True)+'\n')
    return arrays, endpoint, path


def exact_suffix_parity(source, restored, prefix_steps):
    for key in PARITY_KEYS:
        expected, actual = np.asarray(source[key])[prefix_steps:], np.asarray(restored[key])
        if expected.dtype != actual.dtype or expected.shape != actual.shape or expected.tobytes() != actual.tobytes():
            raise ValueError('Restored deterministic suffix differs: '+key)
    return dict(passed=True, suffix_rows=len(restored['actions']), compared_fields=list(PARITY_KEYS),
                comparison='dtype, shape and byte-exact; original absolute horizon retained')


def collect_bank(policy, out, config, budget):
    bankdir = out/'bank'; bankdir.mkdir(); parents = []; entries = []
    for seed in config['bank_seeds']:
        budget.check()
        status(out,'collecting_bank',completed_parents=len(entries),eligible_parents=len(parents),
               expected_parents=len(config['bank_seeds']),current_seed=seed)
        env = shared.ClearanceTrainingEnv(HoopEnv(level=1, randomise=True, hanger=True),
                                          'clearance', seed_base=seed, trace_dir=None)
        found = []
        def capture(current):
            if not found and eligible_checkpoint(current):
                snapshot = capture_checkpoint(current, source_sha256=shared.PROFILE.sha256,
                                              fit_seeds=config['bank_seeds'])
                record = write_checkpoint(bankdir/f'{seed}.pkl', snapshot)
                found.append(dict(seed=seed, prefix_steps=int(current.unwrapped.steps), inherited_passes=2,
                                  **record))
        try:
            public, _ = env.reset(seed=seed); policy.reset()
            arrays, endpoint, path = run_suffix(policy, env, public, bankdir/'source', str(seed),
                dict(kind='bank_source', seed=seed, parent_seed=seed), budget, deterministic=True, capture=capture)
            entry = dict(seed=seed, source_trace=str(path), source_sha256=shared.sha(path),
                         endpoint=endpoint, captured=bool(found))
            if found:
                parent = dict(**found[0], source_trace=str(path), source_sha256=shared.sha(path))
                parents.append(parent); entry['checkpoint'] = parent
            entries.append(entry)
            shared.write_json(bankdir/'collection.json', dict(parents=parents, flights=entries, complete=False))
        finally: env.close()
    shared.write_json(bankdir/'collection.json', dict(parents=parents, flights=entries, complete=True))
    # Restore EVERY captured parent, including if the fixed minimum is not met.
    checks = []
    for parent in parents:
        status(out,'verifying_bank',verified_parents=len(checks),eligible_parents=len(parents),current_seed=parent['seed'])
        budget.check(); restored = load_parent(parent)
        try:
            replay_prefix(policy, restored.public_prefix, budget)
            arrays, _, path = run_suffix(policy, restored.env, restored.current_obs, bankdir/'parity',
                str(parent['seed']), dict(kind='bank_parity', seed=parent['seed'], parent_seed=parent['seed']),
                budget, deterministic=True)
            with np.load(parent['source_trace'], allow_pickle=False) as source:
                checked = exact_suffix_parity(source, arrays, parent['prefix_steps'])
            checks.append(dict(parent_seed=parent['seed'], **checked, source_sha256=parent['source_sha256'],
                checkpoint_sha256=parent['payload_sha256'], restored_trace=str(path), restored_sha256=shared.sha(path)))
            shared.write_json(bankdir/'parity.json', dict(passed=False, complete=False, checks=checks))
        finally: restored.env.close()
    parity = dict(passed=True, complete=True, checks=checks, parent_count=len(parents),
                  minimum_parents=config['minimum_parents'], coverage_passed=len(parents) >= config['minimum_parents'])
    shared.write_json(bankdir/'parity.json', parity)
    if not parity['coverage_passed']:
        raise shared.TrialStopped(f'Insufficient eligible parent coverage: {len(parents)} < {config["minimum_parents"]}')
    return parents


def load_parent(parent):
    snapshot = read_checkpoint(parent['path'], expected_manifest_sha256=parent['manifest_sha256'], trusted_local=True)
    meta = snapshot.metadata
    if (snapshot.sha256 != parent['payload_sha256'] or meta['source_seed'] != parent['seed']
            or meta['source_step'] != parent['prefix_steps'] or meta['historical_full_body_passes'] != 2
            or not meta['clearance_this_step'] or not 1 <= meta['source_step'] <= 450):
        raise ValueError('Parent identity or second-passage checkpoint changed')
    return restore_checkpoint(snapshot, trusted_local=True)


def make_schedule(parents, config):
    ids = sorted(parent['seed'] for parent in parents)
    if (len(ids) != len(set(ids)) or len(ids) < config['minimum_parents']
            or not set(ids).issubset(config['bank_seeds'])):
        raise ValueError('Distinct fixed bank parents required')
    rows = []; normal_index = tail_index = 0
    for iteration in range(1, config['rounds']+1):
        for slot in range(config['normal_per_round']):
            rows.append(dict(trajectory_id=f'r{iteration:02d}-normal-{slot:02d}', iteration=iteration,
                kind='normal', seed=config['normal_seeds'][normal_index], parent_seed=None))
            normal_index += 1
            rows.append(dict(trajectory_id=f'r{iteration:02d}-tail-{slot:02d}', iteration=iteration,
                kind='continuation', seed=ids[tail_index % len(ids)], parent_seed=ids[tail_index % len(ids)]))
            tail_index += 1
    if normal_index != len(config['normal_seeds']) or tail_index != config['rounds']*config['tails_per_round']:
        raise ValueError('Prescribed reset mixture must be exact')
    return rows


def collect_training(policy, entry, parents, out, budget):
    if entry['kind'] == 'normal':
        env = shared.ClearanceTrainingEnv(shared.ApproachCurriculumEnv(level=1, randomise=True, hanger=True),
                                          'clearance', seed_base=entry['seed'], trace_dir=None)
        public, _ = env.reset(seed=entry['seed']); policy.reset()
    elif entry['kind'] == 'continuation':
        restored = load_parent(parents[entry['parent_seed']]); env, public = restored.env, restored.current_obs
        replay_prefix(policy, restored.public_prefix, budget)
    else: raise ValueError('Unknown prescribed reset type')
    try:
        arrays, endpoint, path = run_suffix(policy, env, public, out/'training_trajectories',
            entry['trajectory_id'], entry, budget, deterministic=False)
        return arrays, endpoint, path
    finally: env.close()


def pre_fit_manifest(out, parents, schedule, protocol_digest):
    """Freeze the concrete bank and rotation after collection, before any fitting."""
    out = Path(out); shared.write_json(out/'training_schedule.json', schedule)
    paths = [p for p in (out/'bank').rglob('*') if p.is_file()]+[out/'training_schedule.json']
    record = dict(protocol_sha256=protocol_digest, parents=parents,
                  evidence={str(path.relative_to(out)): shared.sha(path) for path in sorted(paths)})
    shared.write_json(out/'pre_fit_manifest.json', record)
    return shared.sha(out/'pre_fit_manifest.json')


def require_pre_fit(out, digest, protocol_digest, check=lambda: None):
    out = Path(out)
    if shared.sha(out/'pre_fit_manifest.json') != digest: raise ValueError('Pre-fit bank manifest changed')
    bound = json.loads((out/'pre_fit_manifest.json').read_text())
    if bound['protocol_sha256'] != protocol_digest or shared.sha(out/'protocol.json') != protocol_digest:
        raise ValueError('Bank binds a different protocol')
    for relative, expected in bound['evidence'].items():
        check()
        path = (out/relative).resolve()
        if not path.is_relative_to(out.resolve()) or shared.sha(path) != expected:
            raise ValueError('Bank/parity/schedule evidence changed')
    parity = json.loads((out/'bank/parity.json').read_text())
    if (not parity['passed'] or not parity['complete'] or not parity['coverage_passed']
            or len(parity['checks']) != len(bound['parents']) or not all(r['passed'] for r in parity['checks'])):
        raise ValueError('Every authentic parent needs complete exact suffix parity before gradients')
    return bound


def training_complete(result, schedule, config):
    expected = [row['trajectory_id'] for row in schedule]
    return bool(schedule and result.get('completed_training_rounds') == config['rounds']
                and result.get('completed_trajectory_ids') == expected
                and len(expected) == config['rounds']*(config['normal_per_round']+config['tails_per_round'])
                and sum(r['kind']=='normal' for r in schedule) == len(config['normal_seeds']))


def decision_reasons(result, schedule, config):
    reasons = []
    if config['smoke']: reasons.append('Software-only smoke can never qualify a policy')
    if not training_complete(result, schedule, config): reasons.append('Prescribed fresh starts and continuations not completed')
    if not shared.pipeline_complete(result): reasons.append('Final pipeline incomplete: '+str(result.get('error',result.get('status'))))
    if not result.get('frozen_unchanged'): reasons.append('Frozen source artifacts or weights not verified unchanged')
    rows = result.get('iterations', [])
    if ([r.get('iteration') for r in rows] != list(range(1,config['rounds']+1))
            or any(r.get('fresh_starts') != config['normal_per_round']
                   or r.get('continuations') != config['tails_per_round'] for r in rows)):
        reasons.append('Every prescribed round and retention guard is required')
    for row in rows:
        reasons.extend(f'Round {row["iteration"]}: {failure}' for failure in row.get('failures', []))
    return reasons


def record_source_integrity(out, result, protocol_digest):
    """Read-only closure check also runs after STOP; never relaunches evaluation."""
    result['frozen_sources_verified'] = False
    try:
        if protocol_digest is None or shared.sha(out/'protocol.json') != protocol_digest:
            raise ValueError('Immutable run protocol unavailable or changed')
        protocol = json.loads((out/'protocol.json').read_text())
        for relative, digest in protocol['sources'].items():
            if shared.sha(ROOT/relative) != digest or shared.sha(out/'source'/relative) != digest:
                raise ValueError('Frozen code or source snapshot changed: '+relative)
        for path, digest in protocol['input_sha256'].items():
            if shared.sha(path) != digest: raise ValueError('Frozen input artifact changed: '+path)
        result['frozen_sources_verified'] = True
    except Exception as exc:
        result['frozen_unchanged'] = False
        result['integrity_error'] = str(exc)
        # Preparation failure has an earlier, more useful cause; retain it.
        if protocol_digest is not None: result['status'] = 'failed'


def final_screen(out, result, config, schedule, reference, protocol_digest, bank_digest, check=lambda: None):
    reasons = decision_reasons(result, schedule, config); candidate = {}; baseline = {}
    evidence, errors = shared.available_evidence(out, reference); reasons.extend(errors)
    pipeline = shared.pipeline_complete(result)
    if pipeline:
        try:
            check(); require_pre_fit(out, bank_digest, protocol_digest, check)
            protocol = json.loads((out/'protocol.json').read_text())
            if protocol['configuration'] != config: raise ValueError('Execution configuration changed')
            for path, digest in protocol['input_sha256'].items():
                check()
                if shared.sha(path) != digest: raise ValueError('Frozen input changed: '+path)
            for relative, digest in protocol['sources'].items():
                check()
                if shared.sha(ROOT/relative) != digest or shared.sha(out/'source'/relative) != digest:
                    raise ValueError('Frozen code or snapshot changed: '+relative)
            for relative, digest in protocol['prepared_sha256'].items():
                if shared.sha(out/relative) != digest: raise ValueError('Prepared FIT/probe artifact changed')
            name = f'policy_{config["rounds"]:02d}.npz'
            if result.get('final_checkpoint') != name: raise ValueError('Only final prescribed candidate allowed')
            if shared.sha(reference['path']) != reference['sha256']: raise ValueError('Baseline manifest changed')
            frozen = json.loads(Path(reference['path']).read_text()); baseline = frozen['metrics']
            for record in frozen['evidence'].values():
                if shared.sha(record['path']) != record['sha256']: raise ValueError('Baseline evidence changed')
            candidate = shared.verify_evaluation_identity(out/'final_development', out/name, 2 if config['smoke'] else 100, check)
            if not config['smoke']: reasons.extend(shared.gae_screen_reasons(baseline, candidate))
            metrics = json.loads((out/'feature_use.json').read_text())
            for split in ('fit','heldout'):
                with np.load(out/f'feature_actions_{split}.npz', allow_pickle=False) as arrays:
                    actual = shared.feature_use_metrics(**{k:arrays[k] for k in ('source','actual','shuffled','zero')})
                if actual != metrics[split] or not actual['zero_estimate_exact'] or (not config['smoke'] and not actual['passed']):
                    reasons.append(split+': frozen feature-use gate failed')
            check()
        except Exception as exc: reasons.append('Evidence validation failed: '+str(exc))
    passed = not reasons
    decision = dict(decision='software_only' if config['smoke'] else
        'eligible_for_one_fresh_development_screen' if passed else 'reject_close_variant', passed=passed,
        reasons=reasons, candidate=candidate, reference=baseline, evidence=evidence,
        training_budget_complete=training_complete(result,schedule,config), final_pipeline_complete=pipeline,
        final_screen_status='evaluated' if pipeline else 'skipped_incomplete_pipeline',
        fresh_start_count=sum(r['kind']=='normal' for r in schedule),
        continuation_count=sum(r['kind']=='continuation' for r in schedule),
        unique_training_parent_layouts=len({r['seed'] for r in schedule
            if r['trajectory_id'] in set(result.get('completed_trajectory_ids',[]))}),
        acceptance=False, policy_promoted=False, physical_flight=False)
    shared.write_json(out/'decision.json', decision); return decision


def freeze_sources(out, budget):
    files = list((ROOT/'src').glob('*.py'))+[Path(__file__).resolve(), ROOT/PLAN,
        ROOT/'tools/run_estimated_motion_trial.py', ROOT/'tools/run_onpolicy_trial.py',
        ROOT/'tools/run_clearance_experiment.py', ROOT/'tools/check_estimated_motion_initialization.py',
        ROOT/'tools/validate_flight_policies.py', ROOT/'tests/test_checkpoint_reset_trial.py',
        ROOT/'tests/test_authentic_checkpoints.py']
    result = {}
    for path in files:
        budget.check(); relative = path.relative_to(ROOT); target = out/'source'/relative
        target.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(path,target)
        result[relative.as_posix()] = shared.sha(target)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--initialization-report', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv); config = configuration(args.smoke)
    started = time.monotonic(); out = args.out.resolve()
    budget = shared.Budget(out, config['max_total_seconds'], started=started)
    with shared.single_trainer(ROOT.parent/'.clearance-training.lock'):
        out.mkdir(parents=True,exist_ok=False)
        result = dict(status='preparing', iterations=[], completed_trajectory_ids=[],
                      frozen_unchanged=False, acceptance=False, physical_flight=False)
        status(out,'preparing')
        policy = optimizer = value = value_optimizer = estimator_digest = None
        reference = protocol_digest = bank_digest = None; schedule = []
        try:
            budget.check()
            allocation = scan_allocation((RUNS, ROOT.parent), config['training_seeds'], budget.check)
            shared.write_json(out/'seed_allocation.json', allocation)
            initialization = shared.verify_initialization(args.initialization_report, budget.check)
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no CPU training fallback')
            torch.cuda.set_per_process_memory_fraction(.15,device=0)
            torch.set_num_threads(1)
            if torch.get_num_interop_threads() != 1: torch.set_num_interop_threads(1)
            torch.backends.cudnn.allow_tf32 = False; torch.backends.cuda.matmul.allow_tf32 = False
            reference = shared.prepare_gae_screen_reference(ROOT.parent/'lab-data/policy-validation-20261006',
                                                           out/'screen_reference',shared.SOURCE)
            reference_guard = shared.baseline_guard(reference)
            guard, selections, inputs = shared.prepare_features(out,budget,smoke=args.smoke)
            inputs.update({str(shared.SOURCE):shared.sha(shared.SOURCE),str(shared.ESTIMATOR):shared.sha(shared.ESTIMATOR),
                           initialization['path']:initialization['sha256']})
            policy = shared.EstimatedMotionPolicy(shared.SOURCE,shared.ESTIMATOR,device='cuda:0')
            shared.seed_everything(config['algorithm_seed'])
            estimator_digest = shared.tensor_digest(policy.motion._model)
            if torch.count_nonzero(policy.actor.residual_weight).item(): raise ValueError('Fresh zero head required')
            optimizer = torch.optim.Adam([policy.actor.residual_weight],lr=1e-5)
            value = nn.Sequential(nn.Linear(89,128),nn.Tanh(),nn.Linear(128,128),nn.Tanh(),nn.Linear(128,1)).cuda()
            value_optimizer = torch.optim.Adam(value.parameters(),lr=3e-4)
            assert all(p.device.type=='cuda' for p in policy.actor.parameters())
            assert all(p.device.type=='cuda' for p in value.parameters())
            trust = torch.as_tensor(guard,device='cuda')
            with torch.no_grad(): reference_dist = policy.actor.base_actor.get_action_dist_params(trust[:,:85].contiguous())[:2]
            protocol = dict(kind='authentic_second_pass_reset', configuration=config,
                training_seeds=config['training_seeds'], bank_seeds=config['bank_seeds'], normal_seeds=config['normal_seeds'],
                manifest=shared.checkpoint_manifest(), initialization=initialization, sources=freeze_sources(out,budget),
                input_sha256=inputs, reference_record=reference,
                prepared_sha256={n:shared.sha(out/n) for n in ('seed_allocation.json','guard.npz','probe_fit.npz','probe_heldout.npz','warmup_episodes.json')},
                actor_lr=1e-5,value_lr=3e-4,actor_epochs=1,batch_size=256,clip_ratio=.1,
                source_mean_kl=.01,source_max_kl=.1,boundary_weight=.1,max_backtracks=4,
                bank_capture='first second full-body clearance, active/no wall, original step<=450; no future filter',
                reset_mixture='24 existing alternating approach/full fresh starts +24 sorted rotating bank tails per round',
                training_units='120 fresh layouts +120 descendants of fixed FIT bank; descendants are not independent layouts',
                history='reset GRU then replay prior public85 rows; current observation consumed once before first tail action',
                target='dense gamma1 MC over suffix only; zero end tail at crash or original step600',
                inference_contract='strict_float32',runtime=dict(python=sys.version,
                    packages={p:importlib.metadata.version(p) for p in ('torch','numpy','stable-baselines3','gymnasium')},
                    cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),device=str(policy.device),
                    threads=torch.get_num_threads(),interop_threads=torch.get_num_interop_threads(),
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                    autocast=False,memory_fraction=.15),acceptance=False,physical_flight=False)
            shared.write_json(out/'protocol.json',protocol); protocol_digest=shared.sha(out/'protocol.json')
            shared.save_training_state(out,'initial',policy,optimizer,value,value_optimizer,dict(stage='zero_head'))
            parents = collect_bank(policy,out,config,budget); result['bank_parent_count']=len(parents)
            if torch.count_nonzero(policy.actor.residual_weight).item(): raise ValueError('Bank collection changed zero head')
            schedule = make_schedule(parents,config)
            bank_digest = pre_fit_manifest(out,parents,schedule,protocol_digest)
            require_pre_fit(out,bank_digest,protocol_digest,budget.check)  # Mandatory before first actor OR value update.
            result['pre_fit_manifest_sha256']=bank_digest
            parents = {p['seed']:p for p in parents}; accepted=rejected=consecutive=0
            def trust_check():
                with torch.no_grad():
                    mean,std,_=policy.actor.get_action_dist_params(trust)
                    kl=shared.gaussian_kl(*reference_dist,mean,std)
                    metrics=dict(mean_kl=float(kl.mean()),max_kl=float(kl.max()))
                return bool(np.isfinite(list(metrics.values())).all() and metrics['mean_kl']<=.01 and metrics['max_kl']<=.1),metrics
            for iteration in range(1,config['rounds']+1):
                status(out,'training',round=iteration,completed_trajectories=len(result['completed_trajectory_ids']),
                       eligible_parents=len(parents))
                observations=[]; actions=[]; rewards=[]; returns=[]; lengths=[]; endpoints=[]
                for entry in (r for r in schedule if r['iteration']==iteration):
                    budget.check(); arrays,endpoint,path=collect_training(policy,entry,parents,out,budget)
                    observations.append(arrays['features89']);actions.append(arrays['actions']);rewards.extend(arrays['rewards'])
                    returns.extend(tail_returns(arrays['rewards'],endpoint)); lengths.append(len(arrays['rewards']))
                    endpoints.append(dict(**entry,**endpoint,trace=str(path),trace_sha256=shared.sha(path)))
                    result['completed_trajectory_ids'].append(entry['trajectory_id'])
                x=torch.as_tensor(np.concatenate(observations),device='cuda');a=torch.as_tensor(np.concatenate(actions),device='cuda')
                target=torch.as_tensor(np.asarray(returns,np.float32),device='cuda')
                with torch.no_grad(): old_log_prob=shared.action_log_prob(policy.actor,x,a).detach()
                np.savez_compressed(out/f'rollout_{iteration:02d}.npz',features89=x.cpu().numpy(),public85=x[:,:85].cpu().numpy(),
                    actions=a.cpu().numpy(),returns=target.cpu().numpy(),old_log_prob=old_log_prob.cpu().numpy(),
                    episode_lengths=lengths,rewards=np.asarray(rewards,np.float32))
                shared.write_json(out/f'episode_endpoints_{iteration:02d}.json',endpoints)
                shared.fit_value_baseline(value,value_optimizer,x,target,config['value_batches'],budget)
                with torch.no_grad():
                    values=value(x).flatten();raw_adv,adv=shared.actor_advantages(target,values,rewards,lengths,'full_horizon')
                np.savez_compressed(out/f'advantage_inputs_{iteration:02d}.npz',rewards=np.asarray(rewards,np.float32),
                    values=values.cpu().numpy(),value_targets=target.cpu().numpy(),episode_lengths=lengths,
                    raw_advantages=raw_adv.cpu().numpy(),target_mode='full_horizon',gamma=1.0)
                np.save(out/f'advantages_{iteration:02d}.npy',adv.cpu().numpy())
                for batch in np.array_split(np.random.permutation(len(x)),max(1,int(np.ceil(len(x)/256)))):
                    budget.check();log=shared.action_log_prob(policy.actor,x[batch],a[batch])
                    mean,std,_=policy.actor.get_action_dist_params(x[batch])
                    with torch.no_grad(): rm,rs,_=policy.actor.base_actor.get_action_dist_params(x[batch,:85].contiguous())
                    gm,gs,_=policy.actor.get_action_dist_params(trust)
                    loss=shared.clipped_loss(log,old_log_prob[batch],adv[batch])+shared.gaussian_kl(rm,rs,mean,std).mean()+.1*shared.boundary_penalty(
                        shared.gaussian_kl(*reference_dist,gm,gs),.01,.1)
                    if not torch.isfinite(loss):raise ValueError('Non-finite actor loss')
                    ok,drift=shared.guarded_optimizer_step(policy.actor,optimizer,loss,trust_check,max_backtracks=4)
                    accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
                    if consecutive>=100:raise shared.TrialStopped('100 consecutive rejected proposals')
                shared.assert_frozen(policy,estimator_digest)
                checkpoint=shared.save_training_state(out,f'policy_{iteration:02d}',policy,optimizer,value,value_optimizer,
                    dict(iteration=iteration,completed_trajectory_ids=result['completed_trajectory_ids'],pre_fit_manifest_sha256=bank_digest))
                result['completed_training_rounds']=iteration;result['final_checkpoint']=checkpoint.name
                metrics=shared.evaluate_head(checkpoint,out/f'guard_{iteration:02d}',2 if args.smoke else 25,budget)
                failures=[] if args.smoke else shared.retention_failures(reference_guard,metrics)
                ok,drift=trust_check()
                if not ok:raise ValueError('Source KL exceeded hard bounds')
                record=dict(iteration=iteration,trajectories=len(lengths),fresh_starts=config['normal_per_round'],
                    continuations=config['tails_per_round'],transitions=len(x),accepted=accepted,rejected=rejected,
                    metrics=metrics,failures=failures,diagnostics=drift)
                result['iterations'].append(record);result['status']='running';shared.write_json(out/'result.json',result)
                print('ROUND',json.dumps(record),flush=True)
                if failures:raise shared.TrialStopped('Retention regression')
            status(out,'final_eval',completed_training_rounds=config['rounds'],
                   completed_trajectories=len(result['completed_trajectory_ids']))
            result['final_development']=shared.evaluate_head(checkpoint,out/'final_development',2 if args.smoke else 100,budget)
            with np.load(checkpoint,allow_pickle=False) as saved:
                if not np.array_equal(saved['residual_weight'],policy.actor.residual_weight.detach().cpu().numpy()):
                    raise ValueError('Final feature probe actor differs from final saved policy')
            result['feature_use']=shared.feature_probe(policy,selections,out,budget)
            shared.assert_frozen(policy,estimator_digest);result['status']='complete';budget.check()
        except (shared.TrialStopped,KeyboardInterrupt) as exc:
            result['status']='stopped';result['error']=str(exc) or 'Keyboard interrupt requested'
        except Exception as exc:
            result['status']='failed';result['error']=repr(exc);(out/'error.txt').write_text(traceback.format_exc())
        finally:
            if policy is not None and optimizer is not None and value_optimizer is not None:
                try:
                    if result['status']!='complete':
                        shared.save_training_state(out,'bounded_partial',policy,optimizer,value,value_optimizer,
                            dict(status=result['status'],completed_trajectory_ids=result['completed_trajectory_ids']))
                except Exception as exc:result['status']='failed';result['integrity_error']=repr(exc)
                shared.record_frozen_integrity(result,policy,estimator_digest)
            record_source_integrity(out,result,protocol_digest)
            result['training_budget_complete']=training_complete(result,schedule,config)
            result['final_pipeline_complete']=shared.pipeline_complete(result)
            try:result['cuda_memory']=shared.cuda_memory_peaks()
            except Exception as exc:result['cuda_memory']=dict(measurement_error=repr(exc))
            result['elapsed_seconds']=time.monotonic()-started
            shared.write_json(out/'result.json',result)
            status(out,result['status'],completed_training_rounds=result.get('completed_training_rounds',0),
                   completed_trajectories=len(result['completed_trajectory_ids']),error=result.get('error'))
            decision=final_screen(out,result,config,schedule,reference,protocol_digest,bank_digest,budget.check)
            print('DECISION',json.dumps({k:v for k,v in decision.items() if k!='evidence'}),flush=True)


if __name__=='__main__':main()
