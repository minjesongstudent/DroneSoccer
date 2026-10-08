"""One bounded CUDA residual-actor trial; never acceptance or physical flight.

Requires a completed, hash-verified initialization preflight. No experiment runs
on import. STOP and the total wall budget cover preparation, every flight frame,
training batches and evaluation frames. Interrupted runs preserve evidence but
cannot pass the final-only screen.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import shutil
import sys
import time
import traceback

import numpy as np
import torch
from torch import nn

from src.approach_curriculum import ApproachCurriculumEnv
from src.camera_history import CameraHistory
from src.clearance_training import ClearanceTrainingEnv
from src.constrained_sac import boundary_penalty, gaussian_kl, guarded_optimizer_step
from src.trainable_memory_actor import TrainableMemoryPolicy, checkpoint_manifest, SOURCE_PROFILE
from src.policy_runtime import SPECS
from tools.run_clearance_experiment import (ROOT, preserved_rng, retention_failures, rng_state,
                                           sha, single_trainer, summarize, write_json)
from tools.run_onpolicy_trial import (action_log_prob, clipped_loss, counted_development_episodes,
                                     discounted_returns, finite_horizon_gae, gae_screen_reasons, mission_returns,
                                     prepare_gae_screen_reference)
from tools.validate_flight_policies import SUITES, episode


RUNS = Path('D:/DroneSoccerData/runs')
WARMUP = RUNS/'clearance-reward-20261007'
ESTIMATOR = RUNS/'ring-position-estimator-20261008/gru64.pt'
SOURCE = ROOT/'trainedModels/seenflag-s1-level1.zip'
PROFILE = SPECS[SOURCE_PROFILE]
ALGORITHM_SEED = 11604000
MAIN_SEEDS = tuple(range(11243760, 11244000))
SMOKE_SEEDS = (11243752, 11243753)
COHORT_ARMS = ('full_horizon', 'mission', 'local_credit')
DEFAULT_TARGET = 'discounted_mc'


class TrialStopped(RuntimeError):
    pass


class Budget:
    def __init__(self, out, seconds, *, started=None, clock=time.monotonic, parent_check=lambda: None):
        self.out, self.clock = Path(out), clock
        self.started = clock() if started is None else started
        self.deadline = self.started + seconds
        self.parent_check = parent_check

    def check(self):
        if (self.out/'STOP').exists():
            raise TrialStopped('STOP requested')
        if self.clock() >= self.deadline:
            raise TrialStopped('Total wall budget exhausted')
        try:
            self.parent_check()
        except Exception as exc:
            raise TrialStopped('Cohort lease stopped: '+str(exc)) from exc


def seed_everything(seed=ALGORITHM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def target_spec(mode):
    """One declared credit construction; no independent gamma/lambda sweep."""
    if mode not in (DEFAULT_TARGET, *COHORT_ARMS):
        raise ValueError('Unknown declared credit target')
    if mode == 'mission':
        return dict(target_mode=mode, gamma=1.0, gae_lambda=None,
            advantage='Exact binary clean600-step repeat outcome minus150-input value baseline',
            returns='Undiscounted exact mission outcome repeated over each complete episode; no dense reward target')
    if mode == 'local_credit':
        return dict(target_mode=mode, gamma=.995, gae_lambda=.95,
            advantage='GAE95 with frozen predictions from unchanged MC-fitted150-input value baseline',
            returns='Complete dense MC gamma.995 fits value; GAE gamma.995/lambda.95 actor advantages; zero crash/600 tails')
    return dict(target_mode=mode, gamma=1.0 if mode == 'full_horizon' else .995, gae_lambda=None,
        advantage='complete-episode MC minus150-input value baseline',
        returns='zero tail at crash and finite600step cutoff; no Q or GAE')


def episode_targets(rewards, endpoint, mode):
    """Truth is used only for a completed mission label, never as an actor input."""
    rewards = np.asarray(rewards, np.float32)
    if (rewards.ndim != 1 or not 1 <= len(rewards) <= 600 or not np.isfinite(rewards).all()
            or endpoint['steps'] != len(rewards)
            or type(endpoint['terminated']) is not bool or type(endpoint['truncated']) is not bool
            or not (endpoint['terminated'] or endpoint['truncated'])
            or type(endpoint['full_body_passes']) is not int or endpoint['full_body_passes'] < 0
            or type(endpoint['wall_steps']) is not int or endpoint['wall_steps'] < 0):
        raise ValueError('A complete real episode endpoint and matching finite reward rows are required')
    spec = target_spec(mode)
    if mode == 'mission':
        return mission_returns(len(rewards), endpoint['full_body_passes'],
                               endpoint['terminated'], endpoint['wall_steps'])
    return discounted_returns(rewards, spec['gamma'])


def actor_advantages(target, frozen_values, rewards, lengths, mode):
    """Frozen pre-update baseline and isolated finite episodes; preserve default arithmetic."""
    spec = target_spec(mode)
    if target.ndim != 1 or frozen_values.shape != target.shape:
        raise ValueError('Matching one-dimensional target and frozen values required')
    if sum(lengths) != len(target) or any(n <= 0 for n in lengths):
        raise ValueError('Complete episode lengths must partition the target')
    if mode == 'local_credit':
        raw = torch.as_tensor(finite_horizon_gae(rewards, frozen_values.detach().cpu().numpy(),
            lengths, gamma=spec['gamma'], lam=spec['gae_lambda']), device=target.device)
    else:
        raw = target-frozen_values
    if not torch.isfinite(raw).all(): raise ValueError('Non-finite actor advantage')
    normalized = (raw-raw.mean())/(raw.std(unbiased=False)+1e-8)
    return raw, normalized


def prescribed_seeds(smoke, training_seeds=None):
    seeds = tuple(SMOKE_SEEDS if smoke else MAIN_SEEDS) if training_seeds is None else tuple(training_seeds)
    if (len(seeds) != (2 if smoke else 240) or len(set(seeds)) != len(seeds)
            or any(type(seed) is not int or not 11000000 <= seed < 11250000 for seed in seeds)):
        raise ValueError('Exact unique prescribed FIT episode schedule required')
    return seeds


def scan_allocation(root, seeds, check=lambda: None):
    """Cover executed episodes and reservations including coordinator manifests."""
    from tools.run_parallel_estimated_motion import existing_allocation
    check()
    record = existing_allocation(root, list(seeds))
    record['sharing'] = 'One standalone trainable control-memory trial; no sharing'
    check()
    return record


def verify_initialization(path, check=lambda: None):
    """Require the completed real75 preflight and its unchanged sources/evidence."""
    path = Path(path).resolve(); data = json.loads(path.read_text())
    if (data.get('kind') != 'trainable_memory_initialization_parity'
            or data.get('status') != 'complete' or data.get('smoke') is not False
            or data.get('expected_episodes') != 75 or data.get('completed_episodes') != 75
            or data.get('initialization_parity_passed') is not True
            or data.get('source_sha256') != PROFILE.sha256
            or data.get('estimator_sha256') != checkpoint_manifest()['estimator_sha256']):
        raise ValueError('Completed non-smoke 75-episode initialization parity is required')
    protocol_path = path.parent/'protocol.json'
    if sha(protocol_path) != data['protocol_sha256']:
        raise ValueError('Initialization protocol changed')
    for relative, digest in data['evidence_sha256'].items():
        check(); evidence = (path.parent/relative).resolve()
        if not evidence.is_relative_to(path.parent) or sha(evidence) != digest:
            raise ValueError('Initialization evidence changed or escaped its directory')
    if sha(path.parent/'zero_residual.npz') != data['residual_sha256']:
        raise ValueError('Initialization head identity mismatch')
    protocol = json.loads(protocol_path.read_text())
    current = protocol.get('source_files_sha256')
    if not isinstance(current, dict) or not current:
        raise ValueError('Initialization source hash mapping missing')
    for relative, digest in current.items():
        check(); source = ROOT/relative
        if sha(source) != digest: raise ValueError(f'Initialization source changed: {relative}')
    return dict(path=str(path), sha256=sha(path), protocol_sha256=data['protocol_sha256'])


def public_episode_arrays(trace):
    """Read only public observations, issued commands and episode boundary flags.

    Position/headings/targets in an archive are neither retrieved nor consumed.
    Current action is used only to check the following public history row.
    """
    observations = np.asarray(trace['actor_observations'])
    actions = np.asarray(trace['actions'])
    done = np.asarray(trace['terminated'], bool) | np.asarray(trace['truncated'], bool)
    n = len(observations)
    if (observations.dtype != np.float32 or observations.shape != (n, 85)
            or actions.shape != (n, 4) or not 3 <= n <= 600
            or done.shape != (n,) or done[:-1].any() or not done[-1]
            or not np.isfinite(observations).all() or not np.isfinite(actions).all()
            or (np.abs(actions) > 1).any()):
        raise ValueError('Finite public85/issued-action rows in one complete episode required')
    history = CameraHistory(PROFILE.lags)
    for i, observation in enumerate(observations):
        previous = np.zeros(4, np.float32) if i == 0 else actions[i-1]
        if not np.array_equal(history.update(observation[64:72], previous), observation):
            raise ValueError('Saved public camera/issued-command history mismatch')
    return observations.copy()


def warmup_records(folder, check=lambda: None, *, smoke=False):
    folder = Path(folder).resolve()
    rows = [json.loads(line) for line in (folder/'episodes.jsonl').read_text().splitlines() if line.strip()]
    records, cumulative, seen = [], 0, set()
    for row in rows:
        check(); cumulative += row['steps']
        if cumulative > 200000: break
        if row['status'] != 'complete' or row['steps'] < 3: continue
        seed = row['seed']; path = (folder/row['trace']).resolve()
        if path.parent != folder or seed in seen or not 11000000 <= seed < 11250000:
            raise ValueError('Unique FIT source episodes inside declared folder required')
        seen.add(seed); index = len(records)
        records.append(dict(index=index, seed=seed, path=str(path), steps=row['steps'],
                            split='heldout' if index % 5 == 4 else 'fit', sha256=sha(path)))
        if smoke and len(records) == 10: break
    if not smoke and (len(records) != 558 or sum(row['steps'] for row in records) != 199929
                      or [r['seed'] for r in records] != list(range(11000000, 11000558))):
        raise ValueError('Original complete warmup cohort changed')
    return records


def probe_selection(features, count, seed):
    """Freeze valid rows and a joint 64-component shuffle within visibility strata."""
    valid = np.flatnonzero(features[:, 149] == 1)
    if len(valid) < count: raise ValueError('Insufficient valid probe rows')
    rng = np.random.default_rng(seed)
    selected = rng.choice(valid, count, replace=False)
    permutation = np.arange(count)
    visible = features[selected, 70] > .5
    for status in (False, True):
        indices = np.flatnonzero(visible == status)
        permutation[indices] = rng.permutation(indices)
    return selected, permutation


def prepare_features(out, budget, *, smoke):
    """Reconstruct strict runtime estimates from reset, never historical TF32 outputs."""
    protocol = json.loads((WARMUP/'protocol.json').read_text())
    warmup_check = json.loads((WARMUP/'clearance/warmup_check.json').read_text())
    if (protocol['source']['sha256'] != PROFILE.sha256 or protocol['warmup'] != 200000
            or not warmup_check['exact_policy_weights'] or warmup_check['new_updates'] != 0
            or warmup_check['steps'] != 200000):
        raise ValueError('Warmup is not the unchanged original policy')
    records = warmup_records(WARMUP/'clearance/training_trajectories', budget.check, smoke=smoke)
    # Independent CPU runtime; no sampled commands and no actor fitting here.
    with preserved_rng():
        policy = TrainableMemoryPolicy(SOURCE, ESTIMATOR, device='cpu')
        chunks = {'fit': [], 'heldout': []}; offsets = {'fit': 0, 'heldout': 0}
        for record in records:
            budget.check()
            with np.load(record['path'], allow_pickle=False) as archive:
                observations = public_episode_arrays(archive)
            if len(observations) != record['steps']: raise ValueError('Warmup length mismatch')
            policy.reset(); features = []
            for observation in observations:
                budget.check(); features.append(policy.observe(observation))
            features = np.asarray(features, np.float32)
            if not np.array_equal(features[:, :85], observations): raise ValueError('Public prefix changed')
            split = record['split']; record['split_row_start'] = offsets[split]
            offsets[split] += len(features); chunks[split].append(features)
    arrays = {split: np.concatenate(parts) for split, parts in chunks.items()}
    count = 32 if smoke else 8192
    guard_ids = np.random.default_rng(11520000).choice(len(arrays['fit']), count, replace=False)
    guard = arrays['fit'][guard_ids]
    selections = {}
    for split, seed in (('fit', 11591000), ('heldout', 11591001)):
        ids, permutation = probe_selection(arrays[split], count, seed)
        selections[split] = dict(features=arrays[split][ids], row_ids=ids, permutation=permutation)
        np.savez_compressed(out/f'probe_{split}.npz', **selections[split])
    np.savez_compressed(out/'guard.npz', features=guard, fit_row_ids=guard_ids)
    write_json(out/'warmup_episodes.json', records)
    hashes = {record['path']: record['sha256'] for record in records}
    for path in (WARMUP/'protocol.json', WARMUP/'clearance/warmup_check.json',
                 WARMUP/'clearance/training_trajectories/episodes.jsonl'):
        hashes[str(path)] = sha(path)
    return guard, selections, hashes


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode()); digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def assert_frozen(policy, estimator_digest):
    """The initial estimator FILE is frozen; candidate control memory is trainable."""
    policy.actor.assert_source_unchanged()
    if not all(p.requires_grad for p in policy.actor.memory.parameters()):
        raise ValueError('Control-memory parameters unexpectedly frozen')


def episode_batch(sequences, device):
    lengths = torch.tensor([len(x) for x in sequences], dtype=torch.int64)
    public = np.zeros((len(sequences),int(lengths.max()),85),np.float32)
    for i,x in enumerate(sequences):
        if x.dtype != np.float32 or x.shape != (len(x),85) or not np.isfinite(x).all():
            raise ValueError('Complete finite public85 episode required')
        public[i,:len(x)] = x
    return torch.as_tensor(public,device=device),lengths


@torch.no_grad()
def selected_current_features(policy, sequences, selected, check=lambda: None):
    """Recompute causal memory after updates; no stale-hidden trust/probe shortcut."""
    selected=np.asarray(selected)
    if (selected.ndim!=1 or not np.issubdtype(selected.dtype,np.integer)
            or len(selected)==0 or selected.min()<0 or selected.max()>=sum(map(len,sequences))):
        raise ValueError('Valid original split row ids required')
    result=torch.empty((len(selected),150),dtype=torch.float32,device=policy.device)
    offset=0
    for start in range(0,len(sequences),16):
        check(); group=sequences[start:start+16]; n=sum(map(len,group))
        ids=np.flatnonzero((selected>=offset)&(selected<offset+n))
        if len(ids):
            public,lengths=episode_batch(group,policy.device)
            x=policy.actor.sequence_features(public,lengths)
            result[torch.as_tensor(ids,device=policy.device)]=x[torch.as_tensor(selected[ids]-offset,device=policy.device)]
        offset+=n
    return result


def load_public_sequences(out, check=lambda: None):
    result={'fit':[], 'heldout':[]}
    for row in json.loads((Path(out)/'warmup_episodes.json').read_text()):
        check()
        if sha(row['path'])!=row['sha256']:raise ValueError('Warmup changed')
        with np.load(row['path'],allow_pickle=False) as saved:
            result[row['split']].append(public_episode_arrays(saved))
    return result


def evaluate_head(checkpoint, folder, count, budget):
    from tools.check_trainable_memory_initialization import InitializationBridge
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    identity = dict(path=str(Path(checkpoint).resolve()), sha256=sha(checkpoint),
                    manifest=checkpoint_manifest(), episodes_per_suite=count,
                    precision='strict_float32', diagnostic_dynamics=None)
    write_json(folder/'model.json', identity)
    result = {}; files = {}
    with preserved_rng():
        policy = TrainableMemoryPolicy.load(checkpoint, SOURCE, ESTIMATOR, device='cpu', sha256=identity['sha256'])
        bridge = InitializationBridge(policy, reference_sac=None, guard=budget.check)
        for suite, seed_base in SUITES.items():
            target = folder/suite; target.mkdir(); rows = []
            for index in range(count):
                budget.check(); seed = seed_base+index; trajectory = target/f'{seed}.npz'
                row = episode(bridge, suite, seed, trajectory)
                rows.append(row)
                with (target/'episodes.jsonl').open('a', encoding='utf-8') as stream:
                    stream.write(json.dumps(row)+'\n')
                files[f'{suite}/{seed}.npz'] = sha(trajectory)
            result[suite] = summarize(rows)
            files[f'{suite}/episodes.jsonl'] = sha(target/'episodes.jsonl')
            print('EVAL', Path(checkpoint).name, suite, json.dumps(result[suite]), flush=True)
    budget.check(); write_json(folder/'summary.json', result)
    for name in ('model.json', 'summary.json'): files[name] = sha(folder/name)
    write_json(folder/'evidence.json', files)
    return result


def baseline_guard(reference_record):
    frozen = json.loads(Path(reference_record['path']).read_text())
    result = {}
    for suite, seed in SUITES.items():
        path = frozen['evidence'][f'{PROFILE.name}/{suite}/episodes.jsonl']['path']
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        selected = [row for row in rows if seed <= row['seed'] < seed+25]
        if len(selected) != 25: raise ValueError('Exactly25 original guard rows required')
        result[suite] = summarize(selected)
    return result


def save_training_state(out, label, policy, optimizer, value, value_optimizer, metadata):
    checkpoint = out/(label+'.npz'); digest = policy.save(checkpoint)
    torch.save(dict(residual_optimizer=optimizer.state_dict(), value=value.state_dict(),
                    value_optimizer=value_optimizer.state_dict(), rng=rng_state(),
                    metadata=metadata, checkpoint_sha256=digest,
                    source_manifest=checkpoint_manifest()), out/(label+'-state.pt'))
    return checkpoint


def collect_episode(policy, env, seed, folder, budget, *, include_endpoint=False):
    public, _ = env.reset(seed=seed); policy.reset()
    observations, actions, rewards = [], [], []; complete = False; endpoint = None
    try:
        while True:
            budget.check(); features = policy.observe(public)
            if not np.array_equal(features[:85], public): raise ValueError('Actor public prefix changed')
            action, _ = policy.predict(features, deterministic=False)
            next_public, reward, done, cut, _ = env.step(action)
            observations.append(features.copy()); actions.append(action.copy()); rewards.append(reward)
            if not np.isfinite(reward): raise ValueError('Non-finite reward')
            public = next_public
            if done or cut:
                endpoint = dict(steps=len(rewards), full_body_passes=int(env.audit.count),
                    terminated=bool(done), truncated=bool(cut), wall_steps=int(env.unwrapped.wall_steps))
                complete = True; break
    finally:
        if observations:
            x = np.asarray(observations, np.float32)
            np.savez_compressed(folder/f'{seed}.npz', features150=x, public85=x[:, :85],
                                actions=np.asarray(actions, np.float32), rewards=np.asarray(rewards, np.float32),
                                seed=np.int64(seed), complete=complete,
                                **({'endpoint_'+key: value for key, value in endpoint.items()} if endpoint else {}))
        if not complete: env.flush_partial('bounded_partial')
    arrays = (np.asarray(observations, np.float32), np.asarray(actions, np.float32), np.asarray(rewards, np.float32))
    return (*arrays, endpoint) if include_endpoint else arrays


def source_actions(policy, features):
    public = torch.as_tensor(features[:, :85].copy(), device=policy.device)
    with torch.no_grad(): actions = policy.actor.base_actor(public, deterministic=True).cpu().numpy()
    return policy.actor.base_actor.unscale_action(actions)


def feature_use_metrics(source, actual, shuffled, zero):
    if not all(np.isfinite(a).all() for a in (source, actual, shuffled, zero)):
        raise ValueError('Finite action probes required')
    d_source = float(np.mean(np.abs(actual-source)))
    d_shuffle = float(np.mean(np.abs(actual-shuffled)))
    zero_exact = bool(np.array_equal(source, zero))
    passed = zero_exact and d_source > 1e-6 and d_shuffle > 1e-6 and d_shuffle >= .5*d_source
    return dict(D_source=d_source, D_shuffle=d_shuffle, zero_estimate_exact=zero_exact, passed=passed)


def feature_probe(policy, selections, out, budget):
    result = {}
    for split, selection in selections.items():
        budget.check(); x = selection['features']
        shuffled = x.copy(); shuffled[:, 85:149] = x[selection['permutation'], 85:149]
        zero = x.copy(); zero[:, 85:149] = 0
        source = source_actions(policy, x)
        actual = policy.predict(x)[0]; other = policy.predict(shuffled)[0]; blank = policy.predict(zero)[0]
        result[split] = feature_use_metrics(source, actual, other, blank)
        np.savez_compressed(out/f'feature_actions_{split}.npz', source=source, actual=actual,
                            shuffled=other, zero=blank)
    write_json(out/'feature_use.json', result)
    return result


def training_budget_complete(result, smoke, training_seeds=None):
    """Training completion is independent of a subsequent retention/evaluation stop."""
    try: seeds = prescribed_seeds(smoke, training_seeds)
    except (ValueError, TypeError): return False
    expected_rounds = 1 if smoke else 5
    rounds = result.get('iterations', [])
    if 'completed_training_rounds' not in result:
        expected_episodes = 2 if smoke else 48
        if ([r.get('iteration') for r in rounds] != list(range(1, expected_rounds+1))
                or any(r.get('episodes') != expected_episodes for r in rounds)):
            return False
    completed_rounds = result.get('completed_training_rounds', len(rounds))
    return (completed_rounds == expected_rounds
            and result.get('completed_seeds') == list(seeds))


def pipeline_complete(result):
    return (result.get('status') == 'complete' and 'final_development' in result
            and 'feature_use' in result)


def completion_reasons(result, smoke, training_seeds=None):
    reasons = []
    if smoke: reasons.append('Software-only smoke can never pass')
    if not pipeline_complete(result):
        detail = result.get('error', result.get('status', 'unknown status'))
        reasons.append(f'Final validation pipeline not completed: {detail}')
    if not training_budget_complete(result, smoke, training_seeds):
        reasons.append('Prescribed training budget was not completed')
    try: seeds = list(prescribed_seeds(smoke, training_seeds))
    except (ValueError, TypeError): seeds = None
    rounds = result.get('iterations', [])
    expected_rounds, expected_episodes = (1, 2) if smoke else (5, 48)
    if (len(rounds) != expected_rounds or [r.get('iteration') for r in rounds] != list(range(1, expected_rounds+1))
            or any(r.get('episodes') != expected_episodes for r in rounds)
            or seeds is None or result.get('completed_seeds') != seeds):
        reasons.append('Requires every prescribed round and complete episode exactly once')
    for row in rounds:
        reasons.extend(f"Round {row.get('iteration')}: {failure}" for failure in row.get('failures', []))
    if not smoke and not result.get('learned_memory_changed'): reasons.append('Control memory did not change')
    if not result.get('frozen_unchanged'): reasons.append('Frozen sources or weights were not verified unchanged')
    return reasons


def record_frozen_integrity(result, policy, estimator_digest):
    """Report successful frozen checks even when a later flight guard stopped work."""
    result['frozen_unchanged'] = False
    result['frozen_weights_verified'] = False
    try:
        assert_frozen(policy, estimator_digest)
        result['frozen_weights_verified'] = True
        expected = {str(SOURCE): PROFILE.sha256,
                    str(ESTIMATOR): checkpoint_manifest()['estimator_sha256']}
        actual = {path: sha(path) for path in expected}
        result['frozen_artifact_checks'] = {path: dict(expected_sha256=digest,
            actual_sha256=actual[path], unchanged=actual[path] == digest)
            for path, digest in expected.items()}
        if actual != expected:
            raise ValueError('Frozen source/estimator artifact checksum changed')
        result['frozen_unchanged'] = True
    except Exception as exc:
        result['status'] = 'failed'
        result['integrity_error'] = repr(exc)
    return result['frozen_unchanged']


def available_evidence(out, reference_record=None):
    """Hash existing saved artifacts during cleanup without starting skipped work.

    Reporting cleanup is allowed after STOP/budget exhaustion, like checkpoint
    preservation. It performs file reads only. Missing final outputs are not
    requested; the decision being written and temporary files are excluded.
    """
    out = Path(out)
    paths = {path for path in out.rglob('*') if path.is_file()
             and path != out/'decision.json' and path.suffix != '.tmp'}
    paths.update(path for path in (SOURCE, ESTIMATOR) if Path(path).is_file())
    if reference_record and Path(reference_record.get('path', '')).is_file():
        paths.add(Path(reference_record['path']))
    evidence, errors = {}, []
    for path in sorted(paths):
        try:
            evidence[str(path)] = sha(path)
        except OSError as exc:
            errors.append(f'Available saved artifact could not be hashed: {path}: {exc}')
    return evidence, errors


def verify_evaluation_identity(folder, checkpoint, count, check=lambda: None):
    folder = Path(folder); identity = json.loads((folder/'model.json').read_text())
    if (identity['sha256'] != sha(checkpoint) or identity['manifest'] != checkpoint_manifest()
            or identity['episodes_per_suite'] != count or identity['diagnostic_dynamics'] is not None
            or identity['precision'] != 'strict_float32'):
        raise ValueError('Evaluation identity differs from final candidate/contract')
    evidence = json.loads((folder/'evidence.json').read_text())
    required = {'model.json', 'summary.json'}
    for suite, seed in SUITES.items():
        required.add(f'{suite}/episodes.jsonl')
        required.update(f'{suite}/{s}.npz' for s in range(seed, seed+count))
    if set(evidence) != required: raise ValueError('Evaluation evidence must cover all exact trajectories')
    for relative, digest in evidence.items():
        check()
        if sha(folder/relative) != digest: raise ValueError('Evaluation evidence changed')
    summary = json.loads((folder/'summary.json').read_text()); candidate = {}
    for suite, seed in SUITES.items():
        counted = counted_development_episodes(folder/suite/'episodes.jsonl', seed, count)
        if any(summary[suite][key] != val for key, val in counted.items()):
            raise ValueError('Evaluation summary disagrees with episodes')
        candidate[suite] = counted
    return candidate


def final_screen(out, result, reference_record, protocol_digest, *, smoke, check=lambda: None,
                 training_seeds=None):
    out = Path(out)
    reasons = completion_reasons(result, smoke, training_seeds); candidate = {}; reference = {}
    evidence, hash_errors = available_evidence(out, reference_record)
    reasons.extend(hash_errors)
    if not pipeline_complete(result):
        decision = dict(decision='software_only' if smoke else 'reject_close_variant',
            passed=False, reasons=reasons, reference=reference, candidate=candidate,
            evidence=evidence, training_budget_complete=training_budget_complete(result, smoke, training_seeds),
            final_pipeline_complete=False, final_screen_status='skipped_incomplete_pipeline',
            final_development_completed='final_development' in result,
            feature_use_completed='feature_use' in result,
            acceptance=False, policy_promoted=False, physical_flight=False)
        write_json(out/'decision.json', decision)
        return decision
    try:
        check()
        protocol = json.loads((out/'protocol.json').read_text())
        if sha(out/'protocol.json') != protocol_digest: raise ValueError('Pre-fit protocol changed')
        if protocol['training_seeds'] != list(prescribed_seeds(smoke, training_seeds)):
            raise ValueError('Protocol seed schedule differs from the authorized execution configuration')
        if protocol.get('cohort_config') is not None:
            approval = result['first_rollout_parity']
            verify_first_rollout_approval(approval, out, protocol['cohort_config'])
            if sha(out/'first_rollout_parity.json') != approval['sha256']:
                raise ValueError('Saved first-rollout parity snapshot changed')
            evidence[approval['path']] = approval['sha256']
        final_name = 'policy_01.npz' if smoke else 'policy_05.npz'
        if result.get('final_checkpoint') != final_name: raise ValueError('Only final prescribed checkpoint is eligible')
        checkpoint = out/result['final_checkpoint']
        if sha(SOURCE) != PROFILE.sha256 or sha(ESTIMATOR) != checkpoint_manifest()['estimator_sha256']:
            raise ValueError('Frozen artifact changed')
        for path, digest in protocol['input_sha256'].items():
            check()
            if sha(path) != digest: raise ValueError('Input artifact changed: '+path)
        for relative, digest in protocol['sources'].items():
            check()
            if sha(ROOT/relative) != digest or sha(out/'source'/relative) != digest:
                raise ValueError('Run source or snapshot changed')
        for relative, digest in protocol['prepared_sha256'].items():
            check()
            if sha(out/relative) != digest: raise ValueError('Prepared guard/probe/allocation changed')
        if sha(reference_record['path']) != reference_record['sha256']: raise ValueError('Baseline manifest changed')
        frozen = json.loads(Path(reference_record['path']).read_text())
        for item in frozen['evidence'].values():
            check()
            if sha(item['path']) != item['sha256']: raise ValueError('Baseline snapshot changed')
        reference = frozen['metrics']
        candidate = verify_evaluation_identity(out/'final_development', checkpoint, 2 if smoke else 100, check)
        if not smoke: reasons.extend(gae_screen_reasons(reference, candidate))
        use = json.loads((out/'feature_use.json').read_text())
        for split in ('fit', 'heldout'):
            with np.load(out/f'feature_actions_{split}.npz', allow_pickle=False) as arrays:
                recounted = feature_use_metrics(**{name: arrays[name] for name in ('source', 'actual', 'shuffled', 'zero')})
            if recounted != use[split] or not recounted['zero_estimate_exact'] or (not smoke and not recounted['passed']):
                reasons.append(f'{split}: recurrent-memory feature-use gate failed')
        paths = [SOURCE, ESTIMATOR, checkpoint, out/'protocol.json', out/'result.json',
                 Path(reference_record['path']), out/'final_development/evidence.json',
                 out/'feature_use.json', out/'feature_actions_fit.npz', out/'feature_actions_heldout.npz',
                 out/'guard.npz', out/'probe_fit.npz', out/'probe_heldout.npz',
                 out/'final_probe_fit.npz',out/'final_probe_heldout.npz']
        for path in paths:
            check(); evidence[str(path)] = sha(path)
        for path in sorted(out.glob('policy_*-state.pt')):
            check(); evidence[str(path)] = sha(path)
        for pattern in ('trajectories/*', 'estimated_trajectories/*.npz', 'rollout_*.npz', 'advantages_*.npy'):
            for path in sorted(out.glob(pattern)):
                check()
                if path.is_file(): evidence[str(path)] = sha(path)
        check()
    except (ValueError, KeyError, OSError, TypeError, TrialStopped) as exc:
        reasons.append('Evidence validation failed: '+str(exc))
    decision = dict(decision='software_only' if smoke else
                    'eligible_for_one_fresh_development_screen' if not reasons else 'reject_close_variant',
                    passed=not reasons, reasons=reasons, reference=reference, candidate=candidate,
                    evidence=evidence, training_budget_complete=training_budget_complete(result, smoke, training_seeds),
                    final_pipeline_complete=True, final_screen_status='evaluated',
                    acceptance=False, policy_promoted=False, physical_flight=False)
    write_json(out/'decision.json', decision)
    return decision


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--initialization-report', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    args.cohort_manifest = args.arm = None
    return args


def execution_lease(args):
    return single_trainer(ROOT.parent/'.clearance-training.lock')


def execution_config(args, lease):
    """One declared experiment. No alternate target, parallel lease or seed override."""
    if lease is not None or args.cohort_manifest is not None or args.arm is not None:
        raise ValueError('Only the locked standalone trainable-memory contract is supported')
    return dict(training_seeds=list(prescribed_seeds(args.smoke)), algorithm_seed=ALGORITHM_SEED,
                target_mode='full_horizon', max_total_seconds=600 if args.smoke else 3600,
                memory_fraction=.15, cohort_id=None, cohort_manifest_sha256=None)


def cuda_memory_peaks():
    if not torch.cuda.is_initialized():
        return dict(initialized=False, peak_allocated_bytes=None, peak_reserved_bytes=None)
    return dict(initialized=True, peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
                peak_reserved_bytes=int(torch.cuda.max_memory_reserved(0)))


def verify_first_rollout_approval(record, out, config):
    if sha(record['path']) != record['sha256']:
        raise ValueError('First-rollout parity approval changed')
    approval = json.loads(Path(record['path']).read_text())
    if (approval.get('passed') is not True or approval.get('matched') is not True
            or approval.get('cohort_manifest_sha256') != config['cohort_manifest_sha256']
            or set(approval.get('rollout_sha256', {})) != set(COHORT_ARMS)
            or approval['rollout_sha256'][config['target_mode']] != sha(Path(out)/'rollout_01.npz')):
        raise ValueError('First-rollout parity approval does not bind this matching cohort rollout')


def first_rollout_barrier(lease, out, budget):
    """Require live-parent approval of all raw first rollouts before any fitting."""
    out = Path(out); budget.check()
    if not (out/'rollout_01.npz').is_file() or not (out/'episode_endpoints_01.json').is_file():
        raise ValueError('First rollout and episode endpoints must be saved before the parity barrier')
    try:
        record = lease.wait_first_rollout()
    except Exception as exc:
        raise TrialStopped('First-rollout parity barrier stopped: '+str(exc)) from exc
    budget.check()
    verify_first_rollout_approval(record, out, lease.config)
    shutil.copy2(record['path'], out/'first_rollout_parity.json')
    if sha(out/'first_rollout_parity.json') != record['sha256']:
        raise ValueError('First-rollout parity snapshot changed during copying')
    return record


def fit_value_baseline(value, optimizer, x, target, batches, budget, *, lease=None,
                       iteration=None, out=None, result=None):
    if lease is not None and iteration == 1:
        result['first_rollout_parity'] = first_rollout_barrier(lease, out, budget)
    for _ in range(batches):
        budget.check(); batch = np.random.randint(0, len(x), min(256,len(x)))
        loss = (value(x[batch]).flatten()-target[batch]).square().mean()
        if not torch.isfinite(loss): raise ValueError('Non-finite value loss')
        optimizer.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(value.parameters(),10)
        optimizer.step()


def main(argv=None):
    args = parse_args(argv)
    started = time.monotonic(); out = args.out.resolve()
    with execution_lease(args) as lease:
        config = execution_config(args, lease)
        seeds = prescribed_seeds(args.smoke, config['training_seeds'])
        spec = target_spec(config['target_mode'])
        budget = Budget(out, config['max_total_seconds'], started=started,
                        parent_check=lease.check if lease is not None else lambda: None)
        if not torch.cuda.is_available(): raise RuntimeError('CUDA is required; no CPU training fallback')
        if config['memory_fraction'] is not None:
            torch.cuda.set_per_process_memory_fraction(config['memory_fraction'], device=0)
        torch.set_num_threads(1)
        if torch.get_num_interop_threads() != 1:
            torch.set_num_interop_threads(1)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        out.mkdir(parents=True, exist_ok=False)
        policy = optimizer = value = value_optimizer = env = None
        protocol_digest = reference_record = estimator_digest = None
        result = dict(status='preparing', iterations=[], completed_seeds=[], frozen_unchanged=False,
                      target_mode=config['target_mode'], cohort_id=config['cohort_id'], arm=args.arm,
                      acceptance=False, physical_flight=False)
        try:
            budget.check()
            initialization = verify_initialization(args.initialization_report, budget.check)
            if lease is None:
                allocation = scan_allocation(RUNS, seeds, budget.check)
            else:
                budget.check()
                allocation = dict(kind='Authenticated parent shared FIT reservation', seeds=list(seeds),
                    arm=args.arm, cohort_id=config['cohort_id'],
                    cohort_manifest_path=str(args.cohort_manifest.resolve()),
                    cohort_manifest_sha256=config['cohort_manifest_sha256'],
                    parent_allocation=config['allocation_record'])
                allocation_record = config['allocation_record']
                if sha(allocation_record['path']) != allocation_record['sha256']:
                    raise ValueError('Parent FIT allocation evidence changed')
                shutil.copy2(args.cohort_manifest, out/'cohort_manifest.json')
                shutil.copy2(allocation_record['path'], out/'parent_seed_allocation.json')
            write_json(out/'seed_allocation.json', allocation)
            write_json(out/'status.json', dict(status='running', phase='preparing_public_features', completed_rounds=0))
            reference_record = prepare_gae_screen_reference(
                ROOT.parent/'lab-data/policy-validation-20261006', out/'screen_reference', SOURCE)
            reference_guard = baseline_guard(reference_record)
            guard, selections, inputs = prepare_features(out, budget, smoke=args.smoke)
            inputs[str(SOURCE)] = sha(SOURCE); inputs[str(ESTIMATOR)] = sha(ESTIMATOR)
            inputs[initialization['path']] = initialization['sha256']
            if lease is not None:
                inputs[str(args.cohort_manifest.resolve())] = config['cohort_manifest_sha256']
                inputs[allocation_record['path']] = allocation_record['sha256']
            policy = TrainableMemoryPolicy(SOURCE, ESTIMATOR, device='cuda:0')
            seed_everything(config['algorithm_seed'])  # SB3 loading must not replace the trial RNG seed.
            estimator_digest = tensor_digest(policy.actor.memory)
            optimizer = torch.optim.Adam([p for p in policy.actor.parameters() if p.requires_grad], lr=1e-5)
            value = nn.Sequential(nn.Linear(150,128), nn.Tanh(), nn.Linear(128,128),
                                  nn.Tanh(), nn.Linear(128,1)).cuda()
            value_optimizer = torch.optim.Adam(value.parameters(), lr=3e-4)
            assert all(p.device.type == 'cuda' for p in policy.actor.parameters())
            assert all(p.device.type == 'cuda' for p in value.parameters())
            trust = torch.as_tensor(guard, device='cuda')
            histories = load_public_sequences(out,budget.check)
            with np.load(out/'guard.npz',allow_pickle=False) as z: guard_ids=z['fit_row_ids'].copy()
            with torch.no_grad(): reference_dist = policy.actor.base_actor.get_action_dist_params(trust[:, :85].contiguous())[:2]
            sources = {}
            files = list((ROOT/'src').glob('*.py')) + [Path(__file__).resolve(),
                ROOT/'tools/check_trainable_memory_initialization.py', ROOT/'tools/run_onpolicy_trial.py',
                ROOT/'tools/run_clearance_experiment.py', ROOT/'tools/validate_flight_policies.py',
                ROOT/'tools/run_parallel_estimated_motion.py', ROOT/'tools/parallel_training.py',
                ROOT/'tests/test_trainable_memory_actor_trial.py',
                ROOT/'tests/test_trainable_memory_actor.py', ROOT/'tests/test_trainable_memory_initialization.py',
                ROOT/'experiments/faisal/docs/TRAINABLE_MEMORY_ACTOR_TRIAL_20261008.md']
            if lease is not None:
                files.extend([ROOT/'tools/parallel_training.py', ROOT/'tools/run_parallel_recurrent_memory.py',
                    ROOT/'experiments/faisal/docs/PARALLEL_CREDIT_COMPARISON_20261008.md'])
            for path in files:
                budget.check(); relative = path.relative_to(ROOT); destination = out/'source'/relative
                destination.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, destination)
                sources[relative.as_posix()] = sha(destination)
            protocol = dict(kind='Whole-episode trainable control-memory actor; development only',
                smoke=args.smoke, manifest=checkpoint_manifest(), initialization=initialization,
                training_seeds=list(seeds), iterations=1 if args.smoke else 5,
                episodes_per_iteration=2 if args.smoke else 48, algorithm_seed=config['algorithm_seed'],
                actor_lr=1e-5, value_lr=3e-4, value_batches=2 if args.smoke else 100,
                actor_epochs=2 if args.smoke else 4, episodes_per_actor_batch=4, batch_size=None, **spec,
                source_mean_kl=.01, source_max_kl=.1, boundary_weight=.1, boundary_location='current complete-episode minibatch', max_backtracks=4,
                clip_ratio=.1, max_episode_steps=600, max_total_seconds=config['max_total_seconds'],
                cohort_config=config if lease is not None else None, arm=args.arm,
                cohort_manifest_sha256=config['cohort_manifest_sha256'],
                first_round_barrier='All three raw first rollouts must match before any value or actor update' if lease else None,
                stop='STOP and wall checks every frame/batch;100 consecutive rejections;retention regression',
                guard='8192 original whole-warmup FIT frames,32 for software smoke; reconstructed CPU strict_float32',
                probe='8192 valid rows per FIT/heldout,32 smoke; fixed11591000/1 joint64 shuffle within visibility',
                curriculum='existing alternating approach/full-task FIT resets; unchanged dynamics/rewards/success',
                input_sha256=inputs, sources=sources, reference_record=reference_record,
                prepared_sha256={name:sha(out/name) for name in
                    ('guard.npz','probe_fit.npz','probe_heldout.npz','warmup_episodes.json','seed_allocation.json')
                    + (('cohort_manifest.json','parent_seed_allocation.json') if lease is not None else ())},
                runtime=dict(python=sys.version, packages={name:importlib.metadata.version(name)
                    for name in ('torch','numpy','stable-baselines3','gymnasium')}, cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(0), device='cuda:0', dtype='float32', autocast=False,
                    memory_fraction=config['memory_fraction'],
                    torch_num_threads=torch.get_num_threads(),
                    torch_num_interop_threads=torch.get_num_interop_threads(),
                    cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
                    matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32),
                acceptance=False, physical_flight=False)
            write_json(out/'protocol.json', protocol); protocol_digest = sha(out/'protocol.json')
            save_training_state(out, 'initial', policy, optimizer, value, value_optimizer, dict(stage='zero_head'))
            print('PROTOCOL', str(out), flush=True)
            env = ClearanceTrainingEnv(ApproachCurriculumEnv(level=1, randomise=True, hanger=True),
                                       'clearance', seed_base=seeds[0], trace_dir=out/'trajectories')
            feature_folder = out/'estimated_trajectories'; feature_folder.mkdir()
            accepted = rejected = consecutive = 0

            def trust_check():
                with torch.no_grad():
                    current_trust=selected_current_features(policy,histories['fit'],guard_ids,budget.check)
                    mean, std, _ = policy.actor.get_action_dist_params(current_trust)
                    kl = gaussian_kl(*reference_dist, mean, std)
                    metrics = dict(mean_kl=float(kl.mean()), max_kl=float(kl.max()))
                ok = np.isfinite(list(metrics.values())).all() and metrics['mean_kl'] <= .01 and metrics['max_kl'] <= .1
                return bool(ok), metrics

            for iteration in range(1, protocol['iterations']+1):
                observations = []; actions = []; returns = []; lengths = []; raw_rewards = []; endpoints = []
                offset = (iteration-1)*protocol['episodes_per_iteration']
                for seed in seeds[offset:offset+protocol['episodes_per_iteration']]:
                    budget.check()
                    x, a, rewards, endpoint = collect_episode(policy, env, seed, feature_folder, budget,
                                                             include_endpoint=True)
                    observations.append(x); actions.append(a)
                    returns.extend(episode_targets(rewards, endpoint, config['target_mode']))
                    raw_rewards.extend(rewards); endpoints.append(dict(seed=seed, **endpoint))
                    lengths.append(len(rewards)); result['completed_seeds'].append(seed)
                x = torch.as_tensor(np.concatenate(observations), device='cuda')
                a = torch.as_tensor(np.concatenate(actions), device='cuda')
                target = torch.as_tensor(np.asarray(returns, np.float32), device='cuda')
                with torch.no_grad(): old_log_prob = action_log_prob(policy.actor, x, a).detach()
                np.savez_compressed(out/f'rollout_{iteration:02d}.npz', features150=x.cpu().numpy(),
                    public85=x[:, :85].cpu().numpy(), actions=a.cpu().numpy(), returns=target.cpu().numpy(),
                    old_log_prob=old_log_prob.cpu().numpy(), episode_lengths=lengths, rewards=raw_rewards)
                write_json(out/f'episode_endpoints_{iteration:02d}.json', endpoints)
                fit_value_baseline(value, value_optimizer, x, target, protocol['value_batches'], budget,
                                   lease=lease, iteration=iteration, out=out, result=result)
                with torch.no_grad():
                    frozen_values = value(x).flatten()
                    raw_advantage, advantage = actor_advantages(target, frozen_values, raw_rewards, lengths,
                                                               config['target_mode'])
                np.savez_compressed(out/f'advantage_inputs_{iteration:02d}.npz',
                    rewards=np.asarray(raw_rewards,np.float32), values=frozen_values.cpu().numpy(),
                    value_targets=target.cpu().numpy(), episode_lengths=lengths,
                    raw_advantages=raw_advantage.cpu().numpy(), target_mode=config['target_mode'],
                    gamma=spec['gamma'], gae_lambda=[] if spec['gae_lambda'] is None else [spec['gae_lambda']])
                np.save(out/f'advantages_{iteration:02d}.npy', advantage.cpu().numpy())
                # Each new likelihood recomputes memory from complete public episodes.
                offsets=np.r_[0,np.cumsum(lengths)]
                public_episodes=[obs[:,:85].copy() for obs in observations]
                for epoch in range(protocol['actor_epochs']):
                    order=np.random.permutation(len(lengths))
                    for start in range(0,len(order),4):
                        budget.check(); policy.actor.train()
                        ids=order[start:start+4]
                        batch=np.concatenate([np.arange(offsets[i],offsets[i+1]) for i in ids])
                        public,lens=episode_batch([public_episodes[i] for i in ids],policy.device)
                        current=policy.actor.sequence_features(public,lens)
                        log=action_log_prob(policy.actor,current,a[batch])
                        mean,std,_=policy.actor.get_action_dist_params(current)
                        with torch.no_grad(): rm,rs,_=policy.actor.base_actor.get_action_dist_params(current[:,:85].contiguous())
                        kl=gaussian_kl(rm,rs,mean,std)
                        loss=clipped_loss(log,old_log_prob[batch],advantage[batch])+kl.mean()+.1*boundary_penalty(kl,.01,.1)
                        if not torch.isfinite(loss):raise ValueError('Nonfinite recurrent actor loss')
                        ok,drift=guarded_optimizer_step(policy.actor,optimizer,loss,trust_check,max_backtracks=4)
                        accepted+=int(ok);rejected+=int(not ok);consecutive=0 if ok else consecutive+1
                        if consecutive>=100:raise TrialStopped('100 consecutive rejected proposals')
                assert_frozen(policy,estimator_digest)
                checkpoint = save_training_state(out,f'policy_{iteration:02d}',policy,optimizer,value,value_optimizer,
                                                  dict(iteration=iteration,completed_seeds=result['completed_seeds']))
                result['completed_training_rounds'] = iteration
                result['final_checkpoint'] = checkpoint.name
                metrics = evaluate_head(checkpoint,out/f'guard_{iteration:02d}',2 if args.smoke else 25,budget)
                failures = [] if args.smoke else retention_failures(reference_guard,metrics)
                ok, drift = trust_check()
                if not ok: raise ValueError('Source KL exceeded hard bounds')
                record = dict(iteration=iteration,episodes=len(lengths),transitions=len(x),accepted=accepted,rejected=rejected,
                              metrics=metrics,failures=failures,diagnostics=drift)
                result['iterations'].append(record); result['status']='running'
                write_json(out/'status.json', dict(status='running', phase='round_complete', completed_rounds=iteration))
                write_json(out/'result.json',result); print('ROUND',json.dumps(record),flush=True)
                if failures: raise TrialStopped('Retention regression')
            result['final_development'] = evaluate_head(checkpoint,out/'final_development',2 if args.smoke else 100,budget)
            with np.load(checkpoint, allow_pickle=False) as saved:
                if not np.array_equal(saved['residual_weight'],policy.actor.residual_weight.detach().cpu().numpy()):
                    raise ValueError('Final probe actor differs from saved/evaluated candidate')
                for key,value in policy.actor.memory.state_dict().items():
                    if not np.array_equal(saved['memory.'+key],value.detach().cpu().numpy()):
                        raise ValueError('Final probe memory differs from saved/evaluated candidate')
            for split,selection in selections.items():
                selection['features']=selected_current_features(policy,histories[split],selection['row_ids'],budget.check).cpu().numpy()
                np.savez_compressed(out/f'final_probe_{split}.npz',**selection)
            result['feature_use'] = feature_probe(policy,selections,out,budget)
            result['learned_memory_changed'] = tensor_digest(policy.actor.memory)!=estimator_digest
            assert_frozen(policy,estimator_digest)
            result['frozen_unchanged'] = sha(SOURCE)==PROFILE.sha256 and sha(ESTIMATOR)==checkpoint_manifest()['estimator_sha256']
            result['status']='complete'; budget.check()
        except TrialStopped as exc:
            result['status']='stopped'; result['error']=str(exc)
        except KeyboardInterrupt:
            result['status']='stopped'; result['error']='Keyboard interrupt requested'
        except Exception as exc:
            result['status']='failed'; result['error']=repr(exc)
            (out/'error.txt').write_text(traceback.format_exc())
        finally:
            if env is not None: env.close()
            if policy is not None and optimizer is not None and value_optimizer is not None:
                try:
                    if result['status'] != 'complete':
                        save_training_state(out,'bounded_partial',policy,optimizer,value,value_optimizer,
                                            dict(status=result['status'],completed_seeds=result['completed_seeds']))
                except Exception as exc:
                    result['status']='failed'; result['integrity_error']=repr(exc); result['frozen_unchanged']=False
                record_frozen_integrity(result,policy,estimator_digest)
            result['training_budget_complete'] = training_budget_complete(result,args.smoke,seeds)
            result['final_pipeline_complete'] = pipeline_complete(result)
            try:
                result['cuda_memory'] = cuda_memory_peaks()
            except Exception as exc:
                result['cuda_memory'] = dict(measurement_error=repr(exc))
            result['elapsed_seconds']=time.monotonic()-started
            write_json(out/'result.json',result)
            write_json(out/'status.json', dict(status=result['status'], completed_rounds=result.get('completed_training_rounds',0), error=result.get('error')))
            decision=final_screen(out,result,reference_record,protocol_digest,smoke=args.smoke,
                                  check=budget.check,training_seeds=seeds)
            print('DECISION',json.dumps(decision),flush=True)


if __name__ == '__main__':
    main()
