"""Fixed development evaluation of authentic, camera-only return continuations.

This diagnostic task is separate from the 600-step full mission. Geometry is
read only by the scorer; the policy receives the unchanged wrapped camera and
issued-command observation. Historical passes in a snapshot earn no credit.
"""
from collections import Counter
import hashlib
import json
from numbers import Integral

import numpy as np
import torch

from src.curriculum import REWARD_VERSION
from src.mission_observation import VERSION as OBSERVATION_VERSION
from src.return_practice import ReturnPracticeEnv, ReturnSnapshotLibrary
from src.sim import CRASHES, LAPS_MAX_STEPS


VERSION = 'development-return-continuations-v1'
HORIZON = 200
DEVELOPMENT_FLIGHT_SEEDS = (8_000_000, 9_000_000)
SELECTION_SEEDS = (8_500_000, 9_000_000)


def _fingerprint(model):
    """Detect accidental learning while permitting predict's eval-mode switch."""
    counters = {name: getattr(model, name) for name in
                ('num_timesteps', '_n_updates', 'actor_updates', 'actor_attempts',
                 'critic_calibration_updates') if hasattr(model, name)}
    digest = hashlib.sha256()
    policy = getattr(model, 'policy', None)
    if policy is not None and hasattr(policy, 'state_dict'):
        for name, value in policy.state_dict().items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return counters, digest.hexdigest()


def _validate_library(library):
    raw = (library.directory/'manifest.json').read_bytes()
    manifest = json.loads(raw)
    if manifest != library.manifest or manifest.get('snapshots') != library.records:
        raise ValueError('In-memory library differs from its frozen manifest.')
    if manifest.get('split') != 'development':
        raise ValueError('Return evaluation requires a development library, never training snapshots.')
    low, high = DEVELOPMENT_FLIGHT_SEEDS
    if not low <= manifest.get('seed_start', -1) <= manifest.get('seed_end', -1) < high:
        raise ValueError('Development source flights must use the reserved 8M seed range.')
    if manifest.get('minimum_remaining_steps', 0) < HORIZON:
        raise ValueError('Development library must guarantee all 200 continuation steps.')
    if (manifest.get('reward_version') != REWARD_VERSION
            or manifest.get('observation_version') != getattr(library,'source_observation_version',library.observation_version)):
        raise ValueError('Development library reward/observation contract differs.')
    if not library.records:
        raise ValueError('Development library cannot be empty.')
    for record in library.records:
        if (not low <= record.get('seed', -1) < high
                or not 0 <= record.get('source_step', -1) <= LAPS_MAX_STEPS-HORIZON
                or record.get('historical_passes', 0) < 1):
            raise ValueError('Development snapshot provenance cannot support this continuation protocol.')
    return manifest, raw, hashlib.sha256(raw).hexdigest()


def _snapshot_schedule(records, episodes, seed):
    """Freeze an approximately phase-balanced set of distinct source flights.

    Each source flight may offer a side-entry and a front-entry snapshot. Take
    sources exclusive to the scarcer phase before shared ones, preserving the
    other phase's options. This handles the two supported return phases without
    letting duplicate snapshots inflate the number of independent starts.
    """
    by_phase = {}
    for record in records:
        phase = record['phase']
        if phase not in ('return_side', 'return_front'):
            raise ValueError('Unsupported development snapshot phase.')
        by_phase.setdefault(phase, {}).setdefault(record['seed'], []).append(record)
    unique_sources = {record['seed'] for record in records}
    if len(unique_sources) < episodes:
        raise ValueError(f'Return evaluation needs {episodes} distinct source flights; development bank has {len(unique_sources)}. Collect more 8M flights.')
    rng = np.random.default_rng(seed)
    phases = sorted(by_phase, key=lambda phase: (len(by_phase[phase]), phase))
    chosen = []
    if len(phases) == 1:
        selected = rng.permutation(sorted(by_phase[phases[0]]))[:episodes]
        chosen = [(phases[0], int(source)) for source in selected]
    else:
        first, second = phases
        first_sources, second_sources = set(by_phase[first]), set(by_phase[second])
        first_quota = min(len(first_sources), (episodes+1)//2)
        first_quota = max(first_quota, episodes-len(second_sources))
        exclusive = list(rng.permutation(sorted(first_sources-second_sources)))
        shared = list(rng.permutation(sorted(first_sources & second_sources)))
        first_selected = (exclusive+shared)[:first_quota]
        remaining_second = sorted(second_sources-set(first_selected))
        second_selected = rng.permutation(remaining_second)[:episodes-first_quota]
        chosen = [(first, int(source)) for source in first_selected]
        chosen += [(second, int(source)) for source in second_selected]
    if len(chosen) != episodes or len({source for _, source in chosen}) != episodes:
        raise RuntimeError('Could not allocate the fixed distinct-flight evaluation schedule.')
    rng.shuffle(chosen)
    schedule = []
    for phase, source in chosen:
        candidates = sorted(by_phase[phase][source], key=lambda record: record['id'])
        schedule.append(candidates[int(rng.integers(len(candidates)))])
    return schedule


def evaluate_return_practice(model, library, episodes=50, seed_start=8_500_000, progress=None):
    """Return ``(suite, protocol)`` for the fixed 200-step development task.

    The manifest and seed_start determine one approximately phase-balanced
    snapshot schedule with distinct source flights. Each selection seed then
    restores its explicit snapshot ID; checkpoints cannot sample a fresh set.
    ``progress`` follows the normal evaluator's (suite_name, completed, total)
    callback. Neither learning nor policy/environment steering is performed.
    """
    if isinstance(episodes, bool) or not isinstance(episodes, Integral) or episodes <= 0:
        raise ValueError('episodes must be a positive integer')
    if (isinstance(seed_start, bool) or not isinstance(seed_start, Integral)
            or seed_start < SELECTION_SEEDS[0] or seed_start+episodes > SELECTION_SEEDS[1]):
        raise ValueError('Return selection seeds must stay in [8500000, 9000000), disjoint from final 7M tests.')
    library = library if isinstance(library, ReturnSnapshotLibrary) else ReturnSnapshotLibrary(library)
    manifest, manifest_bytes, manifest_sha = _validate_library(library)
    schedule = _snapshot_schedule(library.records, int(episodes), int(seed_start))
    model_before = _fingerprint(model)
    env = ReturnPracticeEnv(library, horizon=HORIZON)
    records = []
    try:
        with torch.inference_mode():
            for index in range(int(episodes)):
                selection_seed = int(seed_start+index)
                selected = schedule[index]
                observation, initial = env.reset(seed=selection_seed, options={'snapshot_id': selected['id']})
                if initial['snapshot_id'] != selected['id'] or initial['snapshot_seed'] != selected['seed']:
                    raise RuntimeError('Return reset did not restore its prescribed development snapshot.')
                base = env.unwrapped
                historical_passes = int(base.passes)
                if historical_passes < 1 or float((base.pos-base.hoop_centre) @ base.hoop_normal) <= 0:
                    raise ValueError('Return snapshots must start behind the ring after a real prior pass.')
                pending_front_clearance = False
                front_reached_step = None
                outside_crossing_steps = []
                crashed = False
                total_reward = 0.
                for elapsed in range(1, HORIZON+1):
                    previous_position = base.pos.copy()
                    action, _ = model.predict(observation, deterministic=True)
                    observation, reward, terminated, truncated, info = env.step(action)
                    total_reward += float(reward)
                    collision = info['outcome'] in CRASHES
                    crashed = crashed or collision
                    # Score the physical route independently of reward phases.
                    crossing = base._crossing(previous_position)
                    if crossing is not None:
                        direction, inside, _ = crossing
                        if direction > 0:
                            pending_front_clearance = False
                        elif not inside and not collision:
                            outside_crossing_steps.append(elapsed)
                            pending_front_clearance = True
                    front_clear = float((base.pos-base.hoop_centre) @ base.hoop_normal) < -base._clearance_reach()
                    if pending_front_clearance and front_clear and not collision:
                        if front_reached_step is None:
                            front_reached_step = elapsed
                        pending_front_clearance = False
                    if terminated or truncated:
                        break
                new_passes = int(base.passes)-historical_passes
                if new_passes < 0:
                    raise RuntimeError('Snapshot pass accounting went backwards.')
                survived = bool(elapsed == HORIZON and truncated and not terminated
                                and not crashed and info['outcome'] == 'timeout')
                front_reached = front_reached_step is not None
                records.append({
                    'seed': selection_seed, 'snapshot_id': initial['snapshot_id'],
                    'snapshot_seed': int(initial['snapshot_seed']),
                    'source_phase': selected['phase'],
                    'source_step': int(initial['source_step']),
                    'historical_passes': historical_passes,
                    'passes': new_passes, 'new_clean_passes': new_passes,
                    'steps': elapsed, 'outcome': info['outcome'], 'reward': total_reward,
                    'crashed': bool(crashed), 'survived_horizon': survived,
                    'front_reached': front_reached, 'front_reached_step': front_reached_step,
                    'outside_back_to_front_crossing_steps': outside_crossing_steps,
                    'front_reached_then_crash': bool(front_reached and crashed),
                    'safe_front_return': bool(front_reached and survived),
                    'successful': bool(new_passes >= 1 and survived),
                })
                if progress:
                    progress('return_practice', index+1, int(episodes))
    finally:
        env.close()
        if (library.directory/'manifest.json').read_bytes() != manifest_bytes:
            raise RuntimeError('Development library manifest changed during evaluation.')
        if _fingerprint(model) != model_before:
            raise RuntimeError('Model weights or training counters changed during evaluation.')
    successes = sum(r['successful'] for r in records)
    safe_returns = sum(r['safe_front_return'] for r in records)
    crashes = sum(r['crashed'] for r in records)
    suite = {
        'episodes': int(episodes), 'successful_episodes': successes, 'success_rate': successes/episodes,
        'safe_front_return_episodes': safe_returns, 'safe_front_return_rate': safe_returns/episodes,
        'front_reached_episodes': sum(r['front_reached'] for r in records),
        'front_reached_then_crash_episodes': sum(r['front_reached_then_crash'] for r in records),
        'crash_episodes': crashes, 'crash_rate': crashes/episodes,
        'episodes_with_pass_rate': sum(r['new_clean_passes'] >= 1 for r in records)/episodes,
        'total_passes': sum(r['new_clean_passes'] for r in records),
        'unique_snapshots': len({r['snapshot_id'] for r in records}),
        'unique_source_flights': len({r['snapshot_seed'] for r in records}),
        'outcomes': dict(Counter(r['outcome'] for r in records)), 'records': records,
    }
    protocol = {
        'id': VERSION, 'environment': 'ReturnPracticeEnv', 'split': 'development',
        'bank_sha256': manifest_sha, 'bank_manifest_sha256': manifest_sha,
        'source_checkpoint_sha256': manifest['source_sha256'],
        'source_seed_start': manifest['seed_start'], 'source_seed_end': manifest['seed_end'],
        'reward_version': REWARD_VERSION, 'observation_version': library.observation_version,
        'max_steps': HORIZON, 'episodes': int(episodes), 'seed_start': int(seed_start),
        'seed_end': int(seed_start+episodes-1), 'deterministic': True,
        'seeds': [r['seed'] for r in records],
        'success': 'at least one new fully cleared pass and no crash through all 200 continuation steps',
        'safe_front_return': 'outside back-to-front plane crossing, whole body clears front, and no crash through all 200 steps',
        'historical_passes_counted': False, 'full_mission_test': False,
        'selection_method': 'fixed approximately phase-balanced snapshots from distinct source flights',
        'unique_snapshots': suite['unique_snapshots'], 'unique_source_flights': suite['unique_source_flights'],
        'phase_counts': dict(Counter(r['source_phase'] for r in records)),
        'snapshot_selection': [{'seed': r['seed'], 'snapshot_id': r['snapshot_id'],
                                'source_seed': r['snapshot_seed'], 'phase': r['source_phase']} for r in records],
    }
    return suite, protocol
