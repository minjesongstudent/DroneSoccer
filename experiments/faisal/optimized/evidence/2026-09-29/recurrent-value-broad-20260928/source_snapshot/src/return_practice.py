"""Authentic mid-flight return practice from saved-policy simulator flights.

Libraries contain trusted local pickle snapshots of the complete wrapped
environment, including observation history, command memory, dynamics and RNG.
Never load libraries from untrusted sources. These are reset states, not
demonstrations: the training actor chooses every continuation action. Rewards
and collisions remain those of the lap task; the practice cap is a truncation.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import pickle
import zipfile

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import SAC

from src.curriculum import REWARD_VERSION
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation, OBSERVATION_DIM, VERSION as OBSERVATION_VERSION
from src.sim import CRASHES, LAPS_MAX_STEPS


VERSION = 'authentic-policy-return-continuations-v1'
SEED_RANGE = (2_000_000, 3_000_000)
SEED_RANGES = {'train': SEED_RANGE, 'development': (8_000_000, 9_000_000)}
ALLOWED_PHASES = ('return_side', 'return_front')
STATE_SOURCE_FILES = ('sim.py', 'curriculum.py', 'guided_curriculum.py', 'contract.py',
                      'params.py', 'mission_observation.py', 'observation_memory.py')


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _code_hashes():
    return {name: _sha((Path(__file__).parent/name).read_bytes()) for name in STATE_SOURCE_FILES}


def _pack_snapshot(env, observation, info):
    if not isinstance(env, MissionObservation) or not isinstance(env.unwrapped, GuidedLessonEnv):
        raise ValueError('Snapshots require the complete MissionObservation-wrapped guided environment.')
    if env.single_pass or env.unwrapped.guided_stage != 4:
        raise ValueError('Return snapshots must retain the stage-4 lap task.')
    observation = np.asarray(observation, dtype=np.float32)
    if observation.shape != (OBSERVATION_DIM,) or observation[-1] != 0:
        raise ValueError('Return snapshots require the unchanged 100-value lap observation.')
    return pickle.dumps({'environment': env, 'observation': observation.copy(),
                         'info': deepcopy(info)}, protocol=pickle.HIGHEST_PROTOCOL)


def _restore_snapshot(payload):
    snapshot = pickle.loads(payload)
    env, observation = snapshot['environment'], snapshot['observation']
    if not isinstance(env, MissionObservation) or not isinstance(env.unwrapped, GuidedLessonEnv):
        raise ValueError('Snapshot environment has an incompatible wrapper stack.')
    if env.single_pass or env.unwrapped.guided_stage != 4 or observation.shape != (100,):
        raise ValueError('Snapshot must contain the unchanged 100-value lap task.')
    # Reconstruct the same observation without advancing any history or RNG.
    reconstructed = env._append(env.env.memory.encode(env.unwrapped._policy_input()))
    if not np.array_equal(reconstructed, observation) or observation[-1] != 0:
        raise ValueError('Snapshot observation and preserved history disagree.')
    return snapshot


def generate_return_library(checkpoint_path, output_dir, *, episodes=64, seed_start=None,
                            phases=ALLOWED_PHASES, horizon=200, device='cuda', split='train'):
    """Generate deterministic reset snapshots using only a saved policy.

    Run collection separately from the training process. Seeds occupy a
    dedicated training-library interval disjoint from 5M/7M evaluation seeds.
    Capture only the first genuine entry to each requested phase per flight,
    after a clean passage, with enough original episode time for the practice.
    Missing later phases remain missing; they are never synthesized.
    """
    if split not in SEED_RANGES:
        raise ValueError('split must be train or development.')
    limits = SEED_RANGES[split]
    seed_start = limits[0] if seed_start is None else seed_start
    if int(episodes) != episodes or episodes <= 0:
        raise ValueError('episodes must be a positive integer')
    if (int(seed_start) != seed_start or seed_start < limits[0]
            or seed_start+episodes > limits[1]):
        raise ValueError(f'Collection seeds must stay in the dedicated {split} range {limits}.')
    if int(horizon) != horizon or not 1 <= horizon < LAPS_MAX_STEPS:
        raise ValueError('horizon must be a positive integer shorter than the original episode.')
    phases = tuple(phases)
    if not phases or len(phases) != len(set(phases)) or any(p not in ALLOWED_PHASES for p in phases):
        raise ValueError('Request unique return_side and/or return_front phases.')
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_hash = _sha(checkpoint_path.read_bytes())
    code_hashes = _code_hashes()
    with zipfile.ZipFile(checkpoint_path) as archive:
        anchored = 'reference_actor.pth' in archive.namelist()
    if anchored:
        from src.anchored_policy import AnchoredSAC
        model = AnchoredSAC.load(checkpoint_path, device=device)
    else:
        model = SAC.load(checkpoint_path, device=device)
    if model.observation_space.shape != (100,) or model.action_space.shape != (4,):
        raise ValueError('Collection requires a saved policy using the current 100-value observation contract.')
    if str(device).startswith('cuda') and next(model.policy.parameters()).device.type != 'cuda':
        raise RuntimeError('CUDA was requested for policy collection; no CPU fallback is permitted.')
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    env = MissionObservation(GuidedLessonEnv(stage=4), single_pass=False)
    manifest = {'version': VERSION, 'split': split, 'reward_version': REWARD_VERSION,
                'observation_version': OBSERVATION_VERSION, 'observation_dim': 100,
                'source_checkpoint': str(checkpoint_path), 'source_sha256': checkpoint_hash,
                'source_steps': int(model.num_timesteps), 'policy_mode': 'deterministic',
                'seed_start': int(seed_start), 'seed_end': int(seed_start+episodes-1),
                'attempted_episodes': int(episodes), 'requested_phases': phases,
                'minimum_remaining_steps': int(horizon), 'code_sha256': code_hashes,
                'geometry': {name: asdict(getattr(env.unwrapped, name))
                             for name in ('cage', 'drone', 'hoop', 'camera', 'pad')},
                'snapshots': [], 'phase_counts': {}, 'rejected_entries': {}, 'episode_outcomes': {}}
    rejections, outcomes = Counter(), Counter()
    try:
        with torch.inference_mode():
            for seed in range(int(seed_start), int(seed_start+episodes)):
                observation, _ = env.reset(seed=seed)
                seen = set()
                previous_phase = env.unwrapped._lap_phase
                while True:
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, info = env.step(action)
                    phase = env.unwrapped._lap_phase
                    if phase in phases and phase != previous_phase and phase not in seen:
                        seen.add(phase)
                        if terminated or truncated:
                            rejections['terminal_entry'] += 1
                        elif env.unwrapped.passes < 1:
                            rejections['no_prior_clean_passage'] += 1
                        elif LAPS_MAX_STEPS-env.unwrapped.steps < horizon:
                            rejections['insufficient_remaining_steps'] += 1
                        else:
                            payload = _pack_snapshot(env, observation, info)
                            identifier = f'snapshot-{len(manifest["snapshots"]):04d}'
                            filename = identifier+'.pkl'
                            (output_dir/filename).write_bytes(payload)
                            manifest['snapshots'].append({
                                'id': identifier, 'file': filename, 'sha256': _sha(payload),
                                'phase': phase, 'seed': seed, 'source_step': int(env.unwrapped.steps),
                                'historical_passes': int(env.unwrapped.passes)})
                    previous_phase = phase
                    if terminated or truncated:
                        outcomes[info['outcome']] += 1
                        break
    finally:
        env.close()
    if _sha(checkpoint_path.read_bytes()) != checkpoint_hash or _code_hashes() != code_hashes:
        raise RuntimeError('Checkpoint or environment source changed during collection; discard this incomplete library.')
    manifest['phase_counts'] = dict(Counter(s['phase'] for s in manifest['snapshots']))
    manifest['rejected_entries'] = dict(rejections)
    manifest['episode_outcomes'] = dict(outcomes)
    (output_dir/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


class ReturnSnapshotLibrary:
    """Validated immutable local payloads; every restore produces a fresh clone."""
    version = VERSION
    observation_version = OBSERVATION_VERSION
    observation_dim = 100
    code_hashes = staticmethod(_code_hashes)
    restore_payload = staticmethod(_restore_snapshot)

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        manifest_bytes = (self.directory/'manifest.json').read_bytes()
        self.bank_sha256 = _sha(manifest_bytes)
        self.manifest = json.loads(manifest_bytes.decode('utf-8'))
        for key, expected in (('version', self.version), ('reward_version', REWARD_VERSION),
                              ('observation_version', self.observation_version), ('observation_dim', self.observation_dim)):
            if self.manifest.get(key) != expected:
                raise ValueError(f'Incompatible snapshot library {key}; regenerate from actual policy flights.')
        if self.manifest.get('code_sha256') != self.code_hashes():
            raise ValueError('Snapshot environment code differs; regenerate rather than importing incompatible state.')
        split = self.manifest.get('split')
        if split not in SEED_RANGES:
            raise ValueError('Snapshot library requires an explicit train/development split.')
        low, high = SEED_RANGES[split]
        if not low <= self.manifest['seed_start'] <= self.manifest['seed_end'] < high:
            raise ValueError('Snapshot library seeds do not match its split.')
        horizon = self.manifest['minimum_remaining_steps']
        if int(horizon) != horizon or not 1 <= horizon < LAPS_MAX_STEPS:
            raise ValueError('Invalid guaranteed continuation horizon.')
        self.records = self.manifest['snapshots']
        if not self.records:
            raise ValueError('No genuine return states were reached; an empty library cannot supply practice.')
        self.payloads = {}
        for record in self.records:
            filename = record['file']
            if Path(filename).name != filename or record['id'] in self.payloads:
                raise ValueError('Invalid or duplicate snapshot identifier/path.')
            payload = (self.directory/filename).read_bytes()
            if _sha(payload) != record['sha256']:
                raise ValueError('Snapshot checksum mismatch.')
            snapshot = self.restore_payload(payload)
            base = snapshot['environment'].unwrapped
            if (base._lap_phase != record['phase'] or base.steps != record['source_step']
                    or base.passes != record['historical_passes'] or base.passes < 1
                    or base._lap_phase not in ALLOWED_PHASES):
                raise ValueError('Snapshot provenance does not match its preserved simulator state.')
            if (not self.manifest['seed_start'] <= record['seed'] <= self.manifest['seed_end']
                    or base.np_random_seed != record['seed']):
                raise ValueError('Snapshot seed does not match its captured RNG provenance.')
            if LAPS_MAX_STEPS-base.steps < horizon:
                raise ValueError('Snapshot cannot provide its guaranteed continuation horizon.')
            if any(asdict(getattr(base, name)) != value for name, value in self.manifest['geometry'].items()):
                raise ValueError('Snapshot geometry differs from its collection manifest.')
            self.payloads[record['id']] = payload
        self.phases = tuple(sorted({r['phase'] for r in self.records}))

    def restore(self, identifier):
        return self.restore_payload(self.payloads[identifier])


class ReturnPracticeEnv(gym.Wrapper):
    """Continue authentic flights with unchanged rewards and a local time cap.

    Info labels explicitly exclude these trials from full-mission evaluation.
    ``steps`` and ``passes`` count only the continuation; historical quantities
    remain separately available. The preserved environment's physical clock,
    action delay, velocity, camera history and RNG are never reset or edited.
    """
    def __init__(self, library, horizon=200):
        self.library = library if isinstance(library, ReturnSnapshotLibrary) else ReturnSnapshotLibrary(library)
        if int(horizon) != horizon or not 1 <= horizon <= self.library.manifest['minimum_remaining_steps']:
            raise ValueError('Practice horizon exceeds the library\'s guaranteed remaining episode time.')
        self.horizon = int(horizon)
        self._selection_rng = np.random.default_rng()
        self.elapsed = 0
        self.finished = True
        first = self.library.restore(self.library.records[0]['id'])
        super().__init__(first['environment'])

    def _info(self, info):
        info = dict(info)
        info['source_stage'] = info.get('stage', 4)
        info.update({'return_practice': True, 'practice_label': 'Authentic return: '+self.record['phase'],
                     'full_mission_success': False, 'stage': 4, 'focus_stage': 4,
                     'single_pass_task': False, 'source_step': self.record['source_step'],
                     'historical_passes': self.record['historical_passes'],
                     'physical_steps': int(self.env.unwrapped.steps), 'steps': self.elapsed,
                     'passes': int(self.env.unwrapped.passes-self.record['historical_passes']),
                     'snapshot_id': self.record['id'], 'snapshot_seed': self.record['seed'],
                     'snapshot_source_sha256': self.library.manifest['source_sha256']})
        info['practice_success'] = info['passes'] >= 1 and info['outcome'] not in CRASHES
        return info

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._selection_rng = np.random.default_rng(seed)
        options = options or {}
        if set(options)-{'snapshot_id', 'phase'} or ('snapshot_id' in options and 'phase' in options):
            raise ValueError('Select a snapshot_id or a phase, not both.')
        if 'snapshot_id' in options:
            candidates = [r for r in self.library.records if r['id'] == options['snapshot_id']]
        else:
            phase = options.get('phase')
            if phase is None:
                phase = self.library.phases[int(self._selection_rng.integers(len(self.library.phases)))]
            candidates = [r for r in self.library.records if r['phase'] == phase]
        if not candidates:
            raise ValueError('Requested return state was never captured from the source policy.')
        self.record = candidates[int(self._selection_rng.integers(len(candidates)))]
        snapshot = self.library.restore(self.record['id'])
        self.env = snapshot['environment']
        self.elapsed, self.finished = 0, False
        info = dict(snapshot['info'])
        info['outcome'] = 'running'
        return snapshot['observation'].copy(), self._info(info)

    def step(self, action):
        if self.finished:
            raise RuntimeError('Reset return practice before stepping a new continuation.')
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.elapsed += 1
        if not terminated and self.elapsed >= self.horizon:
            truncated = True
            info = dict(info)
            info['outcome'] = 'timeout'
        self.finished = bool(terminated or truncated)
        return observation, reward, terminated, truncated, self._info(info)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Collect authentic return reset states; no training.')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=64)
    parser.add_argument('--split', choices=tuple(SEED_RANGES), default='train')
    parser.add_argument('--seed-start', type=int)
    parser.add_argument('--horizon', type=int, default=200)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    result = generate_return_library(args.checkpoint, args.output, episodes=args.episodes,
                                    seed_start=args.seed_start, horizon=args.horizon, device=args.device,
                                    split=args.split)
    print(json.dumps({key: result[key] for key in ('source_sha256', 'phase_counts', 'rejected_entries')}))
