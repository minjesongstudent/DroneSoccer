"""Isolated frozen-source worker; not an acceptance or hardware entry point."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import sys


SUITES = ('original_centered', 'varied_approach', 'full_task', 'full_task_delay_200ms')


def verify_manifest(root, manifest):
    if (manifest.get('kind') != 'legacy compatibility DEVELOPMENT ONLY'
            or manifest.get('training') is not False
            or manifest.get('acceptance_attempted') is not False):
        raise ValueError('Only frozen development compatibility manifests are supported')
    episodes, seed = manifest['episodes_per_suite'], manifest['seed_start']
    if (type(episodes) is not int or not 1 <= episodes <= 200 or type(seed) is not int
            or seed < 10_100_000 or seed + episodes - 1 > 10_100_999):
        raise ValueError('Reserved/fixed seeds blocked; dedicated development range required')
    for relative, digest in manifest['frozen_sha256'].items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root / 'source'):
            raise ValueError('Frozen path outside source directory')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen checksum mismatch: {relative}')
    # Camera() is passed explicitly; also prevent an ambient calibration from
    # entering this frozen legacy tree unnoticed.
    if (root / 'source/legacy/calib.npz').exists():
        raise ValueError('Ambient calibration is prohibited in legacy compatibility evaluation')


def make_environment(name):
    from src.curriculum import LessonEnv
    from src.params import Camera, Drone
    from src.sim import HoopEnv
    if name == 'original_centered':
        return LessonEnv(stage=0, camera=Camera())
    if name == 'varied_approach':
        return LessonEnv(stage=1, camera=Camera())
    if name == 'full_task':
        return HoopEnv(level=1., camera=Camera())
    if name == 'full_task_delay_200ms':
        return HoopEnv(level=1., camera=Camera(), drone=Drone(delay_steps=2))
    raise ValueError('Unknown legacy suite')


def episode(policy, name, seed, trace_path):
    import numpy as np
    from src import contract
    from src.sim import CRASHES
    env = make_environment(name)
    try:
        public, _ = env.reset(seed=seed)
        policy.reset()
        previous = np.zeros(4, np.float32)
        initial = dict(position=env.pos.tolist(), heading=float(env.heading),
                       hoop_centre=env.hoop_centre.tolist(), hoop_normal=env.hoop_normal.tolist(),
                       outer_radii=[float(np.linalg.norm(env._hoop_h)),
                                    float(np.linalg.norm(env._hoop_v))],
                       geometry={key: asdict(getattr(env, key))
                                 for key in ('cage', 'drone', 'hoop', 'camera', 'pad')})
        data = {key: [] for key in ('public_camera', 'actor_observations', 'actions',
                                    'positions', 'headings', 'passes')}
        phases_after_pass = set()
        first_pass_step = None
        while True:
            # Legacy native observation is84 (72camera+12issued commands).
            # Build group85 independently: never feed the old task/memory wrappers.
            end = contract.OBS_STACK * contract.OBS_DIM
            newest = public[end-contract.OBS_DIM:end]
            actor_obs, action = policy.predict(newest, previous)
            data['public_camera'].append(newest.copy())
            data['actor_observations'].append(actor_obs.copy())
            data['actions'].append(action.copy())
            public, _, terminated, truncated, info = env.step(action)
            previous = action.copy()
            data['positions'].append(env.pos.copy())
            data['headings'].append(float(env.heading))
            data['passes'].append(int(info['passes']))
            if info['passes']:
                phases_after_pass.add(info.get('lap_phase', ''))
                if first_pass_step is None:
                    first_pass_step = int(info['steps'])
            if terminated or truncated:
                break
        np.savez_compressed(trace_path, **{key: np.asarray(value) for key, value in data.items()})
        single = name in ('original_centered', 'varied_approach')
        success = (info['outcome'] == 'passed' if single else
                   info['outcome'] == 'timeout' and info['passes'] >= 2)
        return dict(seed=seed, initial=initial, outcome=info['outcome'],
                    passes=int(info['passes']), steps=int(info['steps']),
                    successful=bool(success), crashed=info['outcome'] in CRASHES,
                    first_pass_step=first_pass_step,
                    returned_to_side='return_front' in phases_after_pass,
                    returned_to_front='align_front' in phases_after_pass)
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    root = args.run.resolve()
    manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
    verify_manifest(root, manifest)
    # This process is launched with -I. No checkout's src can shadow the snapshot.
    sys.path[:0] = [str(root / 'source/legacy'), str(root / 'source/bridge')]
    import numpy as np
    import torch
    from group_bridge.policy_runtime import LearnedPolicy, SPECS
    from src import contract
    if contract.POLICY_INPUT_DIM != 84 or contract.OBS_DIM != 8:
        raise ValueError('Legacy public observation contract changed')
    spec = SPECS[manifest['model_name']]
    if spec.inputs != 85 or spec.sha256 != manifest['model_sha256']:
        raise ValueError('Selected85 group model contract mismatch')
    torch.set_num_threads(1)
    np.random.seed(0)
    torch.manual_seed(0)
    policy = LearnedPolicy(root / 'source/models' / (spec.name + '.zip'), spec)
    output = dict(kind=manifest['kind'], model_sha256=spec.sha256,
                  episodes_per_suite=manifest['episodes_per_suite'], seed_start=manifest['seed_start'],
                  policy_mode='deterministic', acceptance_attempted=False,
                  accepted_in_simulation=False, physical_flight_validated=False,
                  versions={name: importlib.metadata.version(name)
                            for name in ('numpy', 'torch', 'stable-baselines3', 'gymnasium')},
                  protocols={}, suites={})
    for name in SUITES:
        env = make_environment(name)
        output['protocols'][name] = dict(
            id='legacy-compatibility-development-' + name + '-v1',
            source_protocol_id='completion-' + name + '-v1',
            environment=type(env).__name__, randomise=False, hanger=True,
            max_steps=150 if name in SUITES[:2] else 600,
            success='passed' if name in SUITES[:2] else 'timeout and passes >= 2',
            actor_profile='group-selected85; public8 history only',
            geometry={key: asdict(getattr(env, key))
                      for key in ('cage', 'drone', 'hoop', 'camera', 'pad')})
        env.close()
        folder = root / name
        folder.mkdir(exist_ok=False)
        rows = []
        for index in range(manifest['episodes_per_suite']):
            seed = manifest['seed_start'] + index
            row = episode(policy, name, seed, folder / f'{seed}.npz')
            rows.append(row)
            with (folder / 'episodes.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(row, allow_nan=False) + '\n')
            print(name, index + 1, row['outcome'], row['passes'], flush=True)
        count = len(rows)
        output['suites'][name] = dict(
            episodes=count, successful_episodes=sum(r['successful'] for r in rows),
            success_rate=sum(r['successful'] for r in rows)/count,
            crash_rate=sum(r['crashed'] for r in rows)/count,
            total_passes=sum(r['passes'] for r in rows),
            episodes_with_pass_rate=sum(r['passes'] > 0 for r in rows)/count,
            episodes_with_two_passes=sum(r['passes'] >= 2 for r in rows),
            returned_to_side_episodes=sum(r['returned_to_side'] for r in rows),
            returned_to_front_episodes=sum(r['returned_to_front'] for r in rows),
            outcomes=dict(Counter(r['outcome'] for r in rows)))
        (root / 'results.json').write_text(json.dumps(output, indent=2, allow_nan=False), encoding='utf-8')
    verify_manifest(root, manifest)


if __name__ == '__main__':
    main()
