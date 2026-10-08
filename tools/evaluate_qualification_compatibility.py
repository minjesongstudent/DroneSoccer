"""Frozen legacy qualification scenarios with the selected 85-input policy.

This is a development compatibility check, never an acceptance or flight release.
Only the dedicated 10,100,000..10,100,999 development range is available. The
original 5M validation and 7M acceptance cohorts are deliberately inaccessible.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = ROOT.parent / 'DroneSoccer-Optimized'
MODEL_NAME = 'seenflag-s1-level1'
MODEL_SHA256 = '8c0711b7d139ca7aa2d4b316fdf4e9143019832f75ecb24f264904a4843f04ef'
DEV_SEED_MIN = 10_100_000
DEV_SEED_MAX = 10_100_999
LEGACY_FILES = ('__init__.py', 'contract.py', 'params.py', 'sim.py', 'curriculum.py')
LEGACY_REFERENCE_FILES = ('unified_evaluation.py', 'evaluation_guard.py',
                          'guided_curriculum.py', 'observation_memory.py', 'mission_observation.py')
BRIDGE_FILES = ('contract.py', 'camera_history.py', 'policy_runtime.py')
IMPORT_REMAPS = {
    'camera_history.py': ('from src import contract', 'from group_bridge import contract'),
    'policy_runtime.py': ('from src.camera_history import CameraHistory',
                          'from group_bridge.camera_history import CameraHistory'),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate_range(seed_start, episodes):
    if type(episodes) is not int or not 1 <= episodes <= 200:
        raise ValueError('Use 1..200 episodes per suite')
    if (type(seed_start) is not int or seed_start < DEV_SEED_MIN
            or seed_start + episodes - 1 > DEV_SEED_MAX):
        raise ValueError('Only dedicated 10,100,000..10,100,999 development seeds are allowed; '
                         'fixed validation and reserved acceptance are blocked')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False), encoding='utf-8')


def snapshot(out, *, episodes=2, seed_start=DEV_SEED_MIN,
             legacy_root=LEGACY_ROOT, group_root=ROOT):
    """Copy exact sources first; record every namespace-only derived bridge file."""
    validate_range(seed_start, episodes)
    out, legacy_root, group_root = map(Path, (out, legacy_root, group_root))
    model = group_root / 'trainedModels' / (MODEL_NAME + '.zip')
    if sha(model) != MODEL_SHA256:
        raise ValueError('Selected original85 model checksum changed')
    out.mkdir(parents=True, exist_ok=False)
    source = out / 'source'
    inputs, frozen = {}, {}

    def copy(original, relative):
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        digest = sha(target)
        if digest != sha(original):
            raise RuntimeError(f'Source changed while copying: {original}')
        inputs[str(original.resolve())] = digest
        frozen[str(target.relative_to(out)).replace('\\', '/')] = digest
        return target

    for name in LEGACY_FILES:
        copy(legacy_root / 'src' / name, Path('legacy/src') / name)
    for name in LEGACY_REFERENCE_FILES:
        copy(legacy_root / 'src' / name, Path('legacy/src') / name)
    transformations = {}
    for name in BRIDGE_FILES:
        original = copy(group_root / 'src' / name, Path('group_original/src') / name)
        payload = original.read_bytes()
        if name in IMPORT_REMAPS:
            before, after = (text.encode('ascii') for text in IMPORT_REMAPS[name])
            if payload.count(before) != 1:
                raise ValueError(f'Expected one exact namespace import in {name}')
            payload = payload.replace(before, after)
            transformations[name] = dict(before=before.decode(), after=after.decode(),
                                        kind='import namespace only')
        target = source / 'bridge/group_bridge' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        frozen[str(target.relative_to(out)).replace('\\', '/')] = sha(target)
    init = source / 'bridge/group_bridge/__init__.py'
    init.write_bytes(b'')
    frozen[str(init.relative_to(out)).replace('\\', '/')] = sha(init)
    copy(model, Path('models') / model.name)
    copy(group_root / 'tools/qualification_legacy_worker.py', Path('qualification_legacy_worker.py'))
    copy(Path(__file__), Path('evaluate_qualification_compatibility.py'))
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=group_root,
                                         text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    manifest = dict(schema_version=1, kind='legacy compatibility DEVELOPMENT ONLY',
                    created_utc=datetime.now(timezone.utc).isoformat(),
                    episodes_per_suite=episodes, seed_start=seed_start,
                    group_commit=commit, input_sha256=inputs, frozen_sha256=frozen,
                    import_namespace_transformations=transformations,
                    model_name=MODEL_NAME, model_sha256=MODEL_SHA256,
                    device='cpu', training=False, accepted_in_simulation=False,
                    acceptance_attempted=False, physical_flight_validated=False,
                    camera='Explicit frozen legacy Camera(); ambient calibration prohibited',
                    actor_inputs='Newest eight public camera features, selected85 documented '
                                 'lags, actually issued learned actions, unseen flag',
                    reference_thresholds_only={'original_centered': .85, 'varied_approach': .80,
                                               'full_task': .80, 'full_task_delay_200ms': .80})
    write_json(out / 'manifest.json', manifest)
    return manifest


def verify_frozen(out, manifest):
    for relative, expected in manifest['frozen_sha256'].items():
        path = Path(out) / relative
        if sha(path) != expected:
            raise ValueError(f'Frozen source/model checksum mismatch: {relative}')


def run(out, **kwargs):
    out = Path(out).resolve()
    manifest = snapshot(out, **kwargs)
    verify_frozen(out, manifest)
    command = [sys.executable, '-I', str(out / 'source/qualification_legacy_worker.py'),
               '--run', str(out)]
    with (out / 'worker.stdout.log').open('w', encoding='utf-8') as stdout, \
            (out / 'worker.stderr.log').open('w', encoding='utf-8') as stderr:
        process = subprocess.run(command, cwd=out, stdout=stdout, stderr=stderr, check=False)
    verify_frozen(out, manifest)
    unchanged = {path: Path(path).exists() and sha(path) == digest
                 for path, digest in manifest['input_sha256'].items()}
    write_json(out / 'execution.json', dict(command=command, returncode=process.returncode,
                                            input_sources_unchanged=unchanged,
                                            frozen_sources_unchanged=True))
    if process.returncode:
        raise RuntimeError(f'Legacy worker failed; inspect {out / "worker.stderr.log"}')
    return json.loads((out / 'results.json').read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=2)
    parser.add_argument('--seed-start', type=int, default=DEV_SEED_MIN)
    args = parser.parse_args()
    try:
        result = run(args.out, episodes=args.episodes, seed_start=args.seed_start)
    except (ValueError, FileExistsError) as error:
        parser.error(str(error))
    print(json.dumps(result['suites'], indent=2))


if __name__ == '__main__':
    main()
