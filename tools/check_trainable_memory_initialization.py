"""Bounded CPU initialization parity against preserved development trajectories.

Run once after review: python -m tools.check_trainable_memory_initialization
    --out <new-directory>
Optional --smoke uses only the first episode per suite and cannot satisfy the
75-episode preflight. Neither mode is mission acceptance or a flight release.
The simulator episode implementation is called unchanged with default physics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import platform
import subprocess
import time
import traceback

import numpy as np
import torch
import stable_baselines3
from stable_baselines3 import SAC

from src import contract
from src.camera_history import CameraHistory
from src.trainable_memory_actor import (TrainableMemoryPolicy, ESTIMATOR_PROFILE,
                                        PROFILE, SOURCE_PROFILE, checkpoint_manifest)
from src.policy_runtime import SPECS
from src.recurrent_ring_position import RING_POSITION_SPECS
from tools.validate_flight_policies import episode, SUITES
from tools.run_clearance_experiment import single_trainer


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT.parent / 'lab-data/policy-validation-20261006'
SOURCE = ROOT / 'trainedModels/seenflag-s1-level1.zip'
ESTIMATOR = Path('D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt')
BASELINE_PROTOCOL_SHA256 = 'fe8b3a33a894a4176326a240dc0514c3a2e43f930d7fa24690363bac57661f16'
FIXED_SUITES = {'aligned': 9410000, 'nominal': 9420000, 'randomised': 9430000}
EXACT_BASELINE_SOURCES = ('src/contract.py', 'src/params.py', 'src/pass_audit.py',
                          'src/policy_runtime.py', 'src/sim.py', 'src/__init__.py')
# These two reviewed additions preserve the defaults used here. All physics,
# scoring, resets, returned fields and trajectory construction are unchanged.
# Reject any subsequent source edit until this preflight is explicitly reviewed.
REVIEWED_DEFAULT_COMPATIBILITY = {
    'src/camera_history.py': {
        'sha256': 'b30d2f8ecb34e063b3b7115ffaa4203a26262429f2c028b0cbdfc7d16d41231c',
        'difference': 'Adds optional action_history=3 or 4; this bridge uses the unchanged default 3.'},
    'tools/validate_flight_policies.py': {
        'sha256': '3f637d84c2702fa99e33971889d98dcab2b5820b9ac74c2b44d9e75a91b21654',
        'difference': 'Adds diagnostic_dynamics=None hook; this runner never passes an override.'},
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    """Exclusive create: never replace an existing report or protocol."""
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def identical(left, right):
    """Bitwise array identity, including dtype, shape and the sign of zero."""
    return (isinstance(left, np.ndarray) and isinstance(right, np.ndarray)
            and left.dtype == right.dtype and left.shape == right.shape
            and left.tobytes(order='C') == right.tobytes(order='C'))


def require_identical(left, right, label):
    if not identical(left, right):
        raise ValueError(f'Bitwise mismatch: {label}')


class StopRequested(RuntimeError):
    pass


class Budget:
    def __init__(self, out, seconds=600):
        self.stop_path = Path(out) / 'STOP'
        self.started = time.monotonic()
        self.seconds = seconds

    def __call__(self):
        if self.stop_path.exists():
            raise StopRequested(f'STOP requested: {self.stop_path}')
        if time.monotonic() - self.started >= self.seconds:
            raise StopRequested(f'Bounded {self.seconds}-second preflight budget exhausted')


class InitializationBridge:
    """Public-only adapter for unchanged validate_flight_policies.episode().

    The preflight always supplies the actual source SAC as reference_sac. A
    caller evaluating a trained residual may explicitly use reference_sac=None;
    that disables only original-action parity, never prefix/recurrence checks.
    guard() is called before every frame, before issuing an action, and on reset.
    No position, heading or simulator object enters this adapter.
    """
    def __init__(self, policy, reference_sac, guard):
        self.policy, self.reference_sac, self.guard = policy, reference_sac, guard
        self.reset()

    def reset(self):
        self.guard()
        self.history = CameraHistory(SPECS[SOURCE_PROFILE].lags)
        self.policy.reset()
        self.frames = 0
        self.action_parity_checks = 0
        self.last_frame = None

    def predict(self, public8, previous_issued):
        self.guard()
        public85 = self.history.update(public8, previous_issued)
        if public85.dtype != np.float32 or public85.shape != (85,):
            raise ValueError('The original explicit public85 history is required')
        if self.policy.frame_count != self.frames:
            raise ValueError('Recurrence advanced outside the once-per-frame bridge')
        features = self.policy.observe(public85)
        self.frames += 1
        if self.policy.frame_count != self.frames:
            raise ValueError('observe() must advance recurrence exactly once')
        if features.dtype != np.float32 or features.shape != (150,):
            raise ValueError('The explicit trainable-control-memory150 profile is required')
        require_identical(features[:85], public85, 'new150 prefix versus public85')
        action, _ = self.policy.predict(features, deterministic=True)
        if self.policy.frame_count != self.frames:
            raise ValueError('predict() must not advance recurrence')
        if (not isinstance(action, np.ndarray) or action.dtype != np.float32
                or action.shape != (4,) or not np.isfinite(action).all()
                or np.any(np.abs(action) > 1)):
            raise ValueError('Invalid learned action')
        self.last_frame = dict(frame=self.frames, public85=public85.tolist(),
                               features150=features.tolist(), action=action.tolist())
        if self.reference_sac is not None:
            expected, _ = self.reference_sac.predict(public85, deterministic=True)
            self.last_frame['source_action'] = expected.tolist()
            require_identical(action, expected, f'actual SAC predict action at frame {self.frames}')
            self.action_parity_checks += 1
        self.guard()
        return features, action


def check_baseline_protocol(protocol):
    if (protocol.get('suites') != FIXED_SUITES or SUITES != FIXED_SUITES
            or protocol.get('episodes') != 100 or protocol.get('max_steps') != 600
            or contract.LAPS_MAX_STEPS != 600 or contract.ACTION_HISTORY != 3
            or protocol.get('flight_release') is not False
            or protocol.get('training') is not False
            or protocol.get('pass_metric') != 'Full body past far noodle face; collision-free transit'):
        raise ValueError('Original development protocol semantics changed')
    profile = protocol.get('profiles', {}).get(SOURCE_PROFILE)
    expected = asdict(SPECS[SOURCE_PROFILE]); expected['lags'] = list(expected['lags'])
    if profile != expected:
        raise ValueError('Original public85 source profile changed')


def assert_zero(policy):
    policy.actor.assert_source_unchanged()
    weight = policy.actor.residual_weight.detach().cpu().numpy()
    if weight.dtype != np.float32 or weight.shape != (4, 64) or np.count_nonzero(weight):
        raise ValueError('Initialization requires an exactly zero residual head')
    if policy.device.type != 'cpu' or policy.device.type != 'cpu':
        raise ValueError('Initialization preflight is CPU only')


def compare_trace(current, baseline):
    with np.load(current, allow_pickle=False) as actual, np.load(baseline, allow_pickle=False) as old:
        if set(actual.files) != set(old.files):
            raise ValueError('Trace field set differs from preserved baseline')
        for name in old.files:
            if name == 'actor_observations':
                new, original = actual[name], old[name]
                if (new.ndim != 2 or original.ndim != 2 or new.shape[1] != 150
                        or original.shape[1] != 85 or new.shape[0] != original.shape[0]
                        or not np.isfinite(new).all()):
                    raise ValueError('Trace observation profile differs')
                require_identical(new[:, :85], original, 'all saved original85 prefixes')
                validity = (np.arange(len(new)) >= 2).astype(np.float32)
                require_identical(new[:, 149], validity, 'saved history-valid flags')
                require_identical(new[:2, 85:149], np.zeros_like(new[:2, 85:149]),
                                  'saved startup estimate mask')
            else:
                require_identical(actual[name], old[name], f'saved {name}')
        return dict(trace_fields=sorted(old.files), prefix85_bitwise=True,
                    actions_bitwise=True, positions_bitwise=True, headings_bitwise=True,
                    scenario_arrays_bitwise=True, startup_mask_and_validity_exact=True)


def summary(rows):
    return dict(episodes=len(rows), clean_first_pass=sum(r['clean_first_pass'] for r in rows),
                crashes=sum(r['crashed'] for r in rows),
                full_body_passes=sum(r['full_body_passes'] for r in rows),
                returned_episodes=sum(r['returns_to_front'] > 0 for r in rows),
                clean_repeated_episodes=sum(r['clean_repeated_episode'] for r in rows),
                endings=dict(Counter(r['outcome'] for r in rows)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    budget = Budget(out, 600)
    torch.set_num_threads(1)
    episodes_per_suite = 1 if args.smoke else 25
    expected_count = 3 * episodes_per_suite
    bindings, evidence, current_sources = {}, {}, {}
    completed, comparisons, results = [], [], {}
    protocol_hash = residual_hash = None
    bridge = policy = None
    phase, current_case = 'freeze', None
    status, error = 'failed', None

    def freeze(source, relative):
        budget()
        source = Path(source)
        payload = source.read_bytes()
        hashed = hashlib.sha256(payload).hexdigest()
        dest = out / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open('xb') as stream:
            stream.write(payload)
        bindings[str(source.resolve())] = hashed
        evidence[Path(relative).as_posix()] = hashed
        return hashed

    try:
        baseline_protocol = BASELINE / 'protocol.json'
        if digest(baseline_protocol) != BASELINE_PROTOCOL_SHA256:
            raise ValueError('Preserved baseline protocol hash mismatch')
        baseline_spec = json.loads(baseline_protocol.read_text())
        check_baseline_protocol(baseline_spec)
        freeze(baseline_protocol, 'baseline/protocol.json')
        baseline_hashes = {key.replace('\\', '/'): value
                           for key, value in baseline_spec['source_sha256'].items()}
        for relative in (*EXACT_BASELINE_SOURCES, *REVIEWED_DEFAULT_COMPATIBILITY):
            frozen = BASELINE / 'source' / relative
            if digest(frozen) != baseline_hashes[relative]:
                raise ValueError(f'Preserved source hash mismatch: {relative}')
            freeze(frozen, 'baseline/source/' + relative)
            expected_hash = (REVIEWED_DEFAULT_COMPATIBILITY[relative]['sha256']
                             if relative in REVIEWED_DEFAULT_COMPATIBILITY
                             else baseline_hashes[relative])
            if digest(ROOT / relative) != expected_hash:
                raise ValueError(f'Unreviewed simulator/history/evaluator change: {relative}')
        if digest(SOURCE) != SPECS[SOURCE_PROFILE].sha256:
            raise ValueError('Original source model hash mismatch')
        if digest(ESTIMATOR) != RING_POSITION_SPECS[ESTIMATOR_PROFILE].checkpoint_sha256:
            raise ValueError('Frozen estimator hash mismatch')
        files = sorted((ROOT / 'src').glob('*.py')) + [
            ROOT / 'tools/validate_flight_policies.py', ROOT / 'tools/run_clearance_experiment.py',
            Path(__file__).resolve()]
        for source in files:
            relative = source.relative_to(ROOT).as_posix()
            current_sources[relative] = freeze(source, 'source/' + relative)
        source_hash = freeze(SOURCE, 'source/trainedModels/seenflag-s1-level1.zip')
        estimator_hash = freeze(ESTIMATOR, 'source/estimator/gru64.pt')
        baseline_rows = {}
        for suite, base in FIXED_SUITES.items():
            log = BASELINE / SOURCE_PROFILE / suite / 'episodes.jsonl'
            freeze(log, f'baseline/{suite}/episodes.jsonl')
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if len(rows) != 100 or [row['seed'] for row in rows] != list(range(base, base + 100)):
                raise ValueError(f'Preserved baseline cohort differs: {suite}')
            for row in rows[:episodes_per_suite]:
                seed = row['seed']
                baseline_rows[suite, seed] = row
                freeze(BASELINE / SOURCE_PROFILE / suite / f'{seed}.npz',
                       f'baseline/{suite}/{seed}.npz')
        protocol = dict(
            kind='trainable_memory_initialization_parity', version=1,
            created_utc=datetime.now(timezone.utc).isoformat(), smoke=args.smoke,
            device='cpu', torch_threads=1, max_seconds=600, stop_file=str(out / 'STOP'),
            episodes_per_suite=episodes_per_suite, expected_episodes=expected_count,
            suites=FIXED_SUITES, max_steps=600, source_profile=SOURCE_PROFILE,
            source_sha256=source_hash, estimator_profile=ESTIMATOR_PROFILE,
            estimator_sha256=estimator_hash, actor_manifest=checkpoint_manifest(),
            baseline_path=str(BASELINE), baseline_protocol_sha256=BASELINE_PROTOCOL_SHA256,
            baseline_comparison='Exact episode fields and all trace arrays; new150 prefix85 exact',
            action_reference='Actual CPU SAC.predict(public85, deterministic=True), including unscale',
            simulator_default_compatibility=REVIEWED_DEFAULT_COMPATIBILITY,
            default_physics=True, diagnostic_dynamics=None, source_files_sha256=current_sources,
            frozen_input_sha256=dict(evidence), original_input_bindings_sha256=dict(bindings),
            git_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
            versions=dict(python=platform.python_version(), numpy=np.__version__,
                          torch=torch.__version__, stable_baselines3=stable_baselines3.__version__),
            acceptance_passed=False, flight_release=False, training=False,
            limitation='Development initialization parity only. No policy improvement or physical validation.')
        write_json(out / 'protocol.json', protocol)
        protocol_hash = digest(out / 'protocol.json')
        evidence['protocol.json'] = protocol_hash

        phase = 'zero_residual_roundtrip'
        budget()
        source_path = out / 'source/trainedModels/seenflag-s1-level1.zip'
        estimator_path = out / 'source/estimator/gru64.pt'
        fresh = TrainableMemoryPolicy(source_path, estimator_path, device='cpu')
        assert_zero(fresh)
        residual_hash = fresh.save(out / 'zero_residual.npz')
        policy = TrainableMemoryPolicy.load(out / 'zero_residual.npz', source_path, estimator_path,
                                           device='cpu', sha256=residual_hash)
        assert_zero(policy)
        reference = SAC.load(BytesIO(source_path.read_bytes()), device='cpu')
        with np.load(out / 'baseline/aligned/9410000.npz', allow_pickle=False) as first:
            public_rows = first['actor_observations'][:32].copy()
        for index, public85 in enumerate(public_rows):
            budget()
            left, right = fresh.observe(public85), policy.observe(public85)
            require_identical(left, right, f'roundtrip features at row {index}')
            a, _ = fresh.predict(left, deterministic=True)
            b, _ = policy.predict(right, deterministic=True)
            expected, _ = reference.predict(public85, deterministic=True)
            require_identical(a, b, f'roundtrip actions at row {index}')
            require_identical(b, expected, f'roundtrip actual source actions at row {index}')
        fresh.reset(); policy.reset()
        assert_zero(fresh); assert_zero(policy)
        del fresh
        write_json(out / 'roundtrip.json', dict(rows=len(public_rows), exact=True,
                   residual_sha256=residual_hash, source_predict_bitwise=True,
                   prefix_features_actions_bitwise=True, residual_all_zero=True,
                   inference_contract='strict_float32', physical_validation=False))
        evidence['zero_residual.npz'] = residual_hash
        evidence['roundtrip.json'] = digest(out / 'roundtrip.json')
        bridge = InitializationBridge(policy, reference, budget)

        phase = 'fixed_episode_parity'
        for suite, base in FIXED_SUITES.items():
            folder = out / suite
            folder.mkdir(exist_ok=False)
            suite_rows = []
            with (folder / 'episodes.jsonl').open('x', encoding='utf-8', newline='\n') as log:
                for seed in range(base, base + episodes_per_suite):
                    budget()
                    current_case = dict(suite=suite, seed=seed)
                    trace_path = folder / f'{seed}.npz'
                    if trace_path.exists():
                        raise FileExistsError(trace_path)
                    row = episode(bridge, suite, seed, trace_path)
                    log.write(json.dumps(row, sort_keys=True) + '\n'); log.flush()
                    baseline_row = baseline_rows[suite, seed]
                    if row != baseline_row:
                        raise ValueError(f'Episode fields mismatch for {suite}/{seed}: '
                                         f'actual={row}, baseline={baseline_row}')
                    comparison = compare_trace(trace_path, out / f'baseline/{suite}/{seed}.npz')
                    if bridge.action_parity_checks != row['steps'] or bridge.frames != row['steps']:
                        raise ValueError('Not every episode frame passed actual-source action parity')
                    assert_zero(policy)
                    comparison.update(suite=suite, seed=seed, episode_fields_exact=True,
                                      actual_source_action_checks=bridge.action_parity_checks,
                                      trace_sha256=digest(trace_path))
                    comparisons.append(comparison)
                    suite_rows.append(row); completed.append(row)
                    evidence[f'{suite}/{seed}.npz'] = comparison['trace_sha256']
                    print(suite, seed, 'exact', row['steps'], flush=True)
            results[suite] = summary(suite_rows)
            evidence[f'{suite}/episodes.jsonl'] = digest(folder / 'episodes.jsonl')

        phase = 'final_integrity'
        budget()
        assert_zero(policy)
        for path, expected_hash in bindings.items():
            budget()
            if digest(path) != expected_hash:
                raise ValueError(f'Original model/source/baseline changed during preflight: {path}')
        for relative, expected_hash in evidence.items():
            budget()
            if digest(out / relative) != expected_hash:
                raise ValueError(f'Frozen evidence changed during preflight: {relative}')
        if len(completed) != expected_count:
            raise ValueError('Incomplete fixed parity cohort')
        status = 'complete'
    except BaseException as exc:
        status = 'stopped' if isinstance(exc, (StopRequested, KeyboardInterrupt)) else 'failed'
        error = dict(type=type(exc).__name__, message=str(exc), phase=phase,
                     current_case=current_case, traceback=traceback.format_exc())
        if bridge is not None:
            error['last_bridge_frame'] = bridge.last_frame
        write_json(out / 'failure.json', error)
    finally:
        with (out / 'comparisons.jsonl').open('x', encoding='utf-8', newline='\n') as stream:
            for item in comparisons:
                stream.write(json.dumps(item, sort_keys=True) + '\n')
        write_json(out / 'results.json', results)
        # Include incomplete traces and failure diagnostics, preserving all evidence.
        evidence = {path.relative_to(out).as_posix(): digest(path)
                    for path in sorted(out.rglob('*')) if path.is_file() and path.name != 'STOP'}
        decision = dict(kind='trainable_memory_initialization_parity', status=status,
            smoke=args.smoke, expected_episodes=expected_count, completed_episodes=len(completed),
            initialization_parity_passed=status == 'complete' and not args.smoke and len(completed) == 75,
            software_smoke_passed=status == 'complete' and args.smoke,
            acceptance_passed=False, flight_release=False, training=False,
            source_sha256=SPECS[SOURCE_PROFILE].sha256,
            estimator_sha256=RING_POSITION_SPECS[ESTIMATOR_PROFILE].checkpoint_sha256,
            protocol_sha256=protocol_hash, residual_sha256=residual_hash,
            elapsed_seconds=time.monotonic() - budget.started,
            evidence_sha256=evidence, error=error)
        write_json(out / 'decision.json', decision)
        print(json.dumps({key: value for key, value in decision.items()
                          if key not in ('evidence_sha256', 'error')}, sort_keys=True), flush=True)
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    with single_trainer(ROOT.parent / '.clearance-training.lock'):
        raise SystemExit(main())
