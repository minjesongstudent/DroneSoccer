"""One authorized three-arm CUDA cohort; no automatic retry or policy promotion.

The coordinator exclusively owns the existing project lock and Windows lifetime
job. Workers share predeclared comparison seeds intentionally but own their RNG,
outputs and .15 GPU allocator limits. Only one final candidate may be nominated
for a later fresh development comparison. No reserved acceptance seeds are used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time
import uuid

from tools.parallel_training import (ARMS, CohortProcesses, canonical, digest, process_identity,
                                     validate_manifest)


ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('D:/DroneSoccerData/runs')
PLAN = 'experiments/faisal/docs/PARALLEL_CREDIT_COMPARISON_20261008.md'
SUITES = {'aligned': 9410000, 'nominal': 9420000, 'randomised': 9430000}


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False); stream.write('\n')


def write_status(path, value):
    path = Path(path); temp = path.with_suffix('.tmp')
    temp.write_text(canonical(value), encoding='utf-8'); temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def existing_allocation(root, seeds):
    wanted = set(seeds); conflicts = []; logs = protocols = count = 0
    if len(seeds) != len(wanted) or not all(11000000 <= seed < 11250000 for seed in seeds):
        raise ValueError('Existing unique FIT seed bounds required')
    for path in sorted(Path(root).rglob('episodes.jsonl')):
        logs += 1
        for line in path.read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line); count += 1
                if row.get('seed') in wanted:
                    conflicts.append(dict(path=str(path), seed=row['seed']))
    paths = sorted(set(Path(root).rglob('protocol.json')) | set(Path(root).rglob('cohort_manifest.json')))
    for path in paths:
        protocols += 1; data = read_json(path)
        declared = set(data.get('training_seeds') or [])
        if isinstance(data.get('reset_seed_base'), int):
            n = data.get('iterations', 0) * data.get('episodes_per_iteration', 0)
            declared.update(range(data['reset_seed_base'], data['reset_seed_base'] + n))
        arms = data.get('arms', {})
        if isinstance(arms, dict):
            configs = list(arms.values())
            if any(not isinstance(config, dict) for config in configs):
                raise ValueError('Unsupported historical arm configuration: ' + str(path))
        elif isinstance(arms, list):
            # Older protocols name arms as strings; their allocation is at root.
            if any(not isinstance(config, (str, dict)) for config in arms):
                raise ValueError('Unsupported historical arm list: ' + str(path))
            configs = [config for config in arms if isinstance(config, dict)]
        else:
            raise ValueError('Unsupported historical arms field: ' + str(path))
        for config in configs:
            declared.update(config.get('training_seeds') or [])
        overlap = sorted(wanted & declared)
        if overlap: conflicts.append(dict(path=str(path), allocated_seeds=overlap))
    record = dict(seeds=seeds, conflicts=conflicts, episode_logs=logs, episode_rows=count,
                  protocols=protocols, sharing='Intentional matched seeds across exactly three registered arms')
    if conflicts:
        raise ValueError('Cohort seed allocation overlaps earlier evidence: ' + canonical(record))
    return record


def initialization_identity(path):
    path = Path(path).resolve(); decision = read_json(path)
    if (decision.get('status') != 'complete' or decision.get('smoke') is not False
            or decision.get('initialization_parity_passed') is not True
            or decision.get('completed_episodes') != 75 or decision.get('expected_episodes') != 75
            or decision.get('acceptance_passed') is not False):
        raise ValueError('The completed non-smoke75 initialization preflight is required')
    protocol_path = path.parent/'protocol.json'
    if digest(protocol_path) != decision['protocol_sha256']:
        raise ValueError('Initialization protocol identity changed')
    for relative, expected in decision['evidence_sha256'].items():
        target = (path.parent/relative).resolve()
        if not target.is_relative_to(path.parent) or digest(target) != expected:
            raise ValueError('Initialization evidence changed')
    for relative, expected in read_json(protocol_path)['source_files_sha256'].items():
        if digest(ROOT/relative) != expected:
            raise ValueError('Initialization-bound source changed: ' + relative)
    return dict(path=str(path), sha256=digest(path), protocol_sha256=decision['protocol_sha256'])


def gpu_snapshot():
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,name,memory.total,memory.used,memory.free',
                                    '--format=csv,noheader,nounits'], text=True, timeout=10,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
    rows = [line.split(',') for line in text.strip().splitlines()]
    first = next(row for row in rows if int(row[0]) == 0)
    return dict(index=0, name=first[1].strip(), total_mib=int(first[2]), used_mib=int(first[3]), free_mib=int(first[4]))


def freeze_sources(out):
    files = list((ROOT/'src').glob('*.py')) + [ROOT/path for path in (
        'tools/parallel_training.py', 'tools/run_parallel_estimated_motion.py',
        'tools/run_estimated_motion_trial.py', 'tools/check_estimated_motion_initialization.py',
        'tools/run_clearance_experiment.py', 'tools/run_onpolicy_trial.py',
        'tools/validate_flight_policies.py', 'tests/test_parallel_training.py',
        'tests/test_estimated_motion_trial.py', PLAN)]
    hashes = {}
    for source in files:
        relative = source.relative_to(ROOT); payload = source.read_bytes()
        target = out/'source'/relative; target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream: stream.write(payload)
        hashes[relative.as_posix()] = digest(target)
        if digest(source) != hashes[relative.as_posix()]: raise ValueError('Source changed while freezing')
    return hashes


def round_one_fairness(out):
    """Read saved first-round bytes only. Never infer or step altered features."""
    import numpy as np
    outputs = {arm: Path(out)/'arms'/arm for arm in ARMS}
    manifest_path = Path(out)/'cohort_manifest.json'; manifest_hash = digest(manifest_path)
    for arm, folder in outputs.items():
        ready_path = folder/'.first_rollout_ready.json'
        if not ready_path.exists():
            return dict(status='waiting', matched=False, missing_round_one=arm)
        try:
            ready = read_json(ready_path)
        except json.JSONDecodeError:
            return dict(status='waiting', matched=False, missing_round_one=arm)
        if (ready['arm'] != arm or ready['cohort_manifest_sha256'] != manifest_hash
                or ready['rollout_sha256'] != digest(folder/'rollout_01.npz')
                or ready['endpoints_sha256'] != digest(folder/'episode_endpoints_01.json')):
            raise ValueError('First rollout ready marker does not match its frozen evidence')
    fields = ('features89', 'actions', 'episode_lengths', 'rewards', 'old_log_prob')
    reference = endpoint_reference = None; comparisons = {}; hashes = {}; endpoint_hashes = {}
    for arm, folder in outputs.items():
        path = folder/'rollout_01.npz'; hashes[arm] = digest(path)
        with np.load(path, allow_pickle=False) as data:
            current = {key: data[key].copy() for key in fields}
        endpoints = read_json(folder/'episode_endpoints_01.json')
        endpoint_hashes[arm] = digest(folder/'episode_endpoints_01.json')
        if reference is None: reference = current
        if endpoint_reference is None: endpoint_reference = endpoints
        comparisons[arm] = {key: bool(value.dtype == reference[key].dtype
            and value.shape == reference[key].shape and value.tobytes() == reference[key].tobytes())
            for key, value in current.items()}
        comparisons[arm]['endpoints'] = endpoints == endpoint_reference
    matched = all(all(values.values()) for values in comparisons.values())
    return dict(status='matched' if matched else 'confounded', matched=matched, passed=matched,
        cohort_manifest_sha256=manifest_hash,
        comparisons=comparisons, rollout_sha256=hashes, endpoint_sha256=endpoint_hashes,
        fields=list(fields)+['endpoints'],
        deliberate_differences=['returns', 'advantages'])


def verify_candidate(folder, manifest, arm):
    folder = Path(folder); decision = read_json(folder/'decision.json'); result = read_json(folder/'result.json')
    if (decision.get('passed') is not True
            or decision.get('decision') != 'eligible_for_one_fresh_development_screen'):
        return None
    if (manifest['smoke'] or result['status'] != 'complete'
            or result['completed_seeds'] != manifest['arms'][arm]['training_seeds']
            or result['final_checkpoint'] != 'policy_05.npz'
            or len(result['iterations']) != 5
            or any(row['episodes'] != 48 or row['failures'] for row in result['iterations'])
            or any(decision[key] is not False for key in ('acceptance', 'policy_promoted', 'physical_flight'))):
        raise ValueError('Incomplete, stopped, smoke or promoted worker cannot qualify')
    if not decision['evidence']:
        raise ValueError('Qualifying worker lacks evidence hashes')
    for path, expected in decision['evidence'].items():
        if digest(path) != expected: raise ValueError('Qualifying worker evidence changed')
    model = read_json(folder/'final_development/model.json')
    if model['sha256'] != digest(folder/'policy_05.npz') or model['episodes_per_suite'] != 100:
        raise ValueError('Worker evaluation is not its prescribed final model')
    for relative, expected in read_json(folder/'final_development/evidence.json').items():
        if digest(folder/'final_development'/relative) != expected: raise ValueError('Final evaluation changed')
    candidate = {}
    for suite, seed in SUITES.items():
        records = [json.loads(line) for line in (folder/'final_development'/suite/'episodes.jsonl').read_text().splitlines()]
        if len(records) != 100 or sorted(row['seed'] for row in records) != list(range(seed, seed+100)):
            raise ValueError('Final worker cohort is incomplete or duplicated')
        candidate[suite] = dict(episodes=100, first_pass=sum(row['clean_first_pass'] for row in records),
            clean_repeat=sum(row['clean_repeated_episode'] for row in records), crashes=sum(row['crashed'] for row in records))
        if candidate[suite] != decision['candidate'][suite]: raise ValueError('Worker final counts differ')
        old, new = decision['reference'][suite], candidate[suite]
        if (new['first_pass'] < old['first_pass']-2 or new['clean_repeat'] < old['clean_repeat']-2
                or new['crashes'] > old['crashes']+2): raise ValueError('Original fixed development gate failed')
    old, new = decision['reference']['randomised'], candidate['randomised']
    if new['clean_repeat'] < old['clean_repeat']+5 or new['crashes'] > old['crashes']-5:
        raise ValueError('Required randomized improvement is absent')
    return dict(arm=arm, checkpoint=str(folder/'policy_05.npz'), sha256=digest(folder/'policy_05.npz'),
                decision_sha256=digest(folder/'decision.json'), counts=candidate)


def select_one(candidates):
    if not candidates: return None
    return sorted(candidates, key=lambda c: (-c['counts']['randomised']['clean_repeat'],
        c['counts']['randomised']['crashes'], -c['counts']['randomised']['first_pass'], ARMS.index(c['arm'])))[0]


def final_summary(out, manifest, manifest_hash, status, registry, fairness, errors):
    candidates = []; workers = {}; errors = list(errors)
    if digest(out/'cohort_manifest.json') != manifest_hash: errors.append('Cohort manifest changed')
    for relative, expected in manifest['source_sha256'].items():
        if digest(ROOT/relative) != expected or digest(out/'source'/relative) != expected:
            errors.append('Bound source changed: ' + relative)
    for arm in ARMS:
        folder = out/'arms'/arm
        try:
            decision = read_json(folder/'decision.json')
            workers[arm] = dict(exit_code=registry.get(arm, {}).get('exit_code'),
                decision=decision['decision'], passed=decision['passed'], reasons=decision.get('reasons', []),
                decision_sha256=digest(folder/'decision.json'))
            candidate = verify_candidate(folder, manifest, arm)
            if candidate is not None and registry.get(arm, {}).get('exit_code') == 0:
                candidates.append(candidate)
        except Exception as exc:
            workers[arm] = dict(error=repr(exc), exit_code=registry.get(arm, {}).get('exit_code'))
            errors.append(arm + ': missing/invalid worker closure')
    if not fairness.get('matched'): errors.append('Round-one matched comparison not established')
    eligible = status == 'complete' and not manifest['smoke'] and not errors
    selected = select_one(candidates) if eligible else None
    result = dict(status=status, smoke=manifest['smoke'], workers=workers, fairness=fairness, errors=errors,
        selection='one_final_candidate_for_fresh_development' if selected else 'none', selected=selected,
        qualifying_final_candidates=candidates, cohort_manifest_sha256=manifest_hash,
        selection_rule='randomised repeat descending, crash ascending, first-pass descending, fixed arm order',
        automatic_fresh_evaluation=False, acceptance=False, policy_promoted=False, physical_flight=False)
    write_new(out/'decision.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--initialization-report', required=True, type=Path)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args(argv)
    started = time.monotonic(); total = 900 if args.smoke else 3900
    out = args.out.resolve(); status = 'preparing'; errors = []; registry = {}; fairness = dict(status='waiting', matched=False)
    manifest = None; manifest_hash = None
    with CohortProcesses(ROOT.parent/'.clearance-training.lock', deadline=started+total) as cohort:
        out.mkdir(parents=True, exist_ok=False)
        try:
            initial = initialization_identity(args.initialization_report)
            seeds = list(range(11244900, 11244902)) if args.smoke else list(range(11244000, 11244240))
            allocation = existing_allocation(RUNS, seeds)
            write_new(out/'seed_allocation.json', allocation)
            device = gpu_snapshot()
            minimum_free = int(3*.15*device['total_mib']) + 2048
            if device['free_mib'] < minimum_free:
                raise RuntimeError(f'Insufficient GPU headroom: {device}, minimum_free_mib={minimum_free}')
            sources = freeze_sources(out)
            configs = {arm: dict(out=str(out/'arms'/arm), target_mode=arm,
                training_seeds=seeds, algorithm_seed=11592000, memory_fraction=.15,
                max_total_seconds=600 if args.smoke else 3600) for arm in ARMS}
            manifest = dict(format='dronesoccer-authorized-cohort-v1', cohort_id=uuid.uuid4().hex,
                created_utc=datetime.now(timezone.utc).isoformat(), parent_pid=os.getpid(),
                parent_creation_time=process_identity(os.getpid()), project_root=str(ROOT),
                max_workers=3, arm_order=list(ARMS), smoke=args.smoke, arms=configs,
                allocation_record=dict(path=str(out/'seed_allocation.json'), sha256=digest(out/'seed_allocation.json')),
                initialization=initial, source_sha256=sources, gpu_before=device,
                gpu_limits='Each PyTorch caching allocator <=.15 of total visible GPU memory; context/non-PyTorch allocations are additional.',
                max_total_seconds=total, child_cpu_compute_threads=1,
                registration='Private inherited-pipe one-use ARM grants, live-parent handle and kill-on-close Windows Job Object',
                seeds_shared_intentionally=True, retries=0,
                gates='Unchanged individual retention/final100/feature-use gates; only final completed model eligible.',
                fairness='Round1 features89/actions/episode_lengths bitwise identical across all three; no winner if confounded.',
                selection='At most one qualifying final: randomised repeat descending, crash ascending, first descending, arm_order.',
                acceptance=False, policy_promoted=False, physical_flight=False)
            validate_manifest(manifest, out/'cohort_manifest.json')
            write_new(out/'cohort_manifest.json', manifest); manifest_hash = digest(out/'cohort_manifest.json')
            for arm in ARMS:
                if (out/'STOP').exists(): raise RuntimeError('STOP before worker launch')
                child = cohort.launch(out/'cohort_manifest.json', arm)
                registry = cohort.snapshot(); write_status(out/'status.json', dict(status='running', workers=registry))
                print('ARMED', arm, child.worker_pid, flush=True)
            status = 'running'; stop_deadline = None; next_gpu = time.monotonic()+10; gpu_log = []
            while True:
                registry = cohort.snapshot()
                if all(row['exit_code'] is not None for row in registry.values()):
                    status = 'stopped' if stop_deadline is not None else 'complete'; break
                if fairness['status'] == 'waiting' and any(row['exit_code'] is not None for row in registry.values()):
                    errors.append('Worker exited before the matched first-rollout barrier')
                    (out/'STOP').touch(exist_ok=True)
                now = time.monotonic()
                if now >= started+total-20 and not (out/'STOP').exists():
                    (out/'STOP').write_text('Cohort wall budget reached; save and stop.\n')
                if (out/'STOP').exists() and stop_deadline is None:
                    stop_deadline = min(started+total, now+20)
                if stop_deadline is not None and now >= stop_deadline:
                    status = 'stopped'; cohort.job.close(); break
                if fairness['status'] == 'waiting':
                    fairness = round_one_fairness(out)
                    if fairness['status'] != 'waiting':
                        write_new(out/'first_rollout_parity.json', fairness)
                        print('ROUND_ONE', fairness['status'], flush=True)
                        if not fairness['matched']:
                            errors.append('Round-one comparison confounded; no winner eligible')
                            (out/'STOP').touch(exist_ok=True)
                if now >= next_gpu:
                    device = gpu_snapshot(); gpu_log.append(dict(elapsed_seconds=now-started, **device)); next_gpu=now+10
                    write_status(out/'gpu_status.json', dict(samples=gpu_log))
                    if device['free_mib'] < 512:
                        errors.append('GPU free memory below512MiB; bounded cohort stop')
                        (out/'STOP').touch(exist_ok=True)
                write_status(out/'status.json', dict(status=status, workers=registry,
                    elapsed_seconds=now-started, fairness=fairness['status']))
                time.sleep(.5)
            for child in cohort.children.values(): child.process.wait(timeout=15)
            registry = cohort.snapshot()
            if fairness['status'] == 'waiting':
                fairness = round_one_fairness(out)
                write_new(out/'first_rollout_parity.json', fairness)
        except BaseException as exc:
            status = 'stopped' if isinstance(exc, KeyboardInterrupt) else 'failed'
            errors.append(repr(exc)); (out/'STOP').touch(exist_ok=True)
            if cohort.job is not None: cohort.job.close()
            for child in cohort.children.values(): child.process.wait(timeout=15)
            registry = cohort.snapshot()
        finally:
            write_status(out/'status.json', dict(status=status, workers=registry,
                elapsed_seconds=time.monotonic()-started, errors=errors))
            if manifest is not None and manifest_hash is not None:
                result = final_summary(out, manifest, manifest_hash, status, registry, fairness, errors)
            else:
                result = dict(status=status, selected=None, selection='none', errors=errors,
                    acceptance=False, policy_promoted=False, physical_flight=False)
                write_new(out/'decision.json', result)
            print('COHORT_DECISION', canonical(result), flush=True)
    return 0 if status == 'complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
