"""Evaluate and rank saved policies on CUDA, without training or teacher actions.

Run ``python tools/evaluate_checkpoints.py`` from the experiment directory, or
invoke this file from another directory. Relative --models and --output paths
are always resolved from experiments/faisal. Reports go to runs/checkpoint-selection.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
# The group repository also has a src package. This experiment must come first
# when the CLI is invoked from the repository root or as a standalone script.
sys.path.insert(0, str(ROOT))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', type=Path, nargs='+',
                        default=[Path('models')/f'c{step}k.zip' for step in (520, 670, 1000)])
    parser.add_argument('--single-episodes', type=int, default=100)
    parser.add_argument('--full-episodes', type=int, default=25)
    parser.add_argument('--seed-start', type=int, default=4_000_000)
    parser.add_argument('--output', type=Path, default=Path('runs/checkpoint-selection'))
    args = parser.parse_args(argv)
    if min(args.single_episodes, args.full_episodes) <= 0:
        parser.error('Episode counts must be positive')
    if args.seed_start < 0:
        parser.error('--seed-start must be nonnegative')
    return args


def resolve_path(path):
    return (ROOT/path).resolve() if not path.is_absolute() else path.resolve()


def portable_path(path, root=ROOT):
    path = path.resolve()
    try:
        return path.relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def source_hashes():
    return {portable_path(path): sha256(path) for path in sorted((ROOT/'src').rglob('*.py'))}


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def rank_key(candidate):
    suites = candidate['suites']
    centered, varied = (suites[name]['success_rate'] for name in ('centered', 'varied'))
    return (min(centered, varied), suites['full_task']['success_rate'],
            (centered+varied)/2, -suites['full_task']['crash_rate'])


def markdown_report(report):
    lines = ['# Checkpoint selection', '',
             f"CUDA: {report['device']}. Deterministic policy evaluation; no learning or teacher actions.",
             '', f"Each suite starts at seed {report['seed_start']:,}. "
             f"Centered and varied: {report['single_pass_episodes']} episodes each; "
             f"full task: {report['full_task_episodes']} episodes.", '',
             '| Rank | Checkpoint | Centered | Varied | Minimum | Full task | Full passes | Full crashes |',
             '|---:|---|---:|---:|---:|---:|---:|---:|']
    for index, candidate in enumerate(sorted(report['candidates'], key=rank_key, reverse=True), 1):
        suites = candidate['suites']
        def fraction(name, field='successful_episodes'):
            return f"{suites[name][field]}/{suites[name]['episodes']}"
        label = candidate['checkpoint'].replace('|', '\\|')
        lines.append(f"| {index} | {label} | {fraction('centered')} | {fraction('varied')} | "
                     f"{rank_key(candidate)[0]:.0%} | {fraction('full_task')} | "
                     f"{suites['full_task']['total_passes']} | {fraction('full_task', 'crashes')} |")
    lines += ['', f"Selected checkpoint: **{report['selected_checkpoint']}**.", '',
              report['selection_rule']+'.', '', report['seed_note'], '',
              'Centered and varied success requires a clean full-body ring pass within 150 steps. '
              'Full-task success requires surviving 600 steps with at least two passes. '
              'This ranking selects a candidate; it does not establish reliable lap flight or delay robustness.', '',
              'Checkpoint hashes, source hashes, and model training/update counters were verified unchanged.', '',
              f"Elapsed: {report['elapsed_seconds']:.1f} seconds.", '']
    return '\n'.join(lines)


def main(argv=None):
    args = parse_args(argv)
    checkpoints = [resolve_path(path) for path in args.models]
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            raise FileNotFoundError(f'Checkpoint not found: {portable_path(checkpoint)}')
    output = resolve_path(args.output)

    import torch
    from stable_baselines3 import SAC
    from src.curriculum import LessonEnv
    from src.sim import CRASHES, HoopEnv

    torch.set_num_threads(1)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required; CPU fallback is disabled')
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    initial_sources = source_hashes()
    report = {
        'status': 'running', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'device': torch.cuda.get_device_name(0), 'threads': 1,
        'learning_enabled': False, 'teacher_enabled': False, 'deterministic': True,
        'seed_start': args.seed_start, 'single_pass_episodes': args.single_episodes,
        'full_task_episodes': args.full_episodes, 'source_sha256': initial_sources,
        'selection_rule': 'Maximize minimum centered/varied success; full-task success secondary; '
                          'mean single-pass success, then lower full-task crash rate break remaining ties. '
                          'Exact ties retain the supplied model order',
        'seed_note': 'The default 4M seed range is additional selection data, separate from '
                     'demonstration seeds starting at 1M and fixed evaluation seeds starting at 3M. '
                     'Custom --seed-start values change that separation. Historical training episode '
                     'seeds were not logged individually, so zero historical overlap is not formally '
                     'proven. Selection results are no longer untouched test data.',
        'candidates': [],
    }
    report_path = output/'candidate-selection.json'
    atomic_json(report_path, report)
    for checkpoint in checkpoints:
        checkpoint_hash = sha256(checkpoint)
        label = portable_path(checkpoint)
        model = SAC.load(checkpoint, device='cuda')
        if next(model.policy.parameters()).device.type != 'cuda':
            raise RuntimeError('Loaded actor is not on CUDA')
        model.policy.set_training_mode(False)
        initial_steps, initial_updates = model.num_timesteps, model._n_updates
        candidate = {'checkpoint': label, 'sha256': checkpoint_hash,
                     'step': int(initial_steps), 'suites': {}}
        report['candidates'].append(candidate)
        print(f'Loaded {label} on {report["device"]}', flush=True)
        suites = [('centered', lambda: LessonEnv(stage=0), args.single_episodes),
                  ('varied', lambda: LessonEnv(stage=1), args.single_episodes),
                  ('full_task', lambda: HoopEnv(level=1.), args.full_episodes)]
        for name, make_env, count in suites:
            suite_started = time.monotonic()
            env, records = make_env(), []
            try:
                with torch.inference_mode():
                    for index in range(count):
                        seed = args.seed_start+index
                        obs, _ = env.reset(seed=seed)
                        total_reward = 0.
                        while True:
                            action, _ = model.predict(obs, deterministic=True)
                            obs, reward, terminated, truncated, info = env.step(action)
                            total_reward += reward
                            if terminated or truncated:
                                break
                        records.append({'seed': seed, 'outcome': info['outcome'],
                                        'passes': int(info['passes']), 'steps': int(info['steps']),
                                        'reward': float(total_reward)})
                        if (index+1) % 25 == 0 or index+1 == count:
                            print(f'{label} {name}: {index+1}/{count} episodes', flush=True)
            finally:
                env.close()
            successes = sum(r['outcome'] == 'passed' if name != 'full_task' else
                            r['outcome'] == 'timeout' and r['passes'] >= 2 for r in records)
            crashes = sum(r['outcome'] in CRASHES for r in records)
            candidate['suites'][name] = {
                'episodes': count, 'successful_episodes': successes,
                'success_rate': successes/count, 'crashes': crashes, 'crash_rate': crashes/count,
                'total_passes': sum(r['passes'] for r in records),
                'episodes_with_passes': sum(r['passes'] > 0 for r in records),
                'outcomes': dict(Counter(r['outcome'] for r in records)),
                'mean_steps': sum(r['steps'] for r in records)/count,
                'mean_reward': sum(r['reward'] for r in records)/count,
                'elapsed_seconds': time.monotonic()-suite_started, 'records': records,
            }
            atomic_json(report_path, report)
            print(f'{label} {name}: {successes}/{count} successes, {crashes}/{count} crashes', flush=True)
        if (model.num_timesteps, model._n_updates) != (initial_steps, initial_updates):
            raise RuntimeError('Evaluation changed training counters')
        if sha256(checkpoint) != checkpoint_hash:
            raise RuntimeError(f'Checkpoint changed during evaluation: {label}')
        candidate.update(checkpoint_unchanged=True, unchanged_training_steps=int(initial_steps),
                         unchanged_gradient_updates=int(initial_updates),
                         minimum_single_pass_success=rank_key(candidate)[0])
        del model
        torch.cuda.empty_cache()
    if source_hashes() != initial_sources:
        raise RuntimeError('Experiment source files changed during evaluation')
    ranked = sorted(report['candidates'], key=rank_key, reverse=True)
    report.update(status='complete', source_files_unchanged=True,
                  selected_checkpoint=ranked[0]['checkpoint'],
                  selected_step=ranked[0]['step'],
                  ranking=[candidate['checkpoint'] for candidate in ranked],
                  elapsed_seconds=time.monotonic()-started)
    atomic_json(report_path, report)
    (output/'REPORT.md').write_text(markdown_report(report), encoding='utf-8')
    print(json.dumps({'selected_checkpoint': report['selected_checkpoint'],
                      'report': portable_path(report_path),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)
    return report


if __name__ == '__main__':
    main()
