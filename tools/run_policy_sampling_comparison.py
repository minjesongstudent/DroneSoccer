"""Bounded paired FIT diagnostic of the original SAC actor; no optimization."""
from dataclasses import asdict
from pathlib import Path
import argparse
import json
import shutil
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import SAC

from src.approach_curriculum import ApproachCurriculumEnv
from src.clearance_training import ClearanceTrainingEnv, PROFILE
from tools.run_clearance_experiment import ROOT, sha, single_trainer, write_json
from tools.run_parallel_estimated_motion import existing_allocation

RUNS = Path('D:/DroneSoccerData/runs')
MAIN_SEEDS = tuple(range(11244720, 11244784))
SMOKE_SEEDS = (11244788, 11244789)
MODES = ('deterministic', 'stochastic_0', 'stochastic_1')
SOURCE = ROOT/'trainedModels/seenflag-s1-level1.zip'


def compare(rows, seeds, smoke=False):
    """Fixed diagnostic gate, not a policy acceptance or best-mode selection."""
    keys = [(r['seed'], r['mode']) for r in rows]
    expected = {(s, mode) for s in seeds for mode in MODES}
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError('Every declared matched reset/mode required exactly once')
    by = {(r['seed'], r['mode']): r for r in rows}
    for seed in seeds:
        trio = [by[seed, mode] for mode in MODES]
        if any(r['initial'] != trio[0]['initial'] for r in trio):
            raise ValueError('Matched reset state or dynamics differs')
        for r in trio:
            if not (1 <= r['steps'] <= 600) or r['status'] != 'complete':
                raise ValueError('Only complete bounded episodes')
            if r['repeat'] and (r['crashed'] or r['wall_steps'] or r['steps'] != 600 or r['passes'] < 2):
                raise ValueError('Invalid clean-repeat accounting')

    def group(ss):
        totals = {mode: {metric: sum(by[s, mode][metric] for s in ss)
                         for metric in ('first', 'returned', 'repeat', 'crashed')}
                  for mode in MODES}
        gain = totals['deterministic']['repeat'] - sum(totals[m]['repeat'] for m in MODES[1:])/2
        crash_drop = sum(totals[m]['crashed'] for m in MODES[1:])/2 - totals['deterministic']['crashed']
        return dict(layouts=len(ss), totals=totals, repeat_gain=gain, crash_reduction=crash_drop)

    groups = dict(all=group(seeds), first_half=group(seeds[:len(seeds)//2]),
                  second_half=group(seeds[len(seeds)//2:]),
                  approach=group([s for s in seeds if s % 2 == 0]),
                  full_task=group([s for s in seeds if s % 2]))
    reasons = []
    if smoke: reasons.append('Software smoke cannot justify a training change')
    if groups['all']['repeat_gain'] < 8 or groups['all']['crash_reduction'] < 8:
        reasons.append('Overall benefit below eight of 64 layouts')
    for name in ('first_half', 'second_half', 'approach', 'full_task'):
        if groups[name]['repeat_gain'] <= 0 or groups[name]['crash_reduction'] <= 0:
            reasons.append(name+': benefit not positive in both outcomes')
    return dict(groups=groups, supports_sampling_mismatch=not reasons, reasons=reasons,
                policy_promoted=False, acceptance=False, physical_flight=False)


def run_episode(model, mode, seed, index, folder, check):
    check()
    # Environment draws have an independent local RNG. Policy draws are explicit.
    torch.manual_seed(11605000 + 2*index + (mode == 'stochastic_1'))
    torch.cuda.manual_seed_all(11605000 + 2*index + (mode == 'stochastic_1'))
    env = ClearanceTrainingEnv(ApproachCurriculumEnv(level=1, randomise=True, hanger=True),
                               'clearance', trace_dir=folder)
    try:
        public, _ = env.reset(seed=seed)
        e = env.unwrapped
        initial = dict(position=e.pos.tolist(), heading=float(e.heading),
                       centre=e.hoop_centre.tolist(), normal=e.hoop_normal.tolist(),
                       dynamics=asdict(e.dyn), public85=public.tolist())
        # Truth above is diagnostic only; actor sees this public85 vector alone.
        before_crossing = False
        while True:
            check()
            with torch.inference_mode():
                action, _ = model.predict(public, deterministic=mode == 'deterministic')
            public, _, done, cut, info = env.step(action)
            if not env.audit.count and (done or e.wall_steps): before_crossing = True
            if done or cut: break
        return dict(seed=seed, mode=mode, status='complete', initial=initial,
                    steps=e.steps, passes=env.audit.count, first=bool(env.audit.count and not before_crossing),
                    returned=env.audit.returns > 0, crashed=bool(done), wall_steps=e.wall_steps,
                    outcome=info['outcome'],
                    repeat=bool(env.audit.count >= 2 and not done and not e.wall_steps and e.steps == 600))
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    started = time.monotonic()
    seeds = SMOKE_SEEDS if args.smoke else MAIN_SEEDS
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    def check():
        if (out/'STOP').exists() or time.monotonic()-started >= 600:
            raise RuntimeError('STOP or 600-second total budget reached')
    rows = []
    try:
        with single_trainer(ROOT.parent/'.clearance-training.lock'):
            allocation = existing_allocation(RUNS, list(seeds))
            allocation['sharing'] = 'One diagnostic process; matched resets across three unchanged-policy action modes; no fitting'
            write_json(out/'seed_allocation.json', allocation)
            torch.set_num_threads(1); torch.set_num_interop_threads(1)
            if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
            torch.cuda.set_per_process_memory_fraction(.15)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            if sha(SOURCE) != PROFILE.sha256: raise ValueError('Source changed')
            files = list((ROOT/'src').glob('*.py')) + [Path(__file__),
                ROOT/'tools/run_clearance_experiment.py', ROOT/'tools/run_parallel_estimated_motion.py',
                ROOT/'tools/parallel_training.py', ROOT/'tools/validate_flight_policies.py',
                ROOT/'tests/test_policy_sampling_comparison.py',
                ROOT/'experiments/faisal/docs/POLICY_SAMPLING_COMPARISON_20261008.md']
            sources = {}
            for path in files:
                check(); rel = path.resolve().relative_to(ROOT)
                target = out/'source'/rel; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target); sources[rel.as_posix()] = sha(target)
            write_json(out/'protocol.json', dict(kind='Original85 SAC sampling comparison; FIT diagnostic only',
                training_seeds=list(seeds), seed_role='Reserved diagnostic FIT; no optimization', modes=MODES,
                smoke=args.smoke, source_sha256=PROFILE.sha256, sources=sources,
                max_seconds=600, max_episode_steps=600, cuda_allocator_fraction=.15,
                command_rng='11605000 + 2*layout_index + stochastic_replica_index',
                gate='repeat gain and crash reduction >=8/64 versus mean of two stochastic attempts; positive in both halves and both reset strata',
                failure='Close; no repetitions, added seeds, noise-scale sweep or automatic optimization',
                success='Supports preparing a separate bounded learning experiment only; not policy improvement',
                torch_version=torch.__version__, numpy_version=np.__version__,
                gpu=torch.cuda.get_device_name(), precision='float32/noTF32', acceptance=False, physical_flight=False))
            model = SAC.load(SOURCE, device='cuda')
            model.policy.set_training_mode(False)
            if model.observation_space.shape != (85,) or model.action_space.shape != (4,):
                raise ValueError('Wrong actor contract')
            before = {k: v.detach().clone() for k, v in model.actor.state_dict().items()}
            write_json(out/'status.json', dict(status='running', completed=0, total=len(seeds)*3))
            for index, seed in enumerate(seeds):
                for mode in MODES:
                    row = run_episode(model, mode, seed, index, out/'trajectories'/mode/str(seed), check)
                    rows.append(row)
                    with (out/'records.jsonl').open('a', encoding='utf-8') as f: f.write(json.dumps(row)+'\n')
                write_json(out/'status.json', dict(status='running', completed=len(rows), total=len(seeds)*3))
            check()
            assert all(torch.equal(v, model.actor.state_dict()[k]) for k, v in before.items())
            assert sha(SOURCE) == PROFILE.sha256
            for rel, digest in sources.items():
                assert sha(ROOT/rel) == digest and sha(out/'source'/rel) == digest
            result = compare(rows, seeds, args.smoke)
            result.update(status='complete', elapsed_seconds=time.monotonic()-started, episodes=len(rows),
                          actor_unchanged=True, smoke=args.smoke)
            write_json(out/'result.json', result)
            write_json(out/'evidence.json', {str(p.relative_to(out)): sha(p) for p in out.rglob('*')
                                            if p.is_file() and p.name != 'status.json'})
            write_json(out/'status.json', dict(status='complete', completed=len(rows), total=len(seeds)*3))
            print(json.dumps(result), flush=True)
    except Exception as exc:
        write_json(out/'status.json', dict(status='failed', completed=len(rows), error=repr(exc)))
        (out/'failure.txt').write_text(traceback.format_exc(), encoding='utf-8')
        raise


if __name__ == '__main__': main()
