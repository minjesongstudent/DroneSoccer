"""One bounded, sequential CUDA comparison; never an acceptance or flight release.

python -m tools.run_clearance_experiment --out D:/DroneSoccerData/runs/<new-name>
Use --smoke for a short software-only check, not a learning experiment.
Creating a file named STOP in the output directory requests a saved stop.
"""
import argparse
from contextlib import contextmanager
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import pickle
import random
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.clearance_training import ClearanceTrainingEnv, PROFILE
from src.policy_runtime import LearnedPolicy, PolicySpec
from src.sim import HoopEnv
from tools.validate_flight_policies import SUITES, episode


ROOT = Path(__file__).resolve().parents[1]
ARMS = ('control', 'clearance')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all())


@contextmanager
def preserved_rng():
    state = rng_state()
    try:
        yield
    finally:
        random.setstate(state['python'])
        np.random.set_state(state['numpy'])
        torch.set_rng_state(state['torch'])
        torch.cuda.set_rng_state_all(state['cuda'])


@contextmanager
def single_trainer(lock_path):
    # An OS lock is released on process death; stale PID text cannot authorize a
    # second run. Other project trainers are checked separately before launch.
    import msvcrt
    with Path(lock_path).open('a+b') as handle:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b'0')
            handle.flush()
        handle.seek(0)
        try:
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError as exc:
            raise RuntimeError('Another clearance trainer owns the lock') from exc
        try:
            yield
        finally:
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)


class LearnedWarmupSAC(SAC):
    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        # Use the learned stochastic policy even before the first gradient step.
        return super()._sample_action(0, action_noise, n_envs)


def make_model(source, env, protocol):
    if sha(source) != PROFILE.sha256:
        raise ValueError('Source checkpoint changed')
    model = LearnedWarmupSAC.load(source, env=env, device='cuda',
        learning_rate=protocol['learning_rate'], learning_starts=protocol['warmup'],
        buffer_size=protocol['total_steps'], train_freq=1, gradient_steps=1,
        seed=protocol['algorithm_seed'], tensorboard_log=None, verbose=0)
    if model.observation_space.shape != (85,) or model.action_space.shape != (4,):
        raise ValueError('Wrong policy contract')
    for module in (model.actor, model.critic, model.critic_target):
        assert all(p.device.type == 'cuda' for p in module.parameters())
    assert model.log_ent_coef.device.type == 'cuda'
    assert model.replay_buffer.size() == 0
    # SAC's algorithm seed also seeds its VecEnv. Replace only that queued reset
    # seed with the declared FIT layout base; keep policy/minibatch RNG separate.
    model.get_env().seed(protocol['training_seed_base'])
    return model


def summarize(rows):
    return dict(episodes=len(rows), first_pass=sum(r['clean_first_pass'] for r in rows),
        crashes=sum(r['crashed'] for r in rows),
        returned=sum(r['returns_to_front'] > 0 for r in rows),
        clean_repeat=sum(r['clean_repeated_episode'] for r in rows),
        full_body_passes=sum(r['full_body_passes'] for r in rows))


def warmup_buffer_hashes(model):
    replay = model.replay_buffer
    used = replay.size()
    hashes = {}
    for name in ('observations', 'next_observations', 'actions', 'dones', 'rewards'):
        array = np.ascontiguousarray(getattr(replay, name)[:used])
        hashes[name] = hashlib.sha256(array.tobytes()).hexdigest()
    return hashes


def evaluate(checkpoint, folder, count):
    folder.mkdir(parents=True, exist_ok=False)
    spec = PolicySpec(checkpoint.stem, sha(checkpoint), PROFILE.lags)
    write_json(folder/'model.json', asdict(spec))
    with preserved_rng():
        policy = LearnedPolicy(checkpoint, spec)
        result = {}
        for suite, base in SUITES.items():
            target = folder/suite
            target.mkdir()
            rows = []
            for i in range(count):
                row = episode(policy, suite, base+i, target/f'{base+i}.npz')
                rows.append(row)
                with (target/'episodes.jsonl').open('a', encoding='utf-8') as log:
                    log.write(json.dumps(row)+'\n')
            result[suite] = summarize(rows)
            print('EVAL', checkpoint.stem, suite, json.dumps(result[suite]), flush=True)
        write_json(folder/'summary.json', result)
        return result


def retention_failures(reference, candidate):
    # Development stopping guards, NOT mission acceptance thresholds.
    reasons = []
    for suite, old in reference.items():
        new = candidate[suite]
        if new['first_pass'] < old['first_pass'] - 2:
            reasons.append(f'{suite}: first-pass count dropped by more than 2/25')
        if new['crashes'] > old['crashes'] + 3:
            reasons.append(f'{suite}: crashes increased by more than 3/25')
        if new['clean_repeat'] < old['clean_repeat'] - 3:
            reasons.append(f'{suite}: clean-repeat count dropped by more than 3/25')
    return reasons


class BudgetAndHealth(BaseCallback):
    def __init__(self, folder, study, arm, deadline, initial_updates):
        super().__init__()
        self.folder, self.study, self.arm = folder, study, arm
        self.deadline, self.initial_updates = deadline, initial_updates
        self.stop_reason = None

    def _on_step(self):
        if not np.isfinite(self.locals['rewards']).all():
            raise FloatingPointError('Non-finite collected rewards')
        if (self.study/'STOP').exists():
            self.stop_reason = 'user_stop_file'
        elif time.monotonic() >= self.deadline:
            self.stop_reason = 'two_hour_cap'
        if self.num_timesteps % 1000 == 0:
            if not all(torch.isfinite(p).all().item() for p in self.model.policy.parameters()):
                raise FloatingPointError('Non-finite model weights')
        if self.num_timesteps % 5000 == 0 or self.stop_reason:
            status = dict(arm=self.arm, steps=self.num_timesteps,
                new_updates=self.model._n_updates-self.initial_updates,
                replay_size=self.model.replay_buffer.size(), stop=self.stop_reason,
                remaining_seconds=max(0, self.deadline-time.monotonic()),
                pid=os.getpid(), device=str(self.model.device),
                utc=datetime.now(timezone.utc).isoformat())
            write_json(self.study/'status.json', status)
            print('TRAIN', json.dumps(status), flush=True)
        return self.stop_reason is None


def save_checkpoint(model, env, folder, label, initial_updates):
    dest = folder/label
    dest.mkdir()
    checkpoint = dest/'model.zip'
    model.save(checkpoint)
    model.save_replay_buffer(dest/'replay_buffer.pkl')
    torch.save(rng_state(), dest/'rng.pt')
    # SB3 models do not save the live simulator. Retain it separately, including
    # delay queues and history, so exact-state diagnostic continuation is possible.
    with (dest/'environment.pkl').open('wb') as file:
        pickle.dump(env, file, protocol=pickle.HIGHEST_PROTOCOL)
    torch.save(dict(last_obs=model._last_obs, last_episode_starts=model._last_episode_starts,
                    episode_num=model._episode_num), dest/'collector_state.pt')
    write_json(dest/'state.json', dict(steps=model.num_timesteps,
        new_updates=model._n_updates-initial_updates, replay_size=model.replay_buffer.size(),
        model_sha256=sha(checkpoint), profile=asdict(PROFILE),
        automatic_resume=False, note='Exact-state artifacts retained; no automatic restart.'))
    return checkpoint


def run_arm(arm, study, protocol, reference):
    folder = study/arm
    folder.mkdir()
    env = Monitor(ClearanceTrainingEnv(HoopEnv(level=1, randomise=True, hanger=True),
                                      arm, trace_dir=folder/'training_trajectories'))
    model = make_model(study/'source/trainedModels/seenflag-s1-level1.zip', env, protocol)
    initial_updates = model._n_updates
    source_weights = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    deadline = time.monotonic() + protocol['arm_seconds']
    callback = BudgetAndHealth(folder, study, arm, deadline, initial_updates)
    result = dict(arm=arm, status='running', evaluations=[], source_unchanged=True)
    write_json(study/'status.json', dict(arm=arm, stage='warmup', steps=0,
        pid=os.getpid(), device=str(model.device), utc=datetime.now(timezone.utc).isoformat()))
    final_checkpoint = None
    try:
        targets = [protocol['warmup']] + [protocol['warmup'] + n for n in protocol['learning_checks']]
        for i, target in enumerate(targets):
            # Chunking retains the actual environment and last observation.
            count = target - (0 if i == 0 else model.num_timesteps)
            model.learn(count, callback=callback, reset_num_timesteps=(i == 0), log_interval=1000)
            final_checkpoint = save_checkpoint(model, env, folder,
                                              f'step_{model.num_timesteps:06d}', initial_updates)
            if i == 0:
                exact = all(torch.equal(v, source_weights[k]) for k, v in model.policy.state_dict().items())
                if not exact or model._n_updates != initial_updates:
                    raise AssertionError('Policy changed during replay warmup')
                hashes = warmup_buffer_hashes(model)
                comparison = None
                if arm == 'clearance':
                    other = json.loads((study/'control/warmup_check.json').read_text())['buffer_hashes']
                    comparison = all(hashes[k] == other[k] for k in hashes if k != 'rewards')
                    if not comparison:
                        raise AssertionError('Paired warmup flights differed before optimization')
                write_json(folder/'warmup_check.json', dict(exact_policy_weights=True,
                    new_updates=0, steps=model.num_timesteps,
                    buffer_hashes=hashes, paired_flights_identical=comparison,
                    action_source='learned stochastic policy; no uniform random warmup'))
                del source_weights
            if callback.stop_reason:
                result['status'] = callback.stop_reason
                break
            if i > 0:
                guard = evaluate(final_checkpoint, folder/f'guard_{model.num_timesteps:06d}',
                                 protocol['guard_episodes'])
                reasons = [] if protocol['smoke'] else retention_failures(reference, guard)
                result['evaluations'].append(dict(steps=model.num_timesteps, metrics=guard,
                                                  retention_failures=reasons))
                write_json(folder/'result.json', result)
                if reasons:
                    result['status'] = 'stopped_retention_regression'
                    break
            if time.monotonic() >= deadline:
                result['status'] = 'two_hour_cap'
                break
        else:
            result['status'] = 'completed_transition_budget'
        result['steps'] = model.num_timesteps
        result['new_updates'] = model._n_updates-initial_updates
        result['final_checkpoint'] = str(final_checkpoint)
        # Full development comparison is separately bounded to 300 episodes;
        # training's two-hour clock never starts another optimizer update.
        if final_checkpoint is not None and not (study/'STOP').exists():
            result['final_development'] = evaluate(final_checkpoint, folder/'final_development',
                                                  protocol['final_episodes'])
    except BaseException:
        result['status'] = 'error'
        result['error'] = traceback.format_exc()
        (folder/'error.txt').write_text(result['error'], encoding='utf-8')
        try:
            result['error_checkpoint'] = str(save_checkpoint(model, env, folder, 'error_checkpoint', initial_updates))
        except Exception:
            result['save_error'] = traceback.format_exc()
        raise
    finally:
        env.close()
        result['source_unchanged'] = sha(ROOT/'trainedModels/seenflag-s1-level1.zip') == PROFILE.sha256
        write_json(folder/'result.json', result)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    return result


def freeze(study, smoke):
    study.mkdir(parents=True, exist_ok=False)
    files = list((ROOT/'src').glob('*.py'))
    files += [ROOT/'tools/run_clearance_experiment.py', ROOT/'tools/validate_flight_policies.py',
              ROOT/'tests/test_clearance_training.py', ROOT/'trainedModels/seenflag-s1-level1.zip']
    hashes = {}
    for src in files:
        rel = src.relative_to(ROOT)
        target = study/'source'/rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, target)
        hashes[str(rel)] = sha(target)
    try:
        source_commit = subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT,
                                                text=True, stderr=subprocess.DEVNULL).strip()
    except subprocess.CalledProcessError:
        source_commit = 'source snapshot; see file hashes'
    protocol = dict(kind='software smoke; no learning conclusions' if smoke else 'bounded development comparison',
        smoke=smoke, arms=list(ARMS), source=asdict(PROFILE), source_hashes=hashes,
        warmup=32 if smoke else 200000, total_steps=64 if smoke else 250000,
        check_every=16 if smoke else 10000, learning_rate=0.00003,
        learning_checks=[16,32] if smoke else [1000,10000,20000,30000,40000,50000],
        arm_seconds=120 if smoke else 7200, algorithm_seed=11500000,
        training_seed_base=11000000, seed_schedule='base + reset index, shared by both arms',
        guard_episodes=2 if smoke else 25, final_episodes=2 if smoke else 100,
        evaluation_seeds=SUITES, evaluation_max_steps=600,
        retention_stops=dict(first_pass_drop_greater_than=2, crash_increase_greater_than=3,
                             clean_repeat_drop_greater_than=3, denominator=25),
        physics='unchanged group HoopEnv, level1, hanger=True, randomise=True',
        actor_input='85 public camera/action-history inputs only',
        reward_control='unchanged', reward_clearance='remove 40 at centre crossing; pay 40 at full clearance',
        checkpoint_selection='retain all; no promotion; last candidate gets full development evaluation',
        acceptance=False, physical_flight=False, heartbeat_resumed=False,
        model_device='cuda', evaluation_device='cpu', gpu=torch.cuda.get_device_name(0),
        cuda=torch.version.cuda, torch=torch.__version__, python=sys.version,
        dependencies={n:importlib.metadata.version(n) for n in ['stable-baselines3','gymnasium','numpy','opencv-python']},
        source_commit=source_commit,
        started_utc=datetime.now(timezone.utc).isoformat(), pid=os.getpid())
    write_json(study/'protocol.json', protocol)
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    if sha(ROOT/'trainedModels/seenflag-s1-level1.zip') != PROFILE.sha256:
        raise ValueError('Source model checksum mismatch')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    args.out = args.out.resolve()
    with single_trainer(ROOT.parent/'.clearance-training.lock'):
        protocol = freeze(args.out, args.smoke)
        print('PROTOCOL', args.out/'protocol.json', flush=True)
        result = dict(kind=protocol['kind'], arms={}, acceptance=False, physical_flight=False)
        try:
            reference = evaluate(args.out/'source/trainedModels/seenflag-s1-level1.zip',
                                 args.out/'reference_guard', protocol['guard_episodes'])
            result['reference_guard'] = reference
            for arm in ARMS:
                if (args.out/'STOP').exists():
                    result['stopped_by_user'] = True
                    break
                result['arms'][arm] = run_arm(arm, args.out, protocol, reference)
                write_json(args.out/'result.json', result)
            result['status'] = 'stopped_by_user' if result.get('stopped_by_user') else 'comparison_finished'
        except BaseException:
            result['status'] = 'error'
            result['error'] = traceback.format_exc()
            raise
        finally:
            result['finished_utc'] = datetime.now(timezone.utc).isoformat()
            write_json(args.out/'result.json', result)
            write_json(args.out/'status.json', dict(status=result['status'], pid=os.getpid(),
                utc=result['finished_utc'], acceptance=False, physical_flight=False))
            lines = ['# Clearance reward comparison', '', f"Status: {result['status']}", '',
                     'Development experiment only. No model promoted; no simulation acceptance or physical flight release.', '',
                     '| Arm | Stop reason | New optimizer updates |', '|---|---|---:|']
            for arm, r in result['arms'].items():
                lines.append(f"| {arm} | {r['status']} | {r['new_updates']} |")
            lines += ['', 'See protocol.json for frozen sources/settings, result.json for outcomes,',
                      'and each arm directory for checkpoints, replay buffers and actual trajectories.',
                      'Replay from the source snapshot with the same Python environment and a NEW output directory:',
                      '', '`python -m tools.run_clearance_experiment --out <new-directory>`' + (' (add --smoke)' if args.smoke else ''), '']
            (args.out/'REPORT.md').write_text('\n'.join(lines), encoding='utf-8')


if __name__ == '__main__':
    main()
