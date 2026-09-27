"""GPU training, staged lessons, held-out evaluation and durable resume.

python -m src.live --steps 1000000 --port 8773
python -m src.live --name EXISTING_RUN --resume --port 8773
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict
import json
import hashlib
import shutil
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from src.params import ROOT
from src.sim import CRASHES
from src.curriculum import REWARD_VERSION
from src.guided_curriculum import GuidedLessonEnv, STAGES
from src.unified_curriculum import MixedLessonEnv as LessonEnv, UnifiedCurriculum as Curriculum, training_mix
from src.unified_policy import UnifiedSAC, start_from_checkpoint, VERSION, REFILL_TRANSITIONS
from src.unified_evaluation import evaluate


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def display_mix(stage):
    return [{'stage': i, 'name': STAGES[i], 'probability': p}
            for i, p in enumerate(training_mix(stage))]


class Session:
    def __init__(self, steps, name, resume=False, eval_every=10000, eval_episodes=25,
                 buffer_size=100000, learning_starts=0, source_checkpoint=None, source_config=None):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Run name must be a single directory name')
        if buffer_size < REFILL_TRANSITIONS:
            raise ValueError('Replay capacity must fit the initial student-policy experience')
        self.out = ROOT / 'runs' / name
        self.out.mkdir(parents=True, exist_ok=resume)
        self.resume_from_disk = resume
        self.source_checkpoint = Path(source_checkpoint).resolve() if source_checkpoint else None
        self.source_config = Path(source_config).resolve() if source_config else None
        self.eval_every, self.eval_episodes = eval_every, eval_episodes
        self.buffer_size, self.learning_starts = buffer_size, learning_starts
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.slow = threading.Event()
        self.wake = threading.Event()
        self.closed = threading.Event()
        self.state = {'mode': 'Starting', 'steps': 0, 'target': steps,
                      'folder': str(self.out), 'device': 'CUDA', 'history': [],
                      'total_passes': 0, 'episodes': 0, 'stage': 4, 'lesson': STAGES[4],
                      'experiment': 'Unified training', 'reward_version': REWARD_VERSION,
                      'unified_version': VERSION, 'training_mix': display_mix(4),
                      'source_checkpoint_steps': 0, 'imitation_updates': 0}
        self.curriculum = Curriculum()
        self.model = None

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def target(self):
        with self.lock:
            return self.state['target']

    def control(self, command):
        with self.lock:
            if command == 'extend':
                self.state['target'] += 100000
                if self.state['mode'] == 'Replay':
                    self.stop.clear()
                    self.wake.set()
            elif command == 'resume':
                if self.state['mode'] != 'Replay':
                    raise ValueError('Resume is available after saving')
                if self.state['target'] <= self.state['steps']:
                    self.state['target'] = self.state['steps'] + 100000
                self.stop.clear()
                self.wake.set()
            elif command == 'stop':
                self.stop.set()
            elif command == 'slow':
                self.slow.set()
            elif command == 'fast':
                self.slow.clear()
            else:
                raise ValueError('Unknown command')

    def frame(self, env, info, path, **values):
        phi = np.linspace(0, 2*np.pi, 65)
        ring = (env.hoop_centre + np.outer(np.cos(phi), env._hoop_h)
                + np.outer(np.sin(phi), env._hoop_v))
        inner = (env.hoop_centre
                 + np.outer(np.cos(phi), (np.linalg.norm(env._hoop_h) - env.hoop.thickness) * env._axis_h)
                 + np.outer(np.sin(phi), (np.linalg.norm(env._hoop_v) - env.hoop.thickness) * env._axis_v))
        self.update(position=info['drone_pos'].tolist(), heading=float(info['heading']),
                    hoop=ring.tolist(), hoop_inner=inner.tolist(), centre=env.hoop_centre.tolist(),
                    drone_size=[env.drone.length, env.drone.width, env.drone.height],
                    normal=env.hoop_normal.tolist(), cage=[env.cage.length, env.cage.width, env.cage.height],
                    hanger_bottom=float(env._hanger_bottom), hanger_width=env.hoop.hanger_half_width,
                    margin=env.cage.margin, min_altitude=env.cage.min_altitude,
                    path=list(path), outcome=info['outcome'], passes=int(info['passes']),
                    episode_seconds=info['steps'] * env.drone.dt, **values)

    def save(self, callback):
        """Two snapshot slots; commit the pointer only after every file is complete.

        Called between learn chunks, after SB3 has stored the final transition.
        The active slot remains usable even if writing the other one is interrupted.
        """
        pointer = self.out / 'resume.json'
        active = json.loads(pointer.read_text())['slot'] if pointer.exists() else 1
        slot = 1-active
        folder = self.out / f'resume-{slot}'
        folder.mkdir(exist_ok=True)
        with open(folder / 'model.zip.tmp', 'wb') as f:
            self.model.save(f)
        with open(folder / 'replay_buffer.pkl.tmp', 'wb') as f:
            self.model.save_replay_buffer(f)
        with self.lock:
            status = dict(self.state)
        status.update(steps=self.model.num_timesteps, updates=self.model._n_updates,
                      imitation_updates=getattr(self.model, 'imitation_updates', 0))
        saved = {'version': 2, 'curriculum': self.curriculum.state(), 'status': status,
                 'recent': list(callback.recent), 'history': list(callback.history),
                 'total_passes': callback.total_passes, 'episodes': callback.episodes,
                 'elapsed': callback.elapsed(), 'last_evaluated': self.last_evaluated}
        for name in ('model.zip', 'replay_buffer.pkl'):
            os.replace(folder / (name+'.tmp'), folder / name)
        atomic_json(folder / 'state.json', saved)
        atomic_json(pointer, {'slot': slot, 'step': self.model.num_timesteps})
        with open(self.out / 'model.zip.tmp', 'wb') as f:
            self.model.save(f)
        os.replace(self.out / 'model.zip.tmp', self.out / 'model.zip')
        self.update(saved=str(self.out / 'model.zip'))
        atomic_json(self.out / 'progress.json', status)

    def load(self, env):
        config = json.loads((self.out / 'config.json').read_text())
        if config.get('unified_version') != VERSION:
            raise ValueError('Unified training version differs; use a fresh run.')
        if config.get('reward_version') != REWARD_VERSION:
            raise ValueError('Reward version differs; start a fresh run and replay buffer.')
        geometry = {k: asdict(getattr(env.unwrapped, k)) for k in ('cage','drone','hoop','camera','pad')}
        if config['geometry'] != geometry:
            raise ValueError('Geometry changed since this run. Use a new run to avoid stale replay data.')
        slot = json.loads((self.out / 'resume.json').read_text())['slot']
        folder = self.out / f'resume-{slot}'
        saved = json.loads((folder / 'state.json').read_text())
        self.curriculum = Curriculum(saved['curriculum'])
        env.unwrapped.set_stage(self.curriculum.stage)
        self.model = UnifiedSAC.load(folder / 'model.zip', env=env, device='cuda')
        self.model.load_replay_buffer(folder / 'replay_buffer.pkl')
        self.model.set_random_seed(config['seed'] + self.model.num_timesteps)
        self.eval_every, self.eval_episodes = config['eval_every'], config['eval_episodes']
        self.learning_starts = config['learning_starts']
        self.last_evaluated = saved['last_evaluated']
        requested_target = self.target()
        self.update(**saved['status'])
        self.update(target=max(requested_target, self.target()), mode='Starting',
                    total_passes=saved['total_passes'], episodes=saved['episodes'])
        return saved

    def run_evaluation(self):
        self.update(mode='Evaluating', phase='Held-out checkpoint evaluation')
        result = evaluate(self.model, self.curriculum.stage, self.eval_episodes,
                          lambda suite, n, total: self.update(eval_progress=f'{suite}: {n}/{total}'))
        folder = self.out / 'evaluations'
        folder.mkdir(exist_ok=True)
        atomic_json(folder / f'{self.model.num_timesteps}.json', result)
        compact = {k: {field: v for field, v in suite.items() if field != 'records'}
                   for k, suite in result['suites'].items()}
        self.last_evaluated = self.model.num_timesteps
        benchmark = compact['original_centered']['success_rate']
        previous = self.state.get('benchmark_streak', 0)
        self.update(evaluation={'step': result['step'], 'stage': result['stage'],
                                'episodes_per_suite': self.eval_episodes, 'suites': compact}, eval_progress='',
                    benchmark_streak=previous+1 if benchmark >= .8 else 0)

    def run(self):
        try:
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required; no CPU fallback.')
            env = Monitor(LessonEnv())
            self.last_evaluated = -1
            saved = {}
            if self.resume_from_disk:
                saved = self.load(env)
            else:
                if self.source_checkpoint is None or self.source_config is None:
                    raise ValueError('A new unified run requires a checkpoint and its source config.')
                source_config = json.loads(self.source_config.read_text())
                geometry = {k: asdict(getattr(env.unwrapped, k))
                            for k in ('cage', 'drone', 'hoop', 'camera', 'pad')}
                if source_config.get('geometry') != geometry or source_config.get('reward_version') != REWARD_VERSION:
                    raise ValueError('Source policy geometry or reward version is incompatible.')
                source_hash = hashlib.sha256(self.source_checkpoint.read_bytes()).hexdigest()
                shutil.copy2(self.source_checkpoint, self.out/'source_model.zip')
                shutil.copy2(self.source_config, self.out/'source_config.json')
                self.model, source = start_from_checkpoint(
                    self.out/'source_model.zip', env, device='cuda', seed=7,
                    buffer_size=self.buffer_size, tensorboard_log=self.out/'tensorboard')
                self.update(source_checkpoint_steps=source['steps'], source_checkpoint=str(self.source_checkpoint),
                            source_sha256=source_hash, source_history=source)
                atomic_json(self.out/'config.json', {
                    'version': 3, 'seed': 7, 'device': 'cuda', 'threads': 1,
                    'reward_version': REWARD_VERSION, 'experiment': 'Unified training',
                    'unified_version': VERSION,
                    'source_checkpoint': str(self.source_checkpoint), 'source_sha256': source_hash,
                    'source_history': source, 'start_focus_stage': 4,
                    'initial_training_mix': display_mix(4),
                    'mixture': '60% current starting layouts; 40% earlier layouts; current task rewards and dynamics',
                    'new_demonstrations': 0, 'new_imitation_updates': 0,
                    'replay_refill_transitions': REFILL_TRANSITIONS,
                    'learning_starts': self.learning_starts, 'buffer_size': self.buffer_size,
                    'eval_every': self.eval_every, 'eval_episodes': self.eval_episodes,
                    'geometry': {k: asdict(getattr(env.unwrapped,k))
                                 for k in ('cage','drone','hoop','camera','pad')}})
            if next(self.model.policy.parameters()).device.type != 'cuda':
                raise RuntimeError('Policy is not on CUDA')
            self.update(device=torch.cuda.get_device_name(0))
            callback = LiveCallback(self, saved)
            if not self.resume_from_disk:
                # Establish the inherited policy's validation before any updates.
                self.model.save(self.out/'initial_model.zip')
                self.run_evaluation()
                self.save(callback)
            while not self.closed.is_set():
                self.update(mode='Training')
                while self.model.num_timesteps < self.target() and not self.stop.is_set():
                    # Bound control latency and save only at a complete transition.
                    steps = min(100, self.target()-self.model.num_timesteps,
                                self.eval_every-self.model.num_timesteps % self.eval_every)
                    self.model.learn(steps, callback=callback, reset_num_timesteps=False, tb_log_name='SAC')
                    self.update(steps=self.model.num_timesteps, updates=self.model._n_updates)
                    if self.model.num_timesteps % self.eval_every == 0:
                        checkpoints = self.out/'checkpoints'
                        checkpoints.mkdir(exist_ok=True)
                        self.model.save(checkpoints/f'model_{self.model.num_timesteps}_steps.zip')
                        self.update(mode='Saving')
                        self.save(callback)
                        if not self.stop.is_set():
                            self.run_evaluation()
                            self.save(callback)
                        self.update(mode='Training')
                self.update(mode='Saving', steps=self.model.num_timesteps)
                self.save(callback)
                if not self.stop.is_set() and self.last_evaluated != self.model.num_timesteps:
                    self.run_evaluation()
                    self.save(callback)
                self.update(mode='Replay')
                self.replay_until_resume()
                # _last_obs remains valid for same-process continuation. Restore
                # from disk begins a new episode with the saved buffer/optimisers.
            env.close()
        except Exception:
            error = traceback.format_exc()
            (self.out/'error.txt').write_text(error)
            self.update(mode='Error', error=error)
            print(error, flush=True)

    def replay_until_resume(self):
        replay = GuidedLessonEnv(stage=self.curriculum.stage)
        episode = 0
        while not self.wake.is_set() and not self.closed.is_set():
            obs, info = replay.reset(seed=2_000_000+episode)
            path = deque(maxlen=600)
            while not self.wake.is_set() and not self.closed.is_set():
                obs, _, term, trunc, info = replay.step(self.model.predict(obs, deterministic=True)[0])
                path.append(info['drone_pos'].tolist())
                self.frame(replay, info, path, replay_episode=episode+1,
                           episode_lesson=STAGES[self.curriculum.stage])
                self.wake.wait(.1)
                if term or trunc:
                    self.wake.wait(.7)
                    break
            episode += 1
        replay.close()
        self.wake.clear()


class LiveCallback(BaseCallback):
    def __init__(self, session, saved=None):
        super().__init__()
        saved = saved or {}
        self.session = session
        self.recent = deque(saved.get('recent', []), maxlen=200)
        self.history = deque(saved.get('history', []), maxlen=300)
        self.path = deque(maxlen=600)
        self.last_frame = 0
        self.started = time.monotonic()
        self.previous_elapsed = saved.get('elapsed', 0.)
        self.start_step = session.state['steps']
        self.total_passes = saved.get('total_passes', 0)
        self.episodes = saved.get('episodes', 0)
        self.last_passes = 0

    def elapsed(self):
        return self.previous_elapsed + time.monotonic()-self.started

    def _on_step(self):
        info = self.locals['infos'][0]
        env = self.training_env.envs[0].unwrapped
        done = bool(self.locals['dones'][0])
        self.total_passes += max(0, int(info['passes'])-self.last_passes)
        self.last_passes = 0 if done else int(info['passes'])
        if done:
            self.episodes += 1
            record = {k: info[k] for k in ('outcome','passes','steps','stage','focus_stage')}
            record['reward'] = float(info['episode']['r'])
            self.recent.append(record)
            if self.session.curriculum.record(info):
                env.set_stage(self.session.curriculum.stage)
            self.path.clear()
        else:
            self.path.append(info['drone_pos'].tolist())
        curriculum = self.session.curriculum
        # Counters update on every transition, even if rendering is throttled.
        self.session.update(total_passes=self.total_passes, episodes=self.episodes,
                            stage=curriculum.stage, lesson=STAGES[curriculum.stage],
                            episode_lesson=STAGES[info['stage']], training_mix=display_mix(curriculum.stage),
                            lesson_rate=curriculum.rate, lesson_window=len(curriculum.recent))
        now = time.monotonic()
        if now-self.last_frame >= .1:
            self.last_frame = now
            metrics = dict(steps=self.num_timesteps, elapsed=self.elapsed(),
                           speed=(self.num_timesteps-self.start_step)/max(now-self.started,.01),
                           level=env.level, updates=self.model._n_updates,
                           imitation_updates=getattr(self.model, 'imitation_updates', 0),
                           gpu_memory_mb=torch.cuda.memory_allocated()/1024**2,
                           phase=('Collecting student-policy experience before SAC updates'
                                  if self.model.replay_buffer.size() < REFILL_TRANSITIONS
                                  else 'Learning one SAC policy across practice starts'))
            if self.recent:
                minutes = sum(i['steps'] for i in self.recent)*.1/60
                metrics.update(pass_rate=sum(i['passes'] for i in self.recent)/minutes,
                               crash_rate=sum(i['outcome'] in CRASHES for i in self.recent)/len(self.recent))
            if not done:
                self.session.frame(env, info, self.path, **metrics)
            else:
                self.session.update(**metrics)
        if self.num_timesteps % 1000 == 0:
            self.history.append({'step': self.num_timesteps,
                                 'reward': float(np.mean([i['reward'] for i in self.recent])) if self.recent else 0})
            self.session.update(history=list(self.history))
            with self.session.lock:
                status = dict(self.session.state)
            atomic_json(self.session.out/'progress.json',status)
        if self.session.slow.is_set():
            self.session.stop.wait(.1)
        return True


def serve(session, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            if self.path == '/':
                body, mime = Path(__file__).with_name('live.html').read_bytes(), 'text/html; charset=utf-8'
            elif self.path == '/state':
                with session.lock:
                    body = json.dumps(session.state).encode()
                mime = 'application/json'
            else:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Cache-Control', 'no-store')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self):
            if self.headers.get('Origin') != f'http://127.0.0.1:{port}':
                self.send_error(403)
                return
            try:
                session.control(self.path.lstrip('/'))
            except ValueError as e:
                self.send_error(400, str(e))
                return
            self.send_response(204)
            self.end_headers()

    server = ThreadingHTTPServer(('127.0.0.1', port), Handler)
    threading.Thread(target=session.run, daemon=True).start()
    print(f'Live GPU viewer: http://127.0.0.1:{port}', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=1000000)
    parser.add_argument('--name', default='unified-'+time.strftime('%Y%m%d-%H%M%S'))
    parser.add_argument('--port', type=int, default=8773)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--source-checkpoint', type=Path, default=ROOT/'models'/'c520k.zip')
    parser.add_argument('--source-config', type=Path, default=ROOT/'models'/'c-source-config.json')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if not args.resume and (not args.source_checkpoint or not args.source_config):
        parser.error('A new run needs --source-checkpoint and --source-config')
    serve(Session(args.steps, args.name, resume=args.resume,
                  source_checkpoint=args.source_checkpoint, source_config=args.source_config), args.port)
