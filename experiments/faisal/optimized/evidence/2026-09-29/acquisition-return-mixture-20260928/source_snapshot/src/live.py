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
from src.completion_curriculum import CompletionEnv as LessonEnv, Curriculum, training_mix
from src.mission_observation import MissionObservation, VERSION as OBSERVATION_VERSION, OBSERVATION_DIM
from src.evaluation_guard import EvaluationGuard
from src.unified_policy import REFILL_TRANSITIONS
from src.anchored_policy import AnchoredSAC as UnifiedSAC, start_from_checkpoint, VERSION
from src.unified_evaluation import evaluate


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    os.replace(temp, path)


def display_mix(stage, return_practice=False):
    if return_practice:
        from src.return_mixture import display_mix as return_mix
        return return_mix(stage)
    return [{'stage': i, 'name': STAGES[i], 'probability': p}
            for i, p in enumerate(training_mix(stage))]


class Session:
    def __init__(self, steps, name, resume=False, eval_every=10000, eval_episodes=25,
                 buffer_size=100000, learning_starts=0, source_checkpoint=None, source_config=None,
                 return_library=None, return_validation=None, runs_dir=None, asymmetric_critic=None,
                 retrace_critic=None, recent_retrace=None, front_memory=None, reset_lessons=None, repair_anchor=None, metric_ring=None, reset_coverage=None, geometric_return=None, balanced_anchor=None, balanced_repair=None, source_anchor=None, standard_source=None):
        if Path(name).name != name or name in ('.', '..'):
            raise ValueError('Run name must be a single directory name')
        if buffer_size < REFILL_TRANSITIONS:
            raise ValueError('Replay capacity must fit the initial student-policy experience')
        self.out = (Path(runs_dir).resolve() if runs_dir else ROOT/'runs') / name
        self.out.mkdir(parents=True, exist_ok=resume)
        self.resume_from_disk = resume
        previous_config = json.loads((self.out/'config.json').read_text()) if resume else {}
        saved_asymmetry = bool(previous_config.get('asymmetric_critic', False))
        if resume and asymmetric_critic is not None and bool(asymmetric_critic) != saved_asymmetry:
            raise ValueError('Critic architecture cannot change on resume; use explicit actor transfer in a fresh run.')
        self.asymmetric_critic = saved_asymmetry if asymmetric_critic is None else bool(asymmetric_critic)
        saved_retrace = bool(previous_config.get('retrace_critic', False))
        saved_recent = bool(previous_config.get('recent_retrace', False))
        saved_standard_source=bool(previous_config.get('standard_source',False))
        if resume and standard_source is not None and bool(standard_source)!=saved_standard_source:
            raise ValueError('Standard source contract cannot change on resume')
        self.standard_source=saved_standard_source if standard_source is None else bool(standard_source)
        if self.standard_source:
            if source_anchor is False:raise ValueError('Standard source requires source reference')
            if balanced_anchor is True:raise ValueError('Standard source disables gradient balancing')
            source_anchor=True
            balanced_anchor=False
            if geometric_return is False:raise ValueError('Standard source retains geometric reward contract')
            geometric_return=True
        saved_source_anchor=bool(previous_config.get('source_anchor',False))
        if resume and source_anchor is not None and bool(source_anchor)!=saved_source_anchor:
            raise ValueError('Source anchor contract cannot change on resume')
        self.source_anchor=saved_source_anchor if source_anchor is None else bool(source_anchor)
        if self.source_anchor:
            if not self.standard_source and balanced_anchor is False:raise ValueError('Source anchor requires gradient balance')
            if balanced_repair or previous_config.get('balanced_repair',False):raise ValueError('Source anchor retains absolute KL acceptance')
            if not self.standard_source:balanced_anchor=True
        saved_balanced_repair=bool(previous_config.get('balanced_repair',False))
        if resume and balanced_repair is not None and bool(balanced_repair)!=saved_balanced_repair:
            raise ValueError('Balanced repair contract cannot change on resume')
        self.balanced_repair=saved_balanced_repair if balanced_repair is None else bool(balanced_repair)
        if self.balanced_repair:
            if balanced_anchor is False:raise ValueError('Balanced repair requires gradient balance')
            balanced_anchor=True
        saved_balance=bool(previous_config.get('balanced_anchor',False))
        if resume and balanced_anchor is not None and bool(balanced_anchor)!=saved_balance:
            raise ValueError('Balanced anchor contract cannot change on resume')
        self.balanced_anchor=saved_balance if balanced_anchor is None else bool(balanced_anchor)
        if self.balanced_anchor:
            if geometric_return is False:raise ValueError('Balanced anchor retains geometric return')
            geometric_return=True
        saved_geo=bool(previous_config.get('geometric_return',False))
        if resume and geometric_return is not None and bool(geometric_return)!=saved_geo:
            raise ValueError('Geometric reward contract cannot change on resume')
        self.geometric_return=saved_geo if geometric_return is None else bool(geometric_return)
        self.reward_version=REWARD_VERSION
        if self.geometric_return:
            if reset_coverage is False:raise ValueError('Geometric trial retains reset coverage')
            reset_coverage=True
        saved_coverage=bool(previous_config.get('reset_coverage',False))
        if resume and reset_coverage is not None and bool(reset_coverage)!=saved_coverage:
            raise ValueError('Reset coverage contract cannot change on resume')
        self.reset_coverage=saved_coverage if reset_coverage is None else bool(reset_coverage)
        if self.reset_coverage:
            if metric_ring is False:raise ValueError('Coverage trial retains metric policy')
            metric_ring=True
        saved_metric=bool(previous_config.get('metric_ring',False))
        if resume and metric_ring is not None and bool(metric_ring)!=saved_metric:
            raise ValueError('Metric ring contract cannot change on resume')
        self.metric_ring=saved_metric if metric_ring is None else bool(metric_ring)
        if self.metric_ring:
            if repair_anchor or previous_config.get('repair_anchor',False):raise ValueError('Metric trial retains original anchor')
            if reset_lessons is False:raise ValueError('Metric trial requires reset lessons')
            reset_lessons=True
        saved_repair=bool(previous_config.get('repair_anchor',False))
        if resume and repair_anchor is not None and bool(repair_anchor)!=saved_repair:
            raise ValueError('Anchor repair contract cannot change on resume')
        self.repair_anchor=saved_repair if repair_anchor is None else bool(repair_anchor)
        if self.repair_anchor:
            if reset_lessons is False:raise ValueError('Repair trial retains reset curriculum')
            reset_lessons=True
        saved_resets = bool(previous_config.get('reset_lessons', False))
        if resume and reset_lessons is not None and bool(reset_lessons)!=saved_resets:
            raise ValueError('Reset lesson contract cannot change on resume')
        self.reset_lessons=saved_resets if reset_lessons is None else bool(reset_lessons)
        if self.reset_lessons:
            if front_memory is False:raise ValueError('Reset lessons require front-memory policy')
            front_memory=True
        saved_front = bool(previous_config.get('front_memory', False))
        if resume and front_memory is not None and bool(front_memory) != saved_front:
            raise ValueError('Front memory cannot change on resume; use a fresh run.')
        self.front_memory = saved_front if front_memory is None else bool(front_memory)
        if self.front_memory:
            if recent_retrace is False:
                raise ValueError('Replay sampling cannot change for front memory.')
            recent_retrace = True
        if resume and recent_retrace is not None and bool(recent_retrace) != saved_recent:
            raise ValueError('Replay sampling cannot change on resume; use a fresh run.')
        self.recent_retrace = saved_recent if recent_retrace is None else bool(recent_retrace)
        if self.recent_retrace:
            if retrace_critic is False:
                raise ValueError('Recent replay requires Retrace.')
            retrace_critic = True
        if resume and retrace_critic is not None and bool(retrace_critic) != saved_retrace:
            raise ValueError('Critic update rule cannot change on resume; use a fresh run.')
        self.retrace_critic = saved_retrace if retrace_critic is None else bool(retrace_critic)
        if self.asymmetric_critic and self.retrace_critic:
            raise ValueError('Test critic context and multi-step targets as separate experiments.')
        self.model_class, self.transfer_actor, self.training_version = UnifiedSAC, start_from_checkpoint, VERSION
        self.critic_config = {'asymmetric_critic': self.asymmetric_critic, 'retrace_critic': self.retrace_critic}
        if self.asymmetric_critic:
            from src.asymmetric_policy import AsymmetricSAC, start_from_checkpoint as asymmetric_transfer, VERSION as asymmetric_version
            from src.critic_context import CONTEXT_DIM, VERSION as context_version
            self.model_class, self.transfer_actor, self.training_version = AsymmetricSAC, asymmetric_transfer, asymmetric_version
            self.critic_config.update(critic_context_version=context_version, critic_context_dim=CONTEXT_DIM,
                                     actor_uses_simulator_context=False)
        if self.retrace_critic:
            from src.retrace_policy import RetraceSAC, start_from_checkpoint as retrace_transfer, VERSION as retrace_version
            from src.retrace_policy import TRACE_LENGTH, TRACE_LAMBDA, REPLAY_VERSION, TARGET_VERSION
            self.model_class, self.transfer_actor, self.training_version = RetraceSAC, retrace_transfer, retrace_version
            self.critic_config.update(trace_length=TRACE_LENGTH, trace_lambda=TRACE_LAMBDA,
                                     retrace_target_version=TARGET_VERSION, retrace_replay_version=REPLAY_VERSION,
                                     actor_uses_simulator_context=False)
        if self.recent_retrace:
            from src.recent_retrace import RecentRetraceSAC, start_from_checkpoint as recent_transfer
            from src.recent_retrace import VERSION as recent_version, RECENT_LAP_COUNT
            self.model_class, self.transfer_actor, self.training_version = RecentRetraceSAC, recent_transfer, recent_version
            self.critic_config.update(recent_retrace=True, recent_sampling_version=recent_version,
                                     recent_lap_count=RECENT_LAP_COUNT)
        from src.return_practice import ReturnSnapshotLibrary
        self.library_class = ReturnSnapshotLibrary
        self.observation_version, self.observation_dim = OBSERVATION_VERSION, OBSERVATION_DIM
        self.observation_wrapper = lambda env: env
        if self.front_memory:
            from src.front_policy import FrontMemorySAC, start_from_checkpoint as front_transfer, VERSION as front_version
            from src.front_memory import FrontMemoryObservation
            from src.front_return_library import FrontReturnLibrary, OBSERVATION_VERSION as front_observation_version
            self.model_class, self.transfer_actor, self.training_version = FrontMemorySAC, front_transfer, front_version
            self.library_class = FrontReturnLibrary
            self.observation_wrapper = FrontMemoryObservation
            self.observation_version, self.observation_dim = front_observation_version, 115
            self.critic_config.update(front_memory=True, front_policy_version=front_version)
        if self.metric_ring:
            from src.metric_policy import MetricRingSAC, start_from_checkpoint as metric_transfer, VERSION as metric_version
            from src.metric_return_library import MetricReturnLibrary, OBSERVATION_VERSION as metric_observation_version
            from src.ring_estimate import MetricRingObservation, VERSION as estimate_version
            self.model_class,self.transfer_actor,self.training_version=MetricRingSAC,metric_transfer,metric_version
            self.library_class=MetricReturnLibrary
            self.observation_wrapper=lambda env:MetricRingObservation(FrontMemoryObservation(env))
            self.observation_version,self.observation_dim=metric_observation_version,119
            self.critic_config.update(metric_ring=True,metric_policy_version=metric_version,ring_estimate_version=estimate_version)
        if self.reset_coverage:
            from src.return_reset_curriculum import COVERAGE_VERSION
            self.critic_config.update(reset_coverage=True,reset_coverage_version=COVERAGE_VERSION)
        if self.geometric_return:
            from src.geometric_policy import GeometricSAC,start_from_checkpoint as geo_transfer
            from src.geometric_library import GeometricReturnLibrary
            from src.return_geometry import GeometricReturnProgress,VERSION as geo_version
            previous_wrapper=self.observation_wrapper
            self.observation_wrapper=lambda env:GeometricReturnProgress(previous_wrapper(env))
            self.model_class,self.transfer_actor,self.training_version=GeometricSAC,geo_transfer,geo_version
            self.library_class=GeometricReturnLibrary
            self.reward_version=geo_version
            self.critic_config.update(geometric_return=True,geometric_reward_version=geo_version)
        if self.balanced_anchor:
            from src.balanced_anchor_policy import BalancedAnchorSAC, start_from_checkpoint as balance_transfer, VERSION as balance_version
            self.model_class,self.transfer_actor=BalancedAnchorSAC,balance_transfer
            self.critic_config.update(balanced_anchor=True,balanced_anchor_version=balance_version)
        if self.balanced_repair:
            from src.balanced_repair_policy import BalancedRepairSAC,start_from_checkpoint as balanced_repair_transfer,VERSION as balanced_repair_version
            self.model_class,self.transfer_actor=BalancedRepairSAC,balanced_repair_transfer
            self.critic_config.update(balanced_repair=True,balanced_repair_version=balanced_repair_version,anchor_repair_minimum_fraction=.01)
        if self.source_anchor:
            from src.source_anchor_policy import SourceAnchorSAC,start_from_checkpoint as source_anchor_transfer,VERSION as source_anchor_version
            self.model_class,self.transfer_actor=SourceAnchorSAC,source_anchor_transfer
            self.critic_config.update(source_anchor=True,source_anchor_version=source_anchor_version,active_retention_reference='verified source actor',original_reference_preserved=True,original_reference_used_for_penalty=False)
        if self.standard_source:
            from src.standard_source_policy import StandardSourceSAC,start_from_checkpoint as standard_source_transfer,VERSION as standard_source_version
            self.model_class,self.transfer_actor=StandardSourceSAC,standard_source_transfer
            self.critic_config.update(standard_source=True,standard_source_version=standard_source_version,anchor_gradient_mode='standard',gradient_balancing_active=False)




        self.reset_scheduler=None
        self.reset_env=None
        if self.reset_lessons:
            from src.return_reset_curriculum import ResetScheduler, VERSION as reset_version
            self.reset_scheduler=ResetScheduler()
            self.critic_config.update(reset_lessons=True, reset_curriculum_version=reset_version)
        if self.repair_anchor:
            from src.repair_anchor_policy import RepairFrontSAC, start_from_checkpoint as repair_transfer, VERSION as repair_version
            self.model_class,self.transfer_actor,self.training_version=RepairFrontSAC,repair_transfer,repair_version
            self.critic_config.update(repair_anchor=True,repair_version=repair_version,anchor_repair_minimum_fraction=.01)
        return_library = return_library or previous_config.get('return_library')
        return_validation = return_validation or previous_config.get('return_validation')
        if bool(return_library) != bool(return_validation):
            raise ValueError('Provide separate training and development return banks together.')
        self.return_library = Path(return_library).resolve() if return_library else None
        self.return_validation = Path(return_validation).resolve() if return_validation else None
        self.return_selector = None
        if self.return_library:
            from src.return_selection import ReturnSkillSelector
            self.return_selector = ReturnSkillSelector()
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
                      'experiment': 'Unified training', 'reward_version': self.reward_version,
                      'unified_version': self.training_version, **self.critic_config,
                      'training_mix': self.training_mix(),
                      'source_checkpoint_steps': 0, 'imitation_updates': 0}
        self.curriculum = Curriculum()
        self.guard = EvaluationGuard()
        self.model = None

    def training_mix(self):
        if self.reset_coverage:
            return [{'name':'Centered single pass','probability':.2},{'name':'Varied single pass','probability':.2},{'name':'Complete front-start flight','probability':.2}]+[{'name':f'Fresh return lesson {k}','probability':.4/6} for k in range(6)]
        if self.reset_lessons:
            level=self.reset_scheduler.level
            return [{'name':'Centered single pass','probability':.2},
                    {'name':'Varied single pass','probability':.2},
                    {'name':'Complete front-start flight','probability':.2},
                    {'name':f'Fresh return lesson {level}','probability':.2},
                    {'name':'Earlier fresh return lessons','probability':.2}]
        return display_mix(getattr(getattr(self,'curriculum',None),'stage',4),bool(self.return_library))

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def target(self):
        with self.lock:
            return self.state['target']

    def control(self, command):
        with self.lock:
            if command in ('resume', 'extend') and (self.guard.should_stop or
                    (self.return_selector is not None and self.return_selector.should_stop) or
                    (self.reset_scheduler is not None and self.reset_scheduler.should_stop)):
                raise ValueError('Evaluation stopped this experiment. Review the results before starting a new run.')
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
                 'evaluation_guard': self.guard.to_dict(),
                 'return_selection': self.return_selector.to_dict() if self.return_selector else None,
                 'reset_scheduler': self.reset_scheduler.state() if self.reset_scheduler else None,
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
        if config.get('unified_version') != self.training_version:
            raise ValueError('Unified training version differs; use a fresh run.')
        if self.asymmetric_critic or self.retrace_critic:
            for key, value in self.critic_config.items():
                default = False if key in ('asymmetric_critic', 'retrace_critic') else None
                if config.get(key, default) != value:
                    raise ValueError('Critic context contract differs or update rule changed; use a fresh run.')
        if config.get('reward_version') != self.reward_version:
            raise ValueError('Reward version differs; start a fresh run and replay buffer.')
        if config.get('observation_version') != self.observation_version:
            raise ValueError('Observation version differs; use a fresh run.')
        if self.return_library:
            from src.return_practice import ReturnSnapshotLibrary
            for field, path in (('return_library', self.return_library), ('return_validation', self.return_validation)):
                if self.library_class(path).bank_sha256 != config[field+'_sha256']:
                    raise ValueError('Return bank changed since the saved run.')
        geometry = {k: asdict(getattr(env.unwrapped, k)) for k in ('cage','drone','hoop','camera','pad')}
        if config['geometry'] != geometry:
            raise ValueError('Geometry changed since this run. Use a new run to avoid stale replay data.')
        slot = json.loads((self.out / 'resume.json').read_text())['slot']
        folder = self.out / f'resume-{slot}'
        saved = json.loads((folder / 'state.json').read_text())
        if self.reset_lessons:
            from src.return_reset_curriculum import ResetScheduler
            restored_scheduler=ResetScheduler.restore(saved['reset_scheduler'])
            self.reset_scheduler=restored_scheduler
            if restored_scheduler.should_stop:self.stop.set()
            env.get_wrapper_attr('set_scheduler')(restored_scheduler)
        self.guard = EvaluationGuard.from_dict(saved['evaluation_guard'])
        if self.return_selector is not None:
            from src.return_selection import ReturnSkillSelector
            self.return_selector = ReturnSkillSelector.from_dict(saved['return_selection'])
            if self.return_selector.should_stop:
                self.stop.set()
        if self.guard.should_stop:
            self.stop.set()
        self.curriculum = Curriculum(saved['curriculum'])
        env.get_wrapper_attr('set_stage')(self.curriculum.stage)
        self.model = self.model_class.load(folder / 'model.zip', env=env, device='cuda')
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
                          lambda suite, n, total: self.update(eval_progress=f'{suite}: {n}/{total}'), **({'geometric_return': True} if self.geometric_return else {'metric_ring': True} if self.metric_ring else {'front_memory': True} if self.front_memory else {}))
        if self.return_validation:
            from src.return_evaluation import evaluate_return_practice
            suite, protocol = evaluate_return_practice(self.model, self.library_class(self.return_validation),
                progress=lambda name, n, total: self.update(eval_progress=f'{name}: {n}/{total}'))
            result['suites']['return_practice'] = suite
            result['protocols']['return_practice'] = protocol
        if self.reset_lessons:
            from src.return_reset_curriculum import evaluate_reset_lessons
            lessons=evaluate_reset_lessons(self.model,metric_ring=self.metric_ring,geometric_return=self.geometric_return,progress=lambda level,n,total:self.update(eval_progress=f'Reset lesson {level}: {n}/{total}'))
            decision=self.reset_scheduler.observe(lessons,result['suites']['original_centered']['success_rate'],result['suites']['varied_approach']['success_rate'])
            lesson_folder=self.out/'reset-evaluations';lesson_folder.mkdir(exist_ok=True)
            atomic_json(lesson_folder/f'{self.model.num_timesteps}.json',lessons)
            self.update(reset_curriculum=decision,reset_successes=lessons['successes'])
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
        decision = self.guard.observe(result)
        atomic_json(self.out/'selection.json', self.guard.to_dict())
        if self.guard.best_step == result['step']:
            with open(self.out/'best_model.zip.tmp', 'wb') as file:
                self.model.save(file)
            os.replace(self.out/'best_model.zip.tmp', self.out/'best_model.zip')
        self.update(selection=decision, best_step=self.guard.best_step,
                    stop_reason=self.guard.stop_reason)
        if self.guard.should_stop:
            self.stop.set()
        if self.return_selector is not None:
            skill = self.return_selector.observe(result)
            atomic_json(self.out/'return-selection.json', self.return_selector.to_dict())
            if skill['selected']:
                with open(self.out/'best_return_model.zip.tmp', 'wb') as file:
                    self.model.save(file)
                os.replace(self.out/'best_return_model.zip.tmp', self.out/'best_return_model.zip')
            self.update(return_selection=skill, best_return_step=skill['best_return_step'])
            if skill['should_stop'] and not self.guard.independent_test_needed:
                self.update(stop_reason=skill['stop_reason'])
                self.stop.set()

        if self.reset_scheduler is not None and self.reset_scheduler.should_stop:
            self.update(stop_reason=self.reset_scheduler.stop_reason)
            self.stop.set()

    def run(self):
        try:
            torch.set_num_threads(1)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is required; no CPU fallback.')
            bank_config = {}
            if self.return_library:
                from src.return_mixture import ReturnMixtureEnv
                from src.return_practice import ReturnSnapshotLibrary
                train_bank = self.library_class(self.return_library)
                validation_bank = self.library_class(self.return_validation)
                if (train_bank.manifest['split'] != 'train'
                        or validation_bank.manifest['split'] != 'development'):
                    raise ValueError('Return training and development banks must remain separate.')
                bank_config = {'return_library': str(self.return_library),
                               'return_validation': str(self.return_validation),
                               'return_library_sha256': train_bank.bank_sha256,
                               'return_validation_sha256': validation_bank.bank_sha256}
                if self.reset_lessons:
                    from src.return_reset_curriculum import ResetMixture
                    self.reset_env=ResetMixture(self.reset_scheduler,metric_ring=self.metric_ring,coverage=self.reset_coverage,geometric_return=self.geometric_return)
                    env=Monitor(self.reset_env)
                else:
                    env = Monitor(ReturnMixtureEnv(train_bank, observation_wrapper=self.observation_wrapper))
            else:
                if self.reset_lessons:raise ValueError('Reset trial requires unchanged return validation banks')
                env = Monitor(self.observation_wrapper(MissionObservation(LessonEnv())))
            if self.asymmetric_critic:
                from src.critic_context import CriticContextWrapper
                env = CriticContextWrapper(env)
            self.last_evaluated = -1
            saved = {}
            if self.resume_from_disk:
                saved = self.load(env)
            else:
                if self.source_checkpoint is None or self.source_config is None:
                    raise ValueError('A new unified run requires a checkpoint and its source config.')
                source_config = json.loads(self.source_config.read_text())
                if (source_config.get('observation_version') is not None
                        and source_config['observation_version'] != OBSERVATION_VERSION):
                    raise ValueError('Source observation meaning differs; dimension alone cannot validate actor transfer.')
                geometry = {k: asdict(getattr(env.unwrapped, k))
                            for k in ('cage', 'drone', 'hoop', 'camera', 'pad')}
                # A fresh warm start may use the frozen pre-fix policy. Its old
                # replay is never imported. Resume below still requires an exact
                # reward version match so incompatible rewards cannot be mixed.
                if (source_config.get('geometry') != geometry
                        or source_config.get('reward_version') not in
                        (REWARD_VERSION, 'lap-clearance-potential-v2', 'single-pass-clearance-v1')):
                    raise ValueError('Source policy geometry or reward version is incompatible.')
                source_hash = hashlib.sha256(self.source_checkpoint.read_bytes()).hexdigest()
                shutil.copy2(self.source_checkpoint, self.out/'source_model.zip')
                shutil.copy2(self.source_config, self.out/'source_config.json')
                self.model, source = self.transfer_actor(
                    self.out/'source_model.zip', env, device='cuda', seed=7,
                    buffer_size=self.buffer_size, tensorboard_log=self.out/'tensorboard')
                # Preserve each experiment's implementation before training;
                # subsequent source edits must not erase its reproducibility.
                snapshot = self.out/'source_snapshot'
                snapshot.mkdir()
                for directory in ('src', 'tests'):
                    shutil.copytree(ROOT/directory, snapshot/directory,
                                    ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
                for filename in ('README.md', 'requirements.txt', 'LAB_CHECK.md', 'RETURN_TRIAL.md',
                                 'RETURN_PRACTICE_TRIAL.md', 'ASYMMETRIC_CRITIC_TRIAL.md', 'RETRACE_TRIAL.md', 'RECENT_RETRACE_TRIAL.md',
                                 'FRONT_MEMORY_TRIAL.md', 'RETURN_RESET_TRIAL.md', 'ANCHOR_REPAIR_TRIAL.md', 'METRIC_RING_TRIAL.md', 'RESET_COVERAGE_TRIAL.md', 'GEOMETRIC_RETURN_TRIAL.md', 'BALANCED_ANCHOR_TRIAL.md', 'BALANCED_REPAIR_TRIAL.md', 'SOURCE_ANCHOR_TRIAL.md', 'STANDARD_SOURCE_TRIAL.md', 'OVERNIGHT_HANDOFF.md', 'pytest.ini'):
                    if (ROOT/filename).exists():
                        shutil.copy2(ROOT/filename, snapshot/filename)
                self.update(source_checkpoint_steps=source['steps'], source_checkpoint=str(self.source_checkpoint),
                            source_sha256=source_hash, source_history=source)
                atomic_json(self.out/'config.json', {
                    'version': 3, 'seed': 7, 'device': 'cuda', 'threads': 1,
                    'reward_version': self.reward_version, 'experiment': 'Unified training',
                    'unified_version': self.training_version, **self.critic_config,
                    'observation_version': self.observation_version, 'observation_dim': self.observation_dim,
                    'source_checkpoint': str(self.source_checkpoint), 'source_sha256': source_hash,
                    'source_reward_version': source_config.get('reward_version'),
                    'source_history': source, 'start_focus_stage': 4,
                    'initial_training_mix': self.training_mix(),
                    'mixture': ('40% single-pass;20% full flights;40% uniform six reset lessons' if self.reset_coverage else '40% single-pass;20% full flights;20% current reset lesson;20% earlier reset lessons' if self.reset_lessons else
                                '40% single-pass; 20% front laps; 20% corner laps; 20% authentic return continuations'
                                if self.return_library else '40% complete single-pass tasks; 40% front laps; 20% corner laps initially'),
                    **bank_config,
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
                # Validation selects a checkpoint; fresh seeds assess that choice.
                # Final testing never writes transitions or updates parameters.
                if self.guard.independent_test_needed:
                    self.run_independent_test()
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

    def run_independent_test(self):
        path = self.out/'independent-test.json'
        if path.exists():
            self.update(independent_test=json.loads(path.read_text()))
            return
        self.update(mode='Evaluating', phase='Independent completion test on 100 fresh layouts per suite')
        candidate = self.model_class.load(self.out/'best_model.zip', device='cuda')
        results = evaluate(candidate, self.curriculum.stage, episodes=100, seed_start=7_000_000,
                           **({'geometric_return': True} if self.geometric_return else {'metric_ring': True} if self.metric_ring else {'front_memory': True} if self.front_memory else {}),
                           progress=lambda suite, n, total: self.update(eval_progress=f'Fresh {suite}: {n}/{total}'))
        gates = {'original_centered': .85, 'varied_approach': .80,
                 'full_task': .80, 'full_task_delay_200ms': .80}
        accepted = all(results['suites'][key]['success_rate'] >= rate for key, rate in gates.items())
        results.update(accepted_in_simulation=accepted, acceptance_thresholds=gates,
                       physical_flight_validated=False)
        atomic_json(path, results)
        self.update(independent_test=results, eval_progress='')

    def replay_until_resume(self):
        replay = self.observation_wrapper(MissionObservation(GuidedLessonEnv(stage=self.curriculum.stage), single_pass=False))
        skill_replay = self.return_selector is not None and (self.out/'best_return_model.zip').exists()
        best_path = self.out/('best_return_model.zip' if skill_replay else 'best_model.zip')
        replay_model = (self.model_class.load(best_path, device='cuda')
                        if best_path.exists() else self.model)
        self.update(replay_checkpoint_step=(self.return_selector.best_return_step if skill_replay else self.guard.best_step),
                    replay_checkpoint_kind='return skill' if skill_replay else 'mission',
                    phase=('Replay of the selected return-skill policy; full mission completion remains separate'
                           if skill_replay else 'Replay of the best retained policy; learning is stopped'))
        episode = 0
        while not self.wake.is_set() and not self.closed.is_set():
            obs, info = replay.reset(seed=2_000_000+episode)
            path = deque(maxlen=600)
            while not self.wake.is_set() and not self.closed.is_set():
                obs, _, term, trunc, info = replay.step(replay_model.predict(obs, deterministic=True)[0])
                path.append(info['drone_pos'].tolist())
                self.frame(replay.unwrapped, info, path, replay_episode=episode+1,
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
            if not self.session.reset_lessons and self.session.curriculum.record(info):
                self.training_env.envs[0].get_wrapper_attr('set_stage')(self.session.curriculum.stage)
            self.path.clear()
        else:
            self.path.append(info['drone_pos'].tolist())
        curriculum = self.session.curriculum
        # Counters update on every transition, even if rendering is throttled.
        self.session.update(total_passes=self.total_passes, episodes=self.episodes,
                            stage=curriculum.stage, lesson=STAGES[curriculum.stage],
                            episode_lesson=info.get('practice_label', STAGES[info['stage']]),
                            training_mix=self.session.training_mix(),
                            lesson_rate=curriculum.rate, lesson_window=len(curriculum.recent))
        now = time.monotonic()
        if now-self.last_frame >= .1:
            self.last_frame = now
            metrics = dict(steps=self.num_timesteps, elapsed=self.elapsed(),
                           speed=(self.num_timesteps-self.start_step)/max(now-self.started,.01),
                           level=env.level, updates=self.model._n_updates,
                           imitation_updates=getattr(self.model, 'imitation_updates', 0),
                           gpu_memory_mb=torch.cuda.memory_allocated()/1024**2,
                           actor_updates=self.model.actor_updates,
                           anchor_accepted=self.model.anchor_accepted,
                           anchor_repairs=getattr(self.model,'anchor_repairs',0),
                           anchor_rejected=self.model.anchor_rejected,
                           anchor_mean_kl=self.model.last_anchor_kl,
                           critic_calibration_updates=self.model.critic_calibration_updates,
                           transfer_phase=self.model.transfer_phase,
                           phase={'replay_refill': 'Collecting experience; movement policy is frozen',
                                  'critic_calibration': 'Learning reward estimates; movement policy is frozen',
                                  'sac_training': 'Learning movements while protecting established passing behaviour'}[self.model.transfer_phase])
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
    parser.add_argument('--source-checkpoint', type=Path)
    parser.add_argument('--source-config', type=Path)
    parser.add_argument('--return-library', type=Path)
    parser.add_argument('--return-validation', type=Path)
    parser.add_argument('--eval-every', type=int, default=10000)
    parser.add_argument('--runs-dir', type=Path, help='Checkpoint storage directory; defaults to the project runs folder')
    critic = parser.add_mutually_exclusive_group()
    critic.add_argument('--asymmetric-critic', action='store_true', default=None,
                        help='Training-only simulator context for critics; actor observations remain 100 camera/history values')
    critic.add_argument('--retrace-critic', action='store_true', default=None,
                        help='Public-input critics with short importance-corrected soft-Retrace targets')
    critic.add_argument('--recent-retrace', action='store_true', default=None,
                        help='Retrace with mixed recent/all lap replay, retaining single-pass rehearsal')
    critic.add_argument('--front-memory', action='store_true', default=None,
                        help='Recent Retrace with appended public front-sighting memory')
    critic.add_argument('--reset-lessons', action='store_true', default=None, help='Fresh return reset curriculum with fixed lesson validation')
    critic.add_argument('--repair-anchor', action='store_true', default=None, help='Permit monotonic repair of already excessive reference KL; retain reset lessons')
    critic.add_argument('--metric-ring', action='store_true', default=None, help='Append camera-derived approximate ring centre; retain original anchor and reset curriculum')
    critic.add_argument('--reset-coverage', action='store_true', default=None, help='Metric policy with uniform sampling across all six fresh reset lessons')
    critic.add_argument('--standard-source', action='store_true', default=None, help='Source reference with standard weighted retention penalty, no gradient balancing')
    critic.add_argument('--source-anchor', action='store_true', default=None, help='Retain verified source actor; archive original reference separately')
    critic.add_argument('--balanced-repair', action='store_true', default=None, help='Gradient balance plus monotonic repair of infeasible reference KL')
    critic.add_argument('--balanced-anchor', action='store_true', default=None, help='Bound anchor gradient norm by RL gradient; preserve hard reference gate')
    critic.add_argument('--geometric-return', action='store_true', default=None, help='Recognize outside return after full front clearance; retain reset coverage')
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error('--steps must be positive')
    if args.eval_every <= 0:
        parser.error('--eval-every must be positive')
    if not args.resume and (not args.source_checkpoint or not args.source_config):
        parser.error('A new run needs --source-checkpoint and --source-config')
    serve(Session(args.steps, args.name, resume=args.resume,
                  eval_every=args.eval_every,
                  source_checkpoint=args.source_checkpoint, source_config=args.source_config,
                  return_library=args.return_library, return_validation=args.return_validation,
                  runs_dir=args.runs_dir, asymmetric_critic=args.asymmetric_critic,
                  retrace_critic=args.retrace_critic, recent_retrace=args.recent_retrace, front_memory=args.front_memory, reset_lessons=args.reset_lessons, repair_anchor=args.repair_anchor, metric_ring=args.metric_ring, reset_coverage=args.reset_coverage, geometric_return=args.geometric_return, balanced_anchor=args.balanced_anchor, balanced_repair=args.balanced_repair, source_anchor=args.source_anchor, standard_source=args.standard_source), args.port)
