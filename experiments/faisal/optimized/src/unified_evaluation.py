"""Fixed validation of the single policy; never sampled into training replay."""
from collections import Counter
from dataclasses import asdict

from src.curriculum import LessonEnv
from src.guided_curriculum import GuidedLessonEnv
from src.params import Drone
from src.sim import CRASHES, HoopEnv
from src.mission_observation import MissionObservation, VERSION as OBSERVATION_VERSION
from src.curriculum import REWARD_VERSION

VALIDATION_SEED_START = 5_000_000


def evaluate(model, stage, episodes=25, progress=None, seed_start=VALIDATION_SEED_START, front_memory=False, metric_ring=False, geometric_return=False, episode_reset=None):
    if episodes <= 0:
        raise ValueError('Evaluation requires at least one episode')
    if geometric_return:metric_ring=True
    reward_version=REWARD_VERSION
    if geometric_return:
        from src.return_geometry import GeometricReturnProgress,VERSION as reward_version
    if metric_ring:front_memory=True
    observation_version = OBSERVATION_VERSION
    if front_memory:
        from src.front_memory import FrontMemoryObservation
        from src.front_return_library import OBSERVATION_VERSION as observation_version
    if metric_ring:
        from src.ring_estimate import MetricRingObservation
        from src.metric_return_library import OBSERVATION_VERSION as observation_version
    suites = [('lesson', GuidedLessonEnv(stage=stage)),
              ('original_centered', LessonEnv(stage=0)),
              ('varied_approach', LessonEnv(stage=1)),
              ('full_task', HoopEnv(level=1.0)),
              ('full_task_delay_200ms', HoopEnv(level=1.0, drone=Drone(delay_steps=2)))]
    report = {'step': model.num_timesteps, 'stage': stage,
              'episodes_per_suite': episodes, 'seed_start': seed_start,
              'observation_version': observation_version, 'reward_version': reward_version,
              'policy_mode': 'deterministic', 'suites': {}, 'protocols': {}}
    for name, base_env in suites:
        single = name in ('original_centered', 'varied_approach') or (name == 'lesson' and stage < 4)
        env = MissionObservation(base_env, single_pass=single)
        if front_memory:
            env = FrontMemoryObservation(env)
        if metric_ring:env=MetricRingObservation(env)
        if geometric_return:env=GeometricReturnProgress(env)
        report['protocols'][name] = {
            'id': f'completion-{name}-v1', 'environment': type(base_env).__name__,
            'reward_version': reward_version, 'observation_version': observation_version,
            'stage': stage if name == 'lesson' else 0 if name == 'original_centered' else 1 if name == 'varied_approach' else None,
            'max_steps': 150 if single else 600,
            'success': 'passed' if single else 'timeout and passes >= 2',
            'deterministic': True,
            'geometry': {key: asdict(getattr(base_env, key))
                         for key in ('cage', 'drone', 'hoop', 'camera', 'pad')},
        }
        records = []
        try:
            for i in range(episodes):
                obs, _ = env.reset(seed=seed_start+i)
                if episode_reset is not None:
                    episode_reset()
                phases_after_pass = set()
                while True:
                    obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                    if info['passes'] > 0:
                        phases_after_pass.add(info.get('lap_phase', ''))
                    if term or trunc:
                        break
                records.append({'seed': seed_start+i, 'outcome': info['outcome'],
                                'passes': info['passes'], 'steps': info['steps'],
                                'returned_to_side': 'return_front' in phases_after_pass,
                                'returned_to_front': 'align_front' in phases_after_pass})
                if progress:
                    progress(name, i+1, episodes)
        finally:
            env.close()
        success = sum((r['outcome'] == 'passed' if single else
                       r['outcome'] == 'timeout' and r['passes'] >= 2) for r in records)
        report['suites'][name] = {
            'successful_episodes': success, 'success_rate': success/episodes,
            'episodes_with_pass_rate': sum(r['passes'] > 0 for r in records)/episodes,
            'crash_rate': sum(r['outcome'] in CRASHES for r in records)/episodes,
            'total_passes': sum(r['passes'] for r in records),
            'episodes_with_two_passes': sum(r['passes'] >= 2 for r in records),
            'returned_to_side_episodes': sum(r['returned_to_side'] for r in records),
            'returned_to_front_episodes': sum(r['returned_to_front'] for r in records),
            'outcomes': dict(Counter(r['outcome'] for r in records)), 'records': records}
    return report
