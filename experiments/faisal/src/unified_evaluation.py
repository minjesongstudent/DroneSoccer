"""Fixed validation of the single policy; never sampled into training replay."""
from collections import Counter
from dataclasses import asdict

from src.curriculum import LessonEnv
from src.guided_curriculum import GuidedLessonEnv
from src.params import Drone
from src.sim import CRASHES, HoopEnv

VALIDATION_SEED_START = 5_000_000


def evaluate(model, stage, episodes=25, progress=None):
    if episodes <= 0:
        raise ValueError('Evaluation requires at least one episode')
    suites = [('lesson', GuidedLessonEnv(stage=stage)),
              ('original_centered', LessonEnv(stage=0)),
              ('varied_approach', LessonEnv(stage=1)),
              ('full_task', HoopEnv(level=1.0)),
              ('full_task_delay_200ms', HoopEnv(level=1.0, drone=Drone(delay_steps=2)))]
    report = {'step': model.num_timesteps, 'stage': stage,
              'episodes_per_suite': episodes, 'seed_start': VALIDATION_SEED_START,
              'policy_mode': 'deterministic', 'suites': {}, 'protocols': {}}
    for name, env in suites:
        single = name in ('original_centered', 'varied_approach') or (name == 'lesson' and stage < 4)
        report['protocols'][name] = {
            'id': f'unified-{name}-v1', 'environment': type(env).__name__,
            'stage': stage if name == 'lesson' else 0 if name == 'original_centered' else 1 if name == 'varied_approach' else None,
            'max_steps': 150 if single else 600,
            'success': 'passed' if single else 'timeout and passes >= 2',
            'deterministic': True,
            'geometry': {key: asdict(getattr(env, key))
                         for key in ('cage', 'drone', 'hoop', 'camera', 'pad')},
        }
        records = []
        try:
            for i in range(episodes):
                obs, _ = env.reset(seed=VALIDATION_SEED_START+i)
                while True:
                    obs, _, term, trunc, info = env.step(model.predict(obs, deterministic=True)[0])
                    if term or trunc:
                        break
                records.append({'seed': VALIDATION_SEED_START+i, 'outcome': info['outcome'],
                                'passes': info['passes'], 'steps': info['steps']})
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
            'outcomes': dict(Counter(r['outcome'] for r in records)), 'records': records}
    return report
