"""Read-only CPU diagnosis of the completed protected-policy trial."""
from pathlib import Path
import json
import sys
import hashlib
from collections import Counter, defaultdict

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/completion-protected-20260927-140307'
SNAPSHOT = RUN/'source_snapshot'
if not (SNAPSHOT/'src'/'sim.py').exists():
    raise RuntimeError('The protected run needs its frozen source_snapshot for reproducible diagnosis.')
sys.path.insert(0, str(SNAPSHOT))
from src import sim, curriculum
from src.anchored_policy import AnchoredSAC, gaussian_kl, protected_observations
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation

OUT = ROOT/'diagnostics/protected-run-audit.json'


def audit():
    torch.set_num_threads(1)
    assert Path(sim.__file__).resolve().parent == (SNAPSHOT/'src').resolve()
    assert curriculum.REWARD_VERSION == 'lap-clearance-potential-v2'
    report = {'run': RUN.name, 'seed_start': 5_000_000, 'episodes': 25, 'models': {},
              'source_snapshot': str(SNAPSHOT), 'reward_version': curriculum.REWARD_VERSION,
              'source_sha256': {name: hashlib.sha256((SNAPSHOT/'src'/name).read_bytes()).hexdigest()
                                for name in ('sim.py', 'anchored_policy.py', 'mission_observation.py',
                                             'observation_memory.py')}}
    with torch.inference_mode():
        for step in (0, 20000, 60000, 100000):
            path = RUN/'initial_model.zip' if step == 0 else RUN/f'checkpoints/model_{step}_steps.zip'
            model = AnchoredSAC.load(path, device='cpu')
            phases = defaultdict(lambda: {'actions': [], 'speeds': [], 'protected': [],
                                          'kl': [], 'reward': [], 'target_distance': []})
            episodes, transitions = [], Counter()
            for seed in range(5_000_000, 5_000_025):
                env = MissionObservation(GuidedLessonEnv(stage=4), single_pass=False)
                obs, _ = env.reset(seed=seed)
                returns, discounted = 0., 0.
                for t in range(600):
                    base = env.unwrapped
                    phase = base._lap_phase
                    x = torch.as_tensor(obs).reshape(1, -1)
                    action = model.predict(obs, deterministic=True)[0]
                    mean, std, _ = model.actor.get_action_dist_params(x)
                    rmean, rstd, _ = model.reference_actor.get_action_dist_params(x)
                    bucket = phases[phase]
                    bucket['actions'].append(action)
                    bucket['speeds'].append(float(np.linalg.norm(base.vel)))
                    bucket['protected'].append(bool(protected_observations(x).item()))
                    bucket['kl'].append(float(gaussian_kl(mean, std, rmean, rstd).item()))
                    bucket['target_distance'].append(float(np.linalg.norm(base.pos-base._progress_target(base.pos))))
                    obs, reward, term, trunc, info = env.step(action)
                    bucket['reward'].append(reward)
                    returns += reward
                    discounted += model.gamma**t*reward
                    if phase != base._lap_phase:
                        transitions[phase+' -> '+base._lap_phase] += 1
                    if term or trunc:
                        break
                rel = base.pos-base.hoop_centre
                episodes.append({'seed': seed, 'outcome': info['outcome'], 'passes': info['passes'],
                                 'steps': t+1, 'phase': base._lap_phase,
                                 'return': returns, 'discounted_return': discounted,
                                 'target_distance': float(np.linalg.norm(base.pos-base._progress_target(base.pos))),
                                 'axial': float(rel@base.hoop_normal), 'side': float(rel@base._axis_h),
                                 'height_error': float(rel[2])})
            summary = {'actor_updates': model.actor_updates, 'actor_attempts': model.actor_attempts,
                       'rejected': model.anchor_rejected,
                       'outcomes': dict(Counter(r['outcome'] for r in episodes)),
                       'final_phase': dict(Counter(r['phase'] for r in episodes)),
                       'pass_counts': dict(Counter(r['passes'] for r in episodes)),
                       'mean_return': float(np.mean([r['return'] for r in episodes])),
                       'transitions': dict(transitions), 'phases': {}, 'episodes': episodes}
            for phase, values in phases.items():
                summary['phases'][phase] = {'steps': len(values['actions']),
                    **{key+'_mean': np.mean(value, axis=0).tolist() for key, value in values.items()}}
            report['models'][str(step)] = summary
            OUT.write_text(json.dumps(report, indent=2), encoding='utf-8')
            print(json.dumps({'step': step, **{k:v for k,v in summary.items() if k != 'episodes'}}), flush=True)


if __name__ == '__main__':
    audit()
