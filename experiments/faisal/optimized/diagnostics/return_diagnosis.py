r"""Read-only deterministic rollout/waypoint audit; no training or teacher data.

Run from the project root:
  ..\DroneSoccer\.venv\Scripts\python.exe diagnostics/return_diagnosis.py
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def rollout(model, seed):
    from src.anchored_policy import protected_observations
    from src.guided_curriculum import GuidedLessonEnv
    from src.mission_observation import MissionObservation
    from src.sim import PROGRESS_UNIT, R_PROGRESS, R_HIDDEN, R_PASS, R_CRASH, CRASHES
    env = MissionObservation(GuidedLessonEnv(stage=4), single_pass=False)
    obs, info = env.reset(seed=seed)
    base = env.unwrapped
    rows = []
    for _ in range(600):
        before = base.pos.copy()
        target = base._progress_target(before).copy()
        phase = base._lap_phase
        input_copy = obs.copy()
        action = model.predict(obs, deterministic=True)[0]
        obs, reward, term, trunc, info = env.step(action)
        next_target = base._progress_target(base.pos).copy()
        progress = R_PROGRESS*(np.linalg.norm(before-target)-np.linalg.norm(base.pos-next_target))/PROGRESS_UNIT
        delta = base.pos-base.hoop_centre
        rows.append({
            'step': info['steps'], 'phase_before': phase, 'phase_after': base._lap_phase,
            'position': base.pos.tolist(), 'ring_relative_position': [float(delta@base.hoop_normal), float(delta@base._axis_h), float(delta[2])],
            'heading': float(base.heading), 'speed': float(np.linalg.norm(base.vel)),
            'target_before': target.tolist(), 'target_after': next_target.tolist(),
            'distance_to_old_target': float(np.linalg.norm(base.pos-target)),
            'distance_to_next_target': float(np.linalg.norm(base.pos-next_target)),
            'action': action.tolist(), 'input': input_copy.tolist(),
            'protected_input': bool(protected_observations(torch.as_tensor(input_copy[None]))[0]),
            'visible': bool(obs[70]), 'has_seen': bool(obs[92]), 'sighting_age_seconds': float(obs[93]*60),
            'memory_displacement_m': (obs[94:97]*3).tolist(),
            'reward': float(reward), 'progress_reward': float(progress),
            'visibility_reward': -R_HIDDEN*(1-float(obs[70])),
            'pass_reward': R_PASS*float(info['outcome']=='passed'),
            'crash_reward': -R_CRASH*float(info['outcome'] in CRASHES),
            'passes': info['passes'], 'outcome': info['outcome'],
        })
        if term or trunc:
            break
    env.close()
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row['phase_before']].append(row)
    first_pass = next((r['step'] for r in rows if r['passes']), None)
    summary = {'seed': seed, 'outcome': info['outcome'], 'passes': info['passes'], 'steps': len(rows),
               'first_pass_step': first_pass, 'total_reward': sum(r['reward'] for r in rows),
               'phase_changes': [{k:r[k] for k in ('step','phase_before','phase_after','reward','progress_reward','distance_to_old_target','distance_to_next_target','passes')}
                                for r in rows if r['phase_before'] != r['phase_after']],
               'phases': {p:{'steps': len(rr), 'closest_target_distance': min(r['distance_to_old_target'] for r in rr),
                             'visible_fraction': float(np.mean([r['visible'] for r in rr])),
                             'protected_fraction': float(np.mean([r['protected_input'] for r in rr])),
                             'mean_speed': float(np.mean([r['speed'] for r in rr])),
                             'mean_action': np.mean([r['action'] for r in rr], axis=0).tolist(),
                             'reward_sum': sum(r['reward'] for r in rr)} for p,rr in by_phase.items()},
               'last_100': {'mean_speed': float(np.mean([r['speed'] for r in rows[-100:]])),
                            'mean_target_distance': float(np.mean([r['distance_to_next_target'] for r in rows[-100:]])),
                            'mean_abs_action': np.abs([r['action'] for r in rows[-100:]]).mean(axis=0).tolist(),
                            'visible_fraction': float(np.mean([r['visible'] for r in rows[-100:]])),
                            'protected_fraction': float(np.mean([r['protected_input'] for r in rows[-100:]]))}}
    return summary, rows


def waypoint_reward_probe():
    """One physical step just outside/inside each 5cm target threshold.

    These are constructed diagnostic states, never training transitions.
    Compare a stationary command with a small move that reaches the waypoint.
    """
    from src.sim import HoopEnv
    records=[]
    for phase in ('return_side','return_front','align_front'):
        for command_name, action in (('hover', [0,0,0,0]), ('reach_waypoint', [0,0,-.1,0])):
            env=HoopEnv()
            env.place([1.6,1.0,1.],0.,[1.025,1.025,1.],[1,0,0])
            env._lap_phase=phase
            target=env._progress_target(env.pos).copy()
            env.pos=target+np.array([0.,0.,.051])
            # No velocity/heading modification; place() already starts at rest.
            _,r,term,trunc,info=env.step(action)
            records.append({'phase':phase,'command':command_name,'reward':r,'outcome':info['outcome'],
                            'next_phase':env._lap_phase,'old_distance':.051,
                            'new_distance_to_old_target':float(np.linalg.norm(env.pos-target)),
                            'new_distance_to_next_target':float(np.linalg.norm(env.pos-env._progress_target(env.pos))),
                            'terminated':term})
            env.close()
    return records


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run', default='completion-protected-20260927-140307')
    parser.add_argument('--device', default='cpu', choices=('cpu','cuda'))
    parser.add_argument('--seeds', nargs='+', type=int, default=[5000000,5000001,5000002,5000003,5000005,5000007])
    args=parser.parse_args()
    torch.set_num_threads(1)
    folder=ROOT/'runs'/args.run
    snapshot=folder/'source_snapshot'
    if not (snapshot/'src'/'sim.py').exists():
        raise FileNotFoundError('This audit requires the run\'s archived source_snapshot; never substitute changed live source.')
    sys.path.insert(0,str(snapshot))
    from src.anchored_policy import AnchoredSAC
    from src.curriculum import REWARD_VERSION
    import src.sim
    if Path(src.sim.__file__).resolve().parent.parent != snapshot.resolve():
        raise RuntimeError('A different src package is already imported. Run this audit in a fresh Python process.')
    if REWARD_VERSION != 'lap-clearance-potential-v2':
        raise ValueError('This diagnostic reproduces the pre-route-fix reward only.')
    paths={'initial':folder/'initial_model.zip', '100k':folder/'checkpoints'/'model_100000_steps.zip'}
    output=ROOT/'diagnostics'/'return-diagnosis'
    output.mkdir(exist_ok=True)
    report={'run':args.run,'device':args.device,'deterministic':True,'seeds':args.seeds,
            'source_snapshot':str(snapshot.relative_to(ROOT)), 'reward_version':REWARD_VERSION,
            'source_sha256':{name:hashlib.sha256((snapshot/'src'/name).read_bytes()).hexdigest()
                             for name in ('sim.py','curriculum.py','guided_curriculum.py','observation_memory.py','mission_observation.py','anchored_policy.py')},
            'task':'GuidedLessonEnv(stage=4), MissionObservation(single_pass=False)',
            'checkpoints':{},'episodes':{},'waypoint_reward_probe':waypoint_reward_probe()}
    for label,path in paths.items():
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        model=AnchoredSAC.load(path,device=args.device)
        before_steps=model.num_timesteps
        report['checkpoints'][label]={'path':str(path.relative_to(ROOT)),'sha256':digest,'steps':before_steps,
                                     'actor_updates':model.actor_updates,'actor_attempts':model.actor_attempts,
                                     'anchor_rejected':model.anchor_rejected}
        report['episodes'][label]=[]
        for seed in args.seeds:
            with torch.inference_mode():
                summary,rows=rollout(model,seed)
            report['episodes'][label].append(summary)
            (output/f'{label}-{seed}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(label,seed,summary['passes'],summary['outcome'],summary['phases'],flush=True)
        assert model.num_timesteps==before_steps
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('waypoint_reward_probe',json.dumps(report['waypoint_reward_probe']),flush=True)
    print('report',output/'summary.json',flush=True)


if __name__=='__main__':
    main()
