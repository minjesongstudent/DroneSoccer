"""Diagnostic only: compare current memory with a persistent front-sighting trace.

The alternative is never installed into training. Substituting its 15 values
at inference is an out-of-distribution ablation, not a trained improvement.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run',default='return-route-40k-20260927')
    parser.add_argument('--checkpoint',type=int,default=20000)
    parser.add_argument('--seeds',nargs='+',type=int,default=[5000000,5000001,5000002,5000003,5000005,5000007])
    args=parser.parse_args()
    run=ROOT/'runs'/args.run
    # Keep diagnostics reproducible if subsequent experiments modify live src.
    source=run/'source_snapshot'
    if not (source/'src'/'sim.py').exists():
        raise FileNotFoundError('The v3 run must retain source_snapshot/src.')
    sys.path.insert(0,str(source))
    from src.anchored_policy import AnchoredSAC,protected_observations
    from src.curriculum import REWARD_VERSION
    from src.guided_curriculum import GuidedLessonEnv
    from src.observation_memory import CameraCommandMemory
    import src.sim
    if REWARD_VERSION!='lap-remaining-route-potential-v3' or Path(src.sim.__file__).resolve().parent.parent!=source.resolve():
        raise RuntimeError('Expected this run\'s explicit v3 source snapshot.')

    class FrontSightingMemory(CameraCommandMemory):
        def _detect(self,base):
            # The sole changed condition uses the public current marker bit.
            if base[70]>.5 and base[71]>.5:
                super()._detect(base)

    torch.set_num_threads(1)
    paths={'initial':run/'initial_model.zip',str(args.checkpoint):run/'checkpoints'/f'model_{args.checkpoint}_steps.zip'}
    models={name:AnchoredSAC.load(path,device='cpu') for name,path in paths.items()}
    report={'source':str(source.relative_to(ROOT)),'reward_version':REWARD_VERSION,'device':'cpu','seeds':args.seeds,
            'checkpoint_hashes':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
            'source_hashes':{name:hashlib.sha256((source/'src'/name).read_bytes()).hexdigest()
                             for name in ('sim.py','observation_memory.py','mission_observation.py','anchored_policy.py')},
            'cases':{}}
    output=ROOT/'diagnostics'/'front-memory-diagnosis';output.mkdir(exist_ok=True)
    frozen_actual=[];frozen_front=[]
    for model_name,mode in [('initial','actual'),(str(args.checkpoint),'actual'),(str(args.checkpoint),'replace_with_front_memory')]:
        model=models[model_name];case=f'{model_name}-{mode}';summaries=[]
        for seed in args.seeds:
            env=GuidedLessonEnv(stage=4);base,info=env.reset(seed=seed)
            latest=CameraCommandMemory();front=FrontSightingMemory()
            actual=np.append(latest.reset(base),0.).astype(np.float32)
            alternative=np.append(front.reset(base),0.).astype(np.float32)
            rows=[];initial_counter=model.num_timesteps
            for t in range(600):
                inp=actual if mode=='actual' else alternative
                before_phase=env._lap_phase
                if model_name==str(args.checkpoint) and mode=='actual':
                    frozen_actual.append(actual.copy());frozen_front.append(alternative.copy())
                action=model.predict(inp,deterministic=True)[0]
                base,reward,term,trunc,info=env.step(action)
                actual=np.append(latest.advance(action,base),0.).astype(np.float32)
                alternative=np.append(front.advance(action,base),0.).astype(np.float32)
                rows.append({'step':t+1,'phase_before':before_phase,'phase_after':env._lap_phase,
                             'outcome':info['outcome'],'passes':info['passes'],'position':env.pos.tolist(),
                             'heading':float(env.heading),'speed':float(np.linalg.norm(env.vel)),
                             'target_distance':float(np.linalg.norm(env.pos-env._progress_target(env.pos))),
                             'action':action.tolist(),'reward':float(reward),
                             'current_frame':actual[64:72].tolist(),'latest_memory':actual[84:99].tolist(),
                             'front_memory':alternative[84:99].tolist(),
                             'protected_input':bool(protected_observations(torch.as_tensor(inp[None]))[0])})
                if term or trunc:break
            assert model.num_timesteps==initial_counter
            phases=defaultdict(list)
            for row in rows:phases[row['phase_before']].append(row)
            def summarize(rr):
                back=[r for r in rr if r['current_frame'][6]>.5 and r['current_frame'][7]<.5]
                return {'steps':len(rr),'min_target_distance':min(r['target_distance'] for r in rr),
                        'mean_speed':float(np.mean([r['speed'] for r in rr])),
                        'mean_action':np.mean([r['action'] for r in rr],axis=0).tolist(),
                        'back_visible_steps':len(back),'protected_steps':sum(r['protected_input'] for r in rr),
                        'back_latest_age_mean':float(np.mean([r['latest_memory'][9]*60 for r in back])) if back else None,
                        'back_front_age_mean':float(np.mean([r['front_memory'][9]*60 for r in back])) if back else None,
                        'back_front_displacement_mean':np.mean([np.asarray(r['front_memory'][10:13])*3 for r in back],axis=0).tolist() if back else None}
            summary={'seed':seed,'passes':info['passes'],'outcome':info['outcome'],'steps':len(rows),
                     'phases':{phase:summarize(rr) for phase,rr in phases.items()},
                     'first_pass_step':next((r['step'] for r in rows if r['passes']),None),
                     'side_return_step':next((r['step'] for r in rows if r['phase_after']=='return_front'),None),
                     'front_return_step':next((r['step'] for r in rows if r['phase_after']=='align_front'),None),
                     'final_latest_memory':rows[-1]['latest_memory'],'final_front_memory':rows[-1]['front_memory']}
            summaries.append(summary)
            (output/f'{case}-{seed}.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            env.close()
        report['cases'][case]={'episodes':summaries,'outcomes':dict(Counter(x['outcome'] for x in summaries)),
                               'passes':sum(x['passes'] for x in summaries),
                               'side_returns':sum(x['side_return_step'] is not None for x in summaries),
                               'front_returns':sum(x['front_return_step'] is not None for x in summaries),
                               'two_pass_episodes':sum(x['passes']>=2 for x in summaries)}
        print(case,{k:v for k,v in report['cases'][case].items() if k!='episodes'},flush=True)
    x=np.array(frozen_actual);alt=np.array(frozen_front)
    a=models[str(args.checkpoint)].predict(x,deterministic=True)[0]
    b=models[str(args.checkpoint)].predict(alt,deterministic=True)[0]
    back=(x[:,70]>.5)&(x[:,71]<.5)
    report['same_state_input_substitution']={'rows':len(x),'visible_back_rows':int(back.sum()),
        'action_mae_all':np.abs(a-b).mean(axis=0).tolist(),
        'action_mae_visible_back':np.abs(a[back]-b[back]).mean(axis=0).tolist(),
        'latest_age_seconds_mean_visible_back':float(x[back,93].mean()*60),
        'front_age_seconds_mean_visible_back':float(alt[back,93].mean()*60)}
    for name,path in paths.items():assert hashlib.sha256(path.read_bytes()).hexdigest()==report['checkpoint_hashes'][name]
    (output/'summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print('substitution',report['same_state_input_substitution'],flush=True)


if __name__=='__main__':main()
