"""Paired learned trajectories; truth used exclusively for diagnostic errors."""
from pathlib import Path
import sys,json
import numpy as np
import torch
R=Path(__file__).parent
RUN=R/'history-guard-rl-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.history_guard_policy import HistoryGuardSAC
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

torch.set_num_threads(1)
a=HistoryGuardSAC.load(RUN/'model_0.zip',device='cuda')
b=HistoryGuardSAC.load(RUN/'model_10000.zip',device='cuda')
base={k:v.clone() for k,v in a.actor.state_dict().items()}
b.actor.load_state_dict({k:v+.01*(b.actor.state_dict()[k]-v) for k,v in base.items()})
seeds=json.loads((RUN/'protocol.json').read_text())['fit_seeds']
OUT=RUN/'first_divergence';OUT.mkdir(exist_ok=False)
groups={'camera':(0,72),'commands':(72,84),'any_memory':(84,99),'task':(99,100),'front_memory':(100,115),'estimate':(115,119)}
report=[]
with torch.inference_mode():
    for seed in seeds:
        flights=[]
        for name,model in [('original',a),('one_percent',b)]:
            env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
            obs,_=env.reset(seed=seed)
            rows=[]
            while True:
                s=env.unwrapped;delta=s.hoop_centre-s.pos
                c,z=np.cos(s.heading),np.sin(s.heading)
                truth=np.array([c*delta[0]+z*delta[1],-z*delta[0]+c*delta[1],delta[2]])
                action=model.predict(obs,deterministic=True)[0]
                rows.append(dict(step=s.steps,observation=obs.tolist(),position=s.pos.tolist(),heading=float(s.heading),
                    estimate_error=float(np.linalg.norm(obs[115:118]*3-truth)) if obs[118]>.5 else None,
                    action=action.tolist(),passes=s.passes))
                obs,_,t,tr,info=env.step(action)
                if t or tr:break
            env.close();flights.append(rows)
            (OUT/f'{name}_{seed}.json').write_text(json.dumps(dict(trace=rows,outcome=info['outcome'],passes=info['passes']),indent=2))
        left,right=flights
        def first(predicate):
            for x,y in zip(left,right):
                if predicate(x,y):
                    d=np.abs(np.array(x['observation'])-y['observation'])
                    return dict(step=x['step'],position_gap=float(np.linalg.norm(np.array(x['position'])-y['position'])),
                        group_max={k:float(d[lo:hi].max()) for k,(lo,hi) in groups.items()},
                        original_estimate_error=x['estimate_error'],changed_estimate_error=y['estimate_error'],
                        camera_original=x['observation'][64:72],camera_changed=y['observation'][64:72])
            return None
        report.append(dict(seed=seed,first_centimetre=first(lambda x,y:np.linalg.norm(np.array(x['position'])-y['position'])>.01),
            first_large_observation=first(lambda x,y:np.max(np.abs(np.array(x['observation'])-y['observation']))>.1),
            first_visibility_mismatch=first(lambda x,y:x['observation'][70]!=y['observation'][70]),
            original_max_estimate_error=max(x['estimate_error'] for x in left if x['estimate_error'] is not None)))
(OUT/'summary.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
