"""FIT-only teacher/student reacquisition audit. Truth is diagnostic output only."""
from pathlib import Path
import sys,json
import numpy as np
import torch

ROOT=Path(__file__).parent
RUN=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
BANK=Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3')
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

def features(o):
    return dict(camera=o[64:72].tolist(),last_any_age=float(o[93]),front_seen=float(o[108]),
                front_age=float(o[109]),estimated_ring_body=o[115:119].tolist())

def main():
    torch.set_num_threads(1)
    model=RecurrentPPO.load(RUN/'model_update_500.zip',device='cuda')
    seeds=json.loads((RUN/'protocol.json').read_text())['fit_lap_seeds']
    records={r['seed']:r for r in json.loads((BANK/'records.json').read_text()) if r['seed'] in seeds}
    rows=[]
    for seed in seeds:
        assert seed%4!=3
        env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
        initial,_=env.reset(seed=seed);b=env.unwrapped
        with np.load(BANK/'trajectories'/f'{seed}.npz') as data:
            obs=data['observations'];pos=data['positions'];actions=data['actions']
            assert np.array_equal(initial,obs[0])
            first=next(e['step'] for e in records[seed]['events'] if e['event']=='full_pass_credit')
            seen=next(i for i in range(first,len(obs)) if obs[i,70]>.5)
            forced=RecurrentPredictor(model)
            with torch.inference_mode():
                for camera_obs in obs[:seen+1]:forced_action,_=forced.predict(camera_obs)
            rel=pos[seen]-b.hoop_centre
            rows.append(dict(kind='teacher',seed=seed,first_pass=first,seen_step=seen,
                normal=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height=float(rel[2]),
                inputs=features(obs[seen]),next_action=actions[seen].tolist(),student_on_teacher_history=forced_action.tolist()))
        predictor=RecurrentPredictor(model);o=initial;first=None;captured=False;row=None
        while True:
            with torch.inference_mode():a,_=predictor.predict(o)
            # Current action uses current observation, before advancing the simulator.
            if first is not None and not captured and o[70]>.5:
                rel=b.pos-b.hoop_centre
                row=dict(kind='student500',seed=seed,first_pass=first,seen_step=b.steps,
                    normal=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height=float(rel[2]),
                    inputs=features(o),next_action=a.tolist())
                captured=True
            o,_,t,tr,info=env.step(a)
            if first is None and info['passes']>0:first=b.steps
            if t or tr:break
        if row is None:row=dict(kind='student500',seed=seed,first_pass=first,seen_step=None)
        row.update(outcome=info['outcome'],passes=info['passes']);rows.append(row);env.close()
    report=dict(fit_only=True,model='baseline500',rows=rows)
    (ROOT/'reacquisition_inputs_audit.json').write_text(json.dumps(report,indent=2))
    for kind in ('teacher','student500'):
        subset=[r for r in rows if r['kind']==kind and r['seen_step'] is not None]
        print(kind,json.dumps(dict(count=len(subset),current_marker=[r['inputs']['camera'][7] for r in subset],
            front_memory_seen=[r['inputs']['front_seen'] for r in subset],
            mean_height=float(np.mean([r['height'] for r in subset])),
            mean_next_action=np.mean([r['next_action'] for r in subset],axis=0).tolist())),flush=True)

if __name__=='__main__':main()
