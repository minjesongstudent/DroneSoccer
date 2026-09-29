"""Inspect the rejected capacity policy's two-pass validation trajectory."""
from pathlib import Path
import sys,json
import torch

ROOT=Path(__file__).parent
RUN=ROOT/'lap-recurrent-anchor10-20260928'
sys.path.insert(0,str(RUN/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

torch.set_num_threads(1)
models={str(seed):RUN/'model_50.zip' for seed in [2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042,5000014]}
report={}
for label,path in models.items():
    m=RecurrentPPO.load(path,device='cuda');p=RecurrentPredictor(m)
    env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
    o,info=env.reset(seed=int(label))
    rows=[];events=[];previous=(0,'approach')
    while True:
        with torch.inference_mode():a,_=p.predict(o)
        o,r,t,tr,info=env.step(a)
        b=env.unwrapped;rel=b.pos-b.hoop_centre
        row=dict(step=info['steps'],passes=info['passes'],phase=info['lap_phase'],outcome=info['outcome'],
            normal=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height=float(rel[2]),
            action=a.tolist(),visible=float(o[70]),reward=float(r))
        rows.append(row)
        state=(row['passes'],row['phase'])
        if state!=previous or t or tr:events.append(row)
        previous=state
        if t or tr:break
    env.close()
    report[label]={'seed':int(label),'events':events,'trace':rows}
    print(label,json.dumps(events),flush=True)
(ROOT/'recurrent50_fit_audit.json').write_text(json.dumps(report,indent=2))
