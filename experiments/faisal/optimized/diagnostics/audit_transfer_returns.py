from pathlib import Path
import sys,json,numpy as np,torch
R=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928');sys.path.insert(0,str(R/'source_snapshot'))
OUT=Path(__file__).parent
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
torch.set_num_threads(1)
allrows=[];summary=[]
fit_mode='--fit' in sys.argv
seeds=[2820026,2820664,2820750,2820805,2820828,2821928,2821958,2822002,2822042] if fit_mode else list(range(5000000,5000025))
for checkpoint in (300,500):
 m=RecurrentPPO.load(R/f'model_update_{checkpoint}.zip',device='cuda')
 for seed in seeds:
  e=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True);o,info=e.reset(seed=seed);p=RecurrentPredictor(m);trace=[]
  while True:
   with torch.inference_mode():a,_=p.predict(o)
   o,r,t,tr,info=e.step(a);b=e.unwrapped;rel=b.pos-b.hoop_centre
   trace.append(dict(step=info['steps'],action=a.tolist(),reward=r,passes=info['passes'],phase=info['lap_phase'],visible=float(b.observe()[6]),normal=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height=float(rel[2]),clearance=float(b._clearance_reach()),outcome=info['outcome']))
   if t or tr:break
  allrows.append(dict(checkpoint=checkpoint,seed=seed,trace=trace));e.close()
  front=[x for x in trace if x['passes'] and x['phase'] in ('align_front','approach')]
  if front:
   first=front[0];after=[x for x in trace if x['step']>=first['step']]
   summary.append(dict(checkpoint=checkpoint,seed=seed,first_front=first,end=trace[-1],remaining_steps=len(after),visible_fraction=float(np.mean([x['visible'] for x in after])),min_abs_lateral=min(abs(x['lateral']) for x in after),max_normal=max(x['normal'] for x in after)))
(OUT/('fit_trajectories.json' if fit_mode else 'trajectories.json')).write_text(json.dumps(allrows));(OUT/('fit_summary.json' if fit_mode else 'summary.json')).write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
