from pathlib import Path
import sys,json
import numpy as np,torch
R=Path('D:/DroneSoccerData/runs/recurrent-close-low-noise-20260928');sys.path.insert(0,str(R/'source_snapshot'))
from src.standard_source_policy import StandardSourceSAC as SAC
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
torch.set_num_threads(1);rows=[]
for name in ('recurrent','preserved_sac'):
 m=RecurrentPredictor(RecurrentPPO.load(R/'model_65536.zip',device='cuda')) if name=='recurrent' else SAC.load('D:/DroneSoccerData/runs/standard-source-20260928/initial_model.zip',device='cuda')
 for i in range(25):
  e=wrap(GuidedLessonEnv(stage=0),single=True,metric_ring=True,geometric_return=True);o,info=e.reset(seed=9400000+i)
  if name=='recurrent':m.reset_episode()
  trace=[]
  while True:
   with torch.inference_mode():a,_=m.predict(o,deterministic=True)
   o,r,t,tr,info=e.step(a);b=e.unwrapped;rel=b.pos-b.hoop_centre
   trace.append(dict(step=info['steps'],pos=b.pos.tolist(),action=a.tolist(),reward=r,normal_distance=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height_error=float(rel[2]),outcome=info['outcome']))
   if t or tr:break
  rows.append(dict(model=name,seed=9400000+i,outcome=info['outcome'],passes=info['passes'],trace=trace));e.close()
 print(name,{out:sum(x['model']==name and x['outcome']==out for x in rows) for out in ('passed','hit','timeout','left_cage')},flush=True)
(R/'comparison_trajectories.json').write_text(json.dumps(rows))

