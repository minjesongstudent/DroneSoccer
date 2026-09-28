from pathlib import Path
import sys,json,numpy as np,torch
R=Path('D:/DroneSoccerData/runs/recurrent-lap-finetune-20260928');sys.path.insert(0,str(R/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
torch.set_num_threads(1);m=RecurrentPPO.load(R/'model_32768.zip',device='cuda');rows=[]
for i in range(25):
 e=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True);o,info=e.reset(seed=5000000+i);p=RecurrentPredictor(m);trace=[]
 while True:
  with torch.inference_mode():a,_=p.predict(o)
  o,r,t,tr,info=e.step(a);b=e.unwrapped;rel=b.pos-b.hoop_centre
  trace.append(dict(step=info['steps'],position=b.pos.tolist(),heading=float(b.heading),action=a.tolist(),reward=r,passes=info['passes'],phase=info['lap_phase'],visible=float(b.observe()[6]),normal=float(rel@b.hoop_normal),lateral=float(rel@b._axis_h),height=float(rel[2]),outcome=info['outcome']))
  if t or tr:break
 rows.append(dict(seed=5000000+i,outcome=info['outcome'],passes=info['passes'],trace=trace));e.close()
(R/'postpass_trajectories.json').write_text(json.dumps(rows))
summary=[]
for row in rows:
 post=[x for x in row['trace'] if x['passes']>0]
 if post:
  summary.append(dict(seed=row['seed'],outcome=row['outcome'],pass_step=post[0]['step'],post_steps=len(post),phases=list(dict.fromkeys(x['phase'] for x in post)),mean_action=np.mean([x['action'] for x in post],axis=0).tolist(),visible_fraction=np.mean([x['visible'] for x in post]),end_position=post[-1]['position'],end_normal=post[-1]['normal'],end_lateral=post[-1]['lateral'],reward=sum(x['reward'] for x in post)))
(R/'postpass_summary.json').write_text(json.dumps(summary,indent=2))
print(json.dumps(summary,indent=2))
