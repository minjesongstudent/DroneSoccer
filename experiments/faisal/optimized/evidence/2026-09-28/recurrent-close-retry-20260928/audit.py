import sys,json
from pathlib import Path
from collections import Counter
import numpy as np, torch
R=Path('D:/DroneSoccerData/runs/recurrent-close-retry-20260928')
sys.path.insert(0,str(R/'source_snapshot'))
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import RecurrentPredictor
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

torch.set_num_threads(1)
rows=[]
for checkpoint in (0,65536):
 m=RecurrentPPO.load(R/f'model_{checkpoint}.zip',device='cuda')
 for stage in (0,1,2):
  for deterministic in (True,False):
   torch.manual_seed(1234)
   for i in range(10):
    env=wrap(GuidedLessonEnv(stage=stage),single=True,metric_ring=True,geometric_return=True)
    obs,info=env.reset(seed=9200000+i);p=RecurrentPredictor(m)
    actions=[];observations=[];rewards=[];start=env.unwrapped.pos.copy()
    while True:
     observations.append(obs.copy())
     with torch.inference_mode():a,_=p.predict(obs,deterministic=deterministic)
     actions.append(a.copy());obs,r,t,tr,info=env.step(a);rewards.append(r)
     if t or tr:break
    a=np.asarray(actions);o=np.asarray(observations)
    rows.append(dict(checkpoint=checkpoint,stage=stage,deterministic=deterministic,seed=9200000+i,
      outcome=info['outcome'],passes=info['passes'],steps=len(a),reward=float(sum(rewards)),
      mean_action=a.mean(0).tolist(),saturation=float((abs(a)>=.99).mean()),
      observation_absmax=float(abs(o).max()),observation_finite=bool(np.isfinite(o).all()),
      displacement=(env.unwrapped.pos-start).tolist()))
    env.close()
   subset=rows[-10:]
   print(checkpoint,stage,deterministic,dict(Counter(x['outcome'] for x in subset)),flush=True)
(R/'diagnostic_close_flights.json').write_text(json.dumps(rows,indent=2))
