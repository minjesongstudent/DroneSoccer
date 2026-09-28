from pathlib import Path
import sys,json,numpy as np,torch
R=Path('D:/DroneSoccerData/runs/recurrent-lap-finetune-20260928');sys.path.insert(0,str(R/'source_snapshot'))
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
bank=Path('D:/DroneSoccerData/diagnostics/learned-success-collection-v3');seeds=json.loads((bank/'result.json').read_text())['successes_by_split']['fit'];rows=[]
for seed in seeds:
 data=np.load(bank/'trajectories'/f'{seed}.npz');e=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True);o,info=e.reset(seed=seed);error=0.
 for i,a in enumerate(data['actions']):
  error=max(error,float(np.max(abs(o-data['observations'][i]))));o,_,t,tr,info=e.step(a)
  if t or tr:break
 error=max(error,float(np.max(abs(o-data['observations'][i+1]))))
 assert error<1e-6 and info['steps']==600 and info['passes']>=2 and info['outcome']=='timeout'
 rows.append(dict(seed=seed,max_observation_error=error,steps=info['steps'],passes=info['passes'],outcome=info['outcome']));e.close()
(R/'complete_fit_history_verification.json').write_text(json.dumps(rows,indent=2));print(rows)
