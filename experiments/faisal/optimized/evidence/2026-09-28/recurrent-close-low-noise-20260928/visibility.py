from pathlib import Path
import sys,json,numpy as np
R=Path('D:/DroneSoccerData/runs/recurrent-close-low-noise-20260928');sys.path.insert(0,str(R/'source_snapshot'))
from src.guided_curriculum import GuidedLessonEnv
rows=[]
for stage in range(4):
 for i in range(100):
  e=GuidedLessonEnv(stage=stage);o,info=e.reset(seed=9500000+i);v=e.observe()
  rows.append(dict(stage=stage,seed=9500000+i,visible=float(v[6]),marker=float(v[7]),frame=v.tolist()));e.close()
for stage in range(4):print(stage,sum(x['visible'] for x in rows if x['stage']==stage))
(R/'reset_visibility.json').write_text(json.dumps(rows))
