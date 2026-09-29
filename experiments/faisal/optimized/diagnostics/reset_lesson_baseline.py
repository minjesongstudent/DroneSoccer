"""Training-seed prerequisite probe; lessons are not full-flight validation."""
import json,hashlib
from pathlib import Path
import torch
from src.front_policy import FrontMemorySAC
from src.front_memory import FrontMemoryObservation
from src.mission_observation import MissionObservation
from src.return_reset_lessons import ReturnResetLesson,LEVELS
from src.return_evaluation import _fingerprint

torch.set_num_threads(1)
p=Path('D:/DroneSoccerData/runs/front-memory-20260927/initial_model.zip')
m=FrontMemorySAC.load(p,device='cuda');before=_fingerprint(m);records=[]
for level in range(len(LEVELS)):
 env=FrontMemoryObservation(MissionObservation(ReturnResetLesson(level),single_pass=False))
 for seed in range(2100000,2100010):
  obs,info=env.reset(seed=seed)
  initial_visible=bool(obs[70]>.5);initial_front=bool(obs[71]>.5)
  while True:
   obs,r,term,trunc,info=env.step(m.predict(obs,deterministic=True)[0])
   if term or trunc:break
  records.append(dict(level=level,seed=seed,outcome=info['outcome'],passes=info['passes'],success=info['reset_lesson_success'],initial_visible=initial_visible,initial_front=initial_front))
 env.close()
assert _fingerprint(m)==before
summary=[dict(level=k,name=LEVELS[k],passes=sum(r['passes']>0 for r in records if r['level']==k),success=sum(r['success'] for r in records if r['level']==k),visible_starts=sum(r['initial_visible'] for r in records if r['level']==k)) for k in range(len(LEVELS))]
out=Path('D:/DroneSoccerData/diagnostics/reset-lesson-baseline-v2.json');assert not out.exists()
out.write_text(json.dumps(dict(summary=summary,records=records,verified_no_learning=True,full_mission_test=False,source_sha256=hashlib.sha256(p.read_bytes()).hexdigest()),indent=2))
print(json.dumps(summary),flush=True)
