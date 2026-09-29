"""Fixed development baseline before any 115D policy updates; no fresh acceptance seeds."""
import json
from pathlib import Path
import torch
from src.front_policy import start_from_checkpoint
from src.front_memory import FrontMemoryObservation
from src.mission_observation import MissionObservation
from src.guided_curriculum import GuidedLessonEnv
from src.front_return_library import FrontReturnLibrary
from src.unified_evaluation import evaluate
from src.return_evaluation import evaluate_return_practice

torch.set_num_threads(1)
out=Path('D:/DroneSoccerData/diagnostics/front-memory-baseline-v2')
out.mkdir(exist_ok=True)
env=FrontMemoryObservation(MissionObservation(GuidedLessonEnv(stage=4),single_pass=False))
model,metadata=start_from_checkpoint(Path('runs/return-route-40k-20260927/best_model.zip'),env,device='cuda',seed=7,buffer_size=100000,tensorboard_log=out/'tensorboard')
result=evaluate(model,4,episodes=25,front_memory=True)
result['suites']['return_practice'],result['protocols']['return_practice']=evaluate_return_practice(model,FrontReturnLibrary('D:/DroneSoccerData/return-libraries/front-memory-development-v1'))
(out/'evaluation.json').write_text(json.dumps(result,indent=2))
control=json.loads(Path('D:/DroneSoccerData/runs/recent-retrace-20260927/evaluations/0.json').read_text())
# New observation/bank versions intentionally differ. Compare measured counts, not those identifiers.
fields=('successful_episodes','success_rate','crash_rate','total_passes','episodes_with_two_passes','returned_to_front_episodes','safe_front_return_episodes','front_reached_episodes','outcomes')
comparison={name:{key:[control['suites'][name].get(key),suite.get(key)] for key in fields if key in suite} for name,suite in result['suites'].items()}
passed=all(a==b for suite in comparison.values() for a,b in suite.values())
audit=dict(passed=passed,comparison=comparison,transfer=metadata,actor_updates=model.actor_updates,steps=model.num_timesteps,independent_seeds_used=False)
(out/'audit.json').write_text(json.dumps(audit,indent=2))
print(json.dumps(audit,indent=2),flush=True)
assert passed and model.actor_updates==0 and model.num_timesteps==0
