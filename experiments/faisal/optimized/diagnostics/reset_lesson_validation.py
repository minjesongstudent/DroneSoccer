"""Fixed reset-lesson validation of preserved initial model; no acceptance test."""
import json
from pathlib import Path
import torch
from src.front_policy import FrontMemorySAC
from src.return_reset_curriculum import evaluate_reset_lessons,ResetScheduler

torch.set_num_threads(1)
m=FrontMemorySAC.load('D:/DroneSoccerData/runs/front-memory-20260927/initial_model.zip',device='cuda')
r=evaluate_reset_lessons(m)
s=ResetScheduler();decision=s.observe(r,24/25,23/25)
p=Path('D:/DroneSoccerData/diagnostics/reset-lesson-fixed-baseline.json')
assert not p.exists()
p.write_text(json.dumps(dict(evaluation=r,scheduler=s.state(),decision=decision),indent=2))
print(json.dumps(dict(successes=r['successes'],decision=decision)),flush=True)
