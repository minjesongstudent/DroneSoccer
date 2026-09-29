"""Fixed validation and saved training safeguards for the historical producer."""
from pathlib import Path
import sys,json,hashlib
import torch
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'lap-recurrent-anchor10-20260928/source_snapshot'))
from src.metric_policy import MetricRingSAC
from src.unified_evaluation import evaluate

torch.set_num_threads(1)
path=Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
before=hashlib.sha256(path.read_bytes()).hexdigest()
m=MetricRingSAC.load(path,device='cuda')
fields=['num_timesteps','refill_transitions','calibration_updates','critic_calibration_updates','actor_updates',
        'actor_learning_rate','critic_learning_rate','anchor_coefficient','anchor_max_kl','actor_attempts',
        'anchor_accepted','anchor_rejected','last_anchor_kl','gamma','batch_size','trace_length','trace_lambda']
metadata={k:getattr(m,k) for k in fields}
metadata['reference_equals_current']=all(torch.equal(v,m.reference_actor.state_dict()[k]) for k,v in m.actor.state_dict().items())
with torch.inference_mode():
    report=evaluate(m,4,metric_ring=True,geometric_return=True)
assert hashlib.sha256(path.read_bytes()).hexdigest()==before
(ROOT/'producer_baseline.json').write_text(json.dumps(dict(sha256=before,metadata=metadata,evaluation=report,model_unchanged=True),indent=2))
print(json.dumps(dict(metadata=metadata,suites={k:{f:v[f] for f in ('successful_episodes','crash_rate','returned_to_front_episodes')} for k,v in report['suites'].items()}),indent=2))
