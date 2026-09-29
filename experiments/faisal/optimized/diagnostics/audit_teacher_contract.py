"""Verify saved action labels against the deterministic source actor on public inputs."""
from pathlib import Path
import sys,json,hashlib
import numpy as np
import torch

ROOT=Path(__file__).parent
RUN=Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
sys.path.insert(0,str(RUN/'source_snapshot'))
from src.metric_policy import MetricRingSAC
from sb3_contrib import RecurrentPPO

torch.set_num_threads(1)
source=Path('D:/DroneSoccerData/runs/reset-coverage-20260928/checkpoints/model_15000_steps.zip')
teacher=MetricRingSAC.load(source,device='cuda')
student=RecurrentPPO.load(RUN/'model_update_500.zip',device='cuda')
seeds=json.loads((RUN/'protocol.json').read_text())['fit_lap_seeds']
rows=[]
for seed in seeds:
    assert seed%4!=3
    with np.load(f'D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories/{seed}.npz') as d:
        observations=d['observations'][:-1];target=d['actions']
    with torch.inference_mode():
        predicted=teacher.predict(observations,deterministic=True)[0]
    delta=predicted-target
    rows.append(dict(seed=seed,mse=float(np.mean(delta**2)),max_absolute_error=float(np.max(np.abs(delta)))))
result=dict(device=torch.cuda.get_device_name(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    rows=rows,teacher_network=str(teacher.actor),student_network=str(student.policy),
    teacher_actor_parameters=sum(p.numel() for p in teacher.actor.parameters()),
    student_actor_parameters=sum(p.numel() for n,p in student.policy.named_parameters() if n.startswith(('lstm_actor.','mlp_extractor.policy_net.','action_net.'))),
    note='FIT-only deterministic-label verification, no optimization, no new flight. Teacher receives119D public vector directly; student routes it through LSTM128 and Tanh128 before action head.')
(ROOT/'teacher_contract_audit.json').write_text(json.dumps(result,indent=2))
print(json.dumps({k:v for k,v in result.items() if not k.endswith('network')},indent=2))
