from pathlib import Path
import sys
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.sim import HoopEnv
from src.mission_observation import MissionObservation
from src.front_memory import FrontMemoryObservation
from src.ring_estimate import MetricRingObservation
from src.return_geometry import GeometricReturnProgress
from src.recurrent_snapshot_rollout import collect_natural_segment,collect_segment,evaluate_segment
from src.recurrent_update_guard import distributions,guarded_step
torch.set_num_threads(1)
m=RecurrentPPO.load('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip',device='cuda')
env=GeometricReturnProgress(MetricRingObservation(FrontMemoryObservation(MissionObservation(HoopEnv(level=1.),single_pass=False))))
torch.manual_seed(986000)
s=collect_natural_segment(m,env,2960000,8)
assert s.natural_start and s.prefix.shape==(0,119)
values,logs,_=evaluate_segment(m,s)
np.testing.assert_allclose(logs.detach().cpu(),s.log_probs,atol=2e-5)
np.testing.assert_allclose(values.detach().cpu().numpy().flatten(),s.values,atol=2e-5)
m.policy.zero_grad();(-logs.mean()).backward()
assert any(p.grad is not None and p.grad.abs().sum()>0 for p in m.policy.parameters())
reference=distributions(m,[s,s])
torch.testing.assert_close(reference[:len(s.observations)],reference[len(s.observations):],rtol=0,atol=0)
before={k:v.clone() for k,v in m.policy.state_dict().items()}
opt=torch.optim.Adam(m.policy.parameters(),lr=1.)
assert not guarded_step(m,opt,[s,s],reference)['accepted']
assert all(torch.equal(v,m.policy.state_dict()[k]) for k,v in before.items())
# Fresh reset does not inherit hidden state from the previous sampled flight.
torch.manual_seed(986000);same=collect_natural_segment(m,env,2960000,8)
np.testing.assert_array_equal(s.observations,same.observations)
np.testing.assert_array_equal(s.actions,same.actions)
assert not same.mission_ended
continued=collect_segment(m,env,same.next_observation,same.next_prefix,8)
assert not continued.natural_start and len(continued.prefix)==8
try:collect_segment(m,env,same.next_observation,np.empty((0,119)),1)
except ValueError:pass
else:raise AssertionError('Midflight empty history accepted')
env.close()
print('PASS: genuine reset zero-state likelihood/value parity, gradients, deterministic seeded reset isolation, continuation prefix, missing midflight history rejected')
