"""Bounded CUDA integration smoke test, no candidate checkpoint exported."""
from pathlib import Path
import sys, json, pickle, hashlib
from types import SimpleNamespace
import numpy as np
import torch
from sb3_contrib import RecurrentPPO

ROOT = Path(__file__).resolve().parents[1]
BANK = ROOT / 'diagnostics/recurrent-return-histories-20260928'
sys.path.insert(0, str(BANK / 'source_snapshot'))
import src
src.__path__.append(str(ROOT / 'src'))
from src.recurrent_snapshot_rollout import collect_segment, evaluate_segment, advantages
from src.recurrent_burnin import rebuild_actor_critic_memory

torch.set_num_threads(1); torch.manual_seed(982001)
source = Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip')
source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
entry = json.loads((BANK/'result.json').read_text())['snapshots'][0]
payload = (BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
assert hashlib.sha256(payload).hexdigest() == entry['sha256']
data = pickle.loads(payload)
model = RecurrentPPO.load(source, device='cuda')
env = data['environment']
segment = collect_segment(model, env, data['observation'], data['prefix_observations'], 8)
assert not segment.mission_ended, 'Selected smoke prefix ended before split test'
assert len(segment.next_prefix) == len(segment.prefix) + len(segment.observations)
np.testing.assert_array_equal(segment.next_prefix[:len(segment.prefix)], segment.prefix)
with torch.no_grad():
    values, logs, _ = evaluate_segment(model, segment)
np.testing.assert_allclose(values.flatten().cpu(), segment.values, atol=2e-5, rtol=1e-4)
np.testing.assert_allclose(logs.cpu(), segment.log_probs, atol=2e-5, rtol=1e-4)
old_state, _ = rebuild_actor_critic_memory(model, segment.next_prefix)

# Two real optimizer steps. Every likelihood evaluation rebuilds prefix memory.
# These are software integration tests, not a reported learning experiment.
optimizer = torch.optim.Adam(model.policy.parameters(), lr=1e-5)
adv, returns = advantages(segment)
adv = torch.as_tensor(adv, device='cuda'); adv = (adv-adv.mean())/(adv.std()+1e-8)
targets = torch.as_tensor(returns, dtype=torch.float32, device='cuda')
old_logs = torch.as_tensor(segment.log_probs, device='cuda')
losses = []
for _ in range(2):
    values, logs, entropy = evaluate_segment(model, segment)
    ratio = (logs-old_logs).exp()
    actor_loss = -torch.minimum(ratio*adv, ratio.clamp(.8,1.2)*adv).mean()
    loss = actor_loss + .01*(values.flatten()-targets).square().mean()
    assert torch.isfinite(loss)
    optimizer.zero_grad(); loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.policy.parameters() if p.grad is not None)
    torch.nn.utils.clip_grad_norm_(model.policy.parameters(), .5); optimizer.step()
    losses.append(float(loss.item()))
new_state, _ = rebuild_actor_critic_memory(model, segment.next_prefix)
assert all(any(not torch.equal(x,y) for x,y in zip(a,b)) for a,b in zip(old_state,new_state))
continuation = collect_segment(model, env, segment.next_observation, segment.next_prefix, 600)
assert continuation.mission_ended and continuation.bootstrap == 0
assert continuation.final_info['steps'] <= 600
np.testing.assert_array_equal(continuation.prefix, segment.next_prefix)
env.close()

# Explicit numerical GAE references: final mission termination vs collector cut.
ended = SimpleNamespace(rewards=np.array([1.,2.]), values=np.array([.5,.6]), bootstrap=0.)
cut = SimpleNamespace(rewards=ended.rewards, values=ended.values, bootstrap=3.)
np.testing.assert_allclose(advantages(ended,1.,1.)[1], [3.,2.], atol=1e-6)
np.testing.assert_allclose(advantages(cut,1.,1.)[1], [6.,5.], atol=1e-6)
assert hashlib.sha256(source.read_bytes()).hexdigest() == source_hash
print(json.dumps(dict(result='PASS',initial_steps=len(segment.rewards),continuation_steps=len(continuation.rewards),
    optimizer_steps=2,losses=losses,outcome=continuation.final_info['outcome'],source_unchanged=True,
    checks=['pre-update values/logprob parity','authentic prefix concatenation','actor and critic updated memory',
            'finite gradients','mission terminal bootstrap zero','collector bootstrap GAE','no cross-reset GAE'],
    acceptance=False)))
