"""CUDA checks on authentic histories; never writes model weights."""
from pathlib import Path
import sys, json, pickle, hashlib
import numpy as np
import torch
from sb3_contrib import RecurrentPPO
from sb3_contrib.common.recurrent.type_aliases import RNNStates

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.recurrent_burnin import rebuild_actor_memory, rebuild_actor_critic_memory

torch.set_num_threads(1)
bank = ROOT / 'diagnostics/recurrent-return-histories-20260928'
entry = json.loads((bank / 'result.json').read_text())['snapshots'][0]
payload = (bank / 'snapshots' / (entry['id'] + '.pkl')).read_bytes()
assert hashlib.sha256(payload).hexdigest() == entry['sha256']
data = pickle.loads(payload)
prefix = data['prefix_observations']
model = RecurrentPPO.load('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip', device='cuda')
policy = model.policy
assert policy.lstm_critic is not None

def check():
    policy.set_training_mode(True)
    rebuilt, start = rebuild_actor_critic_memory(model, prefix)
    assert policy.training and not start.any()
    assert all(not t.requires_grad for pair in rebuilt for t in pair)
    actor, _ = rebuild_actor_memory(model, prefix)
    for x, y in zip(rebuilt.pi, actor):
        np.testing.assert_array_equal(x.cpu().numpy(), y)
    # Independent sequence processing includes current observation, with only
    # the first sample marked as a reset. Compare current outputs and memory.
    observations = np.concatenate([prefix, data['observation'][None]], axis=0)
    zero = lambda: torch.zeros(policy.lstm_hidden_state_shape, device='cuda')
    states = RNNStates((zero(), zero()), (zero(), zero()))
    starts = torch.zeros(len(observations), device='cuda'); starts[0] = 1
    policy.set_training_mode(False)
    with torch.no_grad():
        full = policy.forward(torch.as_tensor(observations, device='cuda'), states, starts, deterministic=True)
        current, _ = policy.obs_to_tensor(data['observation'])
        resumed = policy.forward(current, rebuilt, start, deterministic=True)
    for all_outputs, current_output in zip(full[:3], resumed[:3]):
        torch.testing.assert_close(all_outputs[-1:], current_output, rtol=1e-4, atol=1e-5)
    for pair, other in zip(full[3], resumed[3]):
        for x, y in zip(pair, other):
            torch.testing.assert_close(x, y, rtol=1e-4, atol=1e-5)
    return rebuilt

old = check()
with torch.no_grad():
    policy.lstm_actor.bias_ih_l0.add_(.01)
    policy.lstm_critic.bias_ih_l0.add_(.02)
new = check()
assert all(any(not torch.equal(x, y) for x, y in zip(a, b)) for a, b in zip(old, new))
for invalid in (np.zeros((0,119)), np.zeros((1,118)), np.full((1,119), np.nan)):
    try: rebuild_actor_critic_memory(model, invalid)
    except ValueError: pass
    else: raise AssertionError('Invalid prefix accepted')
print('PASS: actor parity; full-sequence action/value/logprob and dual-state parity; changed actor/critic rebuild; detached boundary; mode restoration; invalid input rejection. No checkpoint modified.')
