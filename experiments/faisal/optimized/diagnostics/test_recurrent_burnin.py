from pathlib import Path
import sys,pickle,json,hashlib
import numpy as np,torch
R=Path(__file__).parent;BANK=R/'recurrent-return-histories-20260928'
sys.path.insert(0,str(R.parent))
from src.recurrent_burnin import rebuild_actor_memory
from src.recurrent_baseline import RecurrentPredictor
from sb3_contrib import RecurrentPPO
torch.set_num_threads(1)
manifest=json.loads((BANK/'result.json').read_text());entry=manifest['snapshots'][0]
payload=(BANK/'snapshots'/f"{entry['id']}.pkl").read_bytes()
assert hashlib.sha256(payload).hexdigest()==entry['sha256']
d=pickle.loads(payload)
m=RecurrentPPO.load('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928/model_update_500.zip',device='cuda')
with torch.inference_mode():
    state,start=rebuild_actor_memory(m,d['prefix_observations'])
    for x,y in zip(state,d['recurrent_state']):np.testing.assert_array_equal(x,y)
    with torch.no_grad():m.policy.lstm_actor.bias_ih_l0.add_(.01)
    changed,_=rebuild_actor_memory(m,d['prefix_observations'])
    p=RecurrentPredictor(m)
    for obs in d['prefix_observations']:p.predict(obs)
    for x,y in zip(changed,p.state):np.testing.assert_array_equal(x,y)
    assert any(not np.array_equal(x,y) for x,y in zip(changed,d['recurrent_state']))
for bad in (np.zeros((0,119)),np.zeros((2,118)),np.full((2,119),np.nan)):
    try:rebuild_actor_memory(m,bad)
    except ValueError:pass
    else:raise AssertionError('Invalid prefix accepted')
print('PASS: saved-state equality, changed-weight rebuild, uninterrupted-prefix equality, invalid-prefix rejection; no checkpoint modified')
