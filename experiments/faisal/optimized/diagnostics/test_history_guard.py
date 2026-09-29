"""CUDA tests for actual proposal acceptance, rollback and persisted bank."""
from pathlib import Path
import sys,tempfile
from copy import deepcopy
from unittest.mock import patch
import torch,numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.history_guard_policy import HistoryGuardSAC
from src.geometric_policy import GeometricSAC
from src.metric_policy import MetricActorPolicy
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap
from stable_baselines3.common.logger import configure

torch.set_num_threads(1)
env=wrap(GuidedLessonEnv(stage=4),metric_ring=True,geometric_return=True)
m=HistoryGuardSAC(MetricActorPolicy,env,device='cuda',buffer_size=6000,learning_starts=0,seed=14)
m.set_logger(configure(None,[]))
o,_=env.reset(seed=123)
bank=np.repeat(o[None],16,axis=0)
m.set_history_bank(bank)
assert m.history_kl()==0
x=torch.as_tensor(bank,device='cuda')
stats=m._actor_step(x,m.ent_coef_tensor)
assert stats['accepted'] and m.history_kl()<=m.history_max_kl
before=deepcopy(m.actor.state_dict());opt=deepcopy(m.actor.optimizer.state_dict())
count=m.actor_updates
def bad_proposal(self,observations,coefficient):
    with torch.no_grad():self.actor.mu.bias.add_(100)
    self.actor_updates+=1;self.anchor_accepted+=1;self.actor_attempts+=1
    return dict(accepted=True,kl=0.)
with patch.object(GeometricSAC,'_actor_step',bad_proposal):
    stats=m._actor_step(x,m.ent_coef_tensor)
assert not stats['accepted'] and m.actor_updates==count and m.history_rejected==1
assert all(torch.equal(v,m.actor.state_dict()[k]) for k,v in before.items())
for k,state in opt['state'].items():
    for n,v in state.items():
        assert torch.equal(v,m.actor.optimizer.state_dict()['state'][k][n]) if torch.is_tensor(v) else v==m.actor.optimizer.state_dict()['state'][k][n]
with tempfile.TemporaryDirectory() as tmp:
    m.save(Path(tmp)/'guard.zip')
    loaded=HistoryGuardSAC.load(Path(tmp)/'guard.zip',device='cuda')
    assert torch.equal(m.history_observations,loaded.history_observations)
    np.testing.assert_array_equal(m.predict(o,deterministic=True)[0],loaded.predict(o,deterministic=True)[0])
    assert loaded.history_kl()==m.history_kl() and loaded.history_rejected==1
env.close()
print('PASS: actual update, fixed-history rejection, actor/optimizer rollback, bank/model roundtrip')
