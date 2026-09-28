import numpy as np
from src.return_reset_curriculum import ResetScheduler,ResetMixture
from src import live

def test_all_lessons_sampled_without_inventing_history_or_passes():
 env=ResetMixture(ResetScheduler(),metric_ring=True,coverage=True)
 counts={str(k):0 for k in range(6)};counts.update(single=0,full=0)
 for seed in range(600):
  obs,info=env.reset(seed=seed)
  assert obs.shape==(119,) and info['passes']==0
  key=str(info['reset_lesson']) if 'reset_lesson' in info else 'single' if obs[99]>.5 else 'full'
  counts[key]+=1
 assert all(counts[str(k)]>15 for k in range(6))
 assert 180<counts['single']<300 and 80<counts['full']<160
 assert env.scheduler.level==0
 env.close()

def test_live_coverage_contract_and_display(tmp_path):
 session=live.Session(40000,'coverage',runs_dir=tmp_path,reset_coverage=True)
 assert session.metric_ring and session.reset_lessons and not session.repair_anchor
 assert session.observation_dim==119
 assert len(session.training_mix())==9
 assert abs(sum(x['probability'] for x in session.training_mix())-1)<1e-8
