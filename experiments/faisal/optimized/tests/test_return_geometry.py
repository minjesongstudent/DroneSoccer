import json
from pathlib import Path
import numpy as np
import pytest
from src.sim import HoopEnv
from src.return_geometry import GeometricReturnProgress


def test_front_position_alone_cannot_advance_return():
 base=HoopEnv();base.place([.6,1.,1.],0,[1.05,1.,1.],[1.,0.,0.],.45)
 base._lap_phase='return_side';env=GeometricReturnProgress(base)
 _,_,_,_,info=env.step(np.zeros(4,dtype=np.float32))
 assert not info['geometric_return_advanced'] and base._lap_phase=='return_side'
 assert info['passes']==0


def test_real_learned_success_actions_preserve_physics_passes_and_reward_accounting():
 artifact=Path('D:/DroneSoccerData/diagnostics/coverage-first-two-pass.json')
 if not artifact.exists():pytest.skip('Saved learned trajectory required')
 from src.guided_curriculum import GuidedLessonEnv
 old=GuidedLessonEnv(stage=4);new=GeometricReturnProgress(GuidedLessonEnv(stage=4))
 a,_=old.reset(seed=5000006);b,_=new.reset(seed=5000006)
 np.testing.assert_array_equal(a,b)
 advances=[];corrections=0.
 for row in json.loads(artifact.read_text())['trace']:
  action=np.array(row['action'],dtype=np.float32)
  before=new.unwrapped._progress_distance(new.unwrapped.pos)
  a,ra,ta,tra,ia=old.step(action);b,rb,tb,trb,ib=new.step(action)
  np.testing.assert_array_equal(a,b)
  assert (ta,tra,ia['passes'],ia['outcome'])==(tb,trb,ib['passes'],ib['outcome'])
  if ib['geometric_return_advanced']:advances.append(row['step'])
  corrections+=ib['geometric_reward_correction']
  # Corrected reward remains a single potential difference plus original events.
  from src.sim import R_PROGRESS,PROGRESS_UNIT,R_HIDDEN,R_PASS,R_CRASH
  now=new.unwrapped
  expected=R_PROGRESS*(before-now._progress_distance(now.pos))/PROGRESS_UNIT-R_HIDDEN*(1-now.observe()[6])+R_PASS*(ib['outcome']=='passed')-R_CRASH*tb
  assert rb==pytest.approx(expected,abs=1e-6)
 assert advances and ia['passes']==2 and ia['outcome']=='timeout'
 assert advances[0]<507
 assert corrections>0
 old.close();new.close()


def placed_return(y=.5):
 base=HoopEnv();base.place([1.06,y,1.],0,[1.05,1.,1.],[1.,0.,0.],.45)
 base._lap_phase='return_side'
 return GeometricReturnProgress(base)


def advance(env,direction):
 # Unit-test geometry probe only; never a training controller.
 env.unwrapped.vel=np.array([direction*.4,0.,0.])
 return env.step(np.array([0.,direction,0.,0.],np.float32))


def test_outside_crossing_waits_for_whole_body_clearance_and_latches_once():
 env=placed_return();_,_,t,_,info=advance(env,-1)
 assert not t and env.outside_return_seen and not info['geometric_return_advanced']
 assert (env.unwrapped.pos-env.unwrapped.hoop_centre)@env.unwrapped.hoop_normal > -env.unwrapped._clearance_reach()
 count=0
 for _ in range(8):
  _,_,t,_,info=advance(env,-1);assert not t
  count+=info['geometric_return_advanced']
 assert count==1 and not env.outside_return_seen
 assert info['passes']==0
 env.close()


@pytest.mark.parametrize('y,outcome',[(1.,'wrong_way'),(1.225,'hit')])
def test_illegal_crossings_do_not_advance_or_earn_correction(y,outcome):
 env=placed_return(y);env.outside_return_seen=True
 _,_,t,_,info=advance(env,-1)
 assert t and info['outcome']==outcome
 assert not info['geometric_return_advanced'] and info['geometric_reward_correction']==0
 env.close()


def test_repeated_outside_zero_pass_loop_has_no_renewable_correction():
 env=placed_return();cycles=[];advances=[]
 for cycle in range(3):
  total=0.;count=0
  for direction in [-1]*10+[1]*10:
   _,r,t,_,info=advance(env,direction);assert not t
   total+=r;count+=info['geometric_return_advanced']
  cycles.append(total);advances.append(count)
 assert advances==[1,0,0]
 assert cycles[1]<=1e-6 and cycles[2]<=1e-6
 assert info['passes']==0
 env.outside_return_seen=True;env.reset(seed=14)
 assert not env.outside_return_seen
 env.close()


def test_authentic_bank_adaptation_preserves_input_and_source():
 from src.metric_return_library import MetricReturnLibrary
 from src.geometric_library import GeometricReturnLibrary
 path=Path('D:/DroneSoccerData/return-libraries/front-memory-development-v1')
 if not path.exists():pytest.skip('Authentic source bank unavailable')
 old=MetricReturnLibrary(path);new=GeometricReturnLibrary(path)
 assert old.bank_sha256==new.bank_sha256
 for record in new.records:
  a=old.restore(record['id']);b=new.restore(record['id'])
  np.testing.assert_array_equal(a['observation'],b['observation'])
  assert not b['environment'].outside_return_seen
  a['environment'].close();b['environment'].close()
