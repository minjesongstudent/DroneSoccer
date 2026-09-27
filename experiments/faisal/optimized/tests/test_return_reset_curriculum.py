from copy import deepcopy
import numpy as np
import pytest
from src.return_reset_curriculum import ResetScheduler,ResetMixture,VERSION,LESSON_VERSION,EPISODES,SEED_START

def report(step,scores=None):return dict(version=VERSION,lesson_version=LESSON_VERSION,episodes=EPISODES,seed_start=SEED_START,step=step,successes=[25]*6 if scores is None else scores)

def test_two_checks_retention_previous_lessons_and_resume():
 s=ResetScheduler()
 assert not s.observe(report(0),.96,.92)['advanced']
 assert s.observe(report(5000),.96,.92)['advanced'] and s.level==1
 s.observe(report(10000,[25,19,25,25,25,25]),.96,.92)
 assert s.streak==0
 s.observe(report(15000),.83,.92);assert s.streak==0
 s.observe(report(20000),.96,.79);assert s.streak==0
 s.observe(report(25000),.96,.92);assert s.streak==1
 r=ResetScheduler.restore(s.state());assert r.state()==s.state() and r.level==1
 assert r.observe(report(30000),.96,.92)['advanced'] and r.level==2
 assert not r.observe(report(35000),.96,.92)['full_mission_accepted']

@pytest.mark.parametrize('field,value',[('version','bad'),('episodes',1),('seed_start',7000000),('successes',[25]*5),('successes',[26]*6)])
def test_protocol_drift_rejected_transactionally(field,value):
 s=ResetScheduler();old=s.state();d=report(0);d[field]=value
 with pytest.raises(ValueError):s.observe(d,.96,.92)
 assert s.state()==old

def test_repeated_step_and_tampered_resume_rejected():
 s=ResetScheduler();s.observe(report(0),.96,.92)
 with pytest.raises(ValueError):s.observe(report(0),.96,.92)
 state=s.state();state['history'][0]['level']=5
 with pytest.raises(ValueError):ResetScheduler.restore(state)

def test_mixture_rehearses_all_task_types_and_fresh_histories():
 s=ResetScheduler();s.observe(report(0),1,1);s.observe(report(1),1,1)
 env=ResetMixture(s);seen=set()
 for seed in range(100):
  obs,info=env.reset(seed=seed)
  assert obs.shape==(115,) and info['passes']==0
  np.testing.assert_array_equal(obs[72:84],np.zeros(12))
  seen.add(('reset',info['reset_lesson']) if 'reset_lesson' in info else ('normal',info['stage']))
  assert obs[99]==float(info['stage']<4)
 assert {('normal',2),('normal',3),('normal',4),('reset',0),('reset',1)}==seen
 env.close()


def test_mastered_skill_regression_stops_and_survives_restore():
 s=ResetScheduler();s.observe(report(0),.96,.92);s.observe(report(5000),.96,.92)
 assert s.level==1
 decision=s.observe(report(10000,[19,25,25,25,25,25]),.96,.92)
 assert decision['should_stop'] and decision['stop_reason']=='reset_mastered_lesson_regression'
 assert not decision['advanced']
 restored=ResetScheduler.restore(s.state())
 assert restored.should_stop and restored.stop_reason==s.stop_reason
 with pytest.raises(ValueError,match='already stopped'):restored.observe(report(15000),.96,.92)


@pytest.mark.parametrize('command',['resume','extend'])
def test_live_controls_cannot_bypass_reset_retention_stop(tmp_path,command):
 from src.live import Session
 session=Session(20000,'stopped-reset',runs_dir=tmp_path,repair_anchor=True)
 scheduler=session.reset_scheduler
 scheduler.observe(report(0),.96,.92);scheduler.observe(report(5000),.96,.92)
 scheduler.observe(report(10000,[15,25,25,25,25,25]),.96,.92)
 session.update(mode='Replay',steps=10000)
 with pytest.raises(ValueError,match='Evaluation stopped'):session.control(command)
 assert not session.wake.is_set()
