import numpy as np
import pytest
from src.return_reset_lessons import ReturnResetLesson,LEVELS
from src.front_memory import FrontMemoryObservation
from src.mission_observation import MissionObservation

@pytest.mark.parametrize('lesson',range(len(LEVELS)))
def test_fresh_reset_geometry_camera_history_and_no_phantom_pass(lesson):
 for seed in range(20):
  env=FrontMemoryObservation(MissionObservation(ReturnResetLesson(lesson,horizon=2),single_pass=False))
  obs,info=env.reset(seed=seed)
  assert obs.shape==(115,) and obs[99]==0
  if lesson==0:assert obs[70]>.5 and obs[71]>.5
  assert info['passes']==info['historical_passes']==0 and not info['full_mission_success']
  assert not info['reset_lesson_success']
  np.testing.assert_array_equal(obs[72:84],np.zeros(12))
  assert obs[109]==0
  if not (obs[70]>.5 and obs[71]>.5):np.testing.assert_array_equal(obs[100:],np.zeros(15))
  for _ in range(2):obs,r,term,trunc,info=env.step(np.zeros(4,np.float32))
  assert not term and trunc and info['passes']==0 and not info['reset_lesson_success']
  with pytest.raises(RuntimeError):env.step(np.zeros(4,np.float32))
  env.close()

@pytest.mark.parametrize('lesson',[-1,6,True,1.5])
def test_invalid_lesson_rejected(lesson):
 with pytest.raises(ValueError):ReturnResetLesson(lesson)
