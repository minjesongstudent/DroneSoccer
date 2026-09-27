import numpy as np
import pytest

from src.completion_curriculum import CompletionEnv, training_mix
from src.mission_observation import MissionObservation, OBSERVATION_DIM
from src.guided_curriculum import GuidedLessonEnv


@pytest.mark.parametrize('focus', range(4, 9))
def test_complete_tasks_remain_in_training_mix(focus):
    p = training_mix(focus)
    assert sum(p) == pytest.approx(1)
    assert p[2] == p[3] == .2 and p[focus] >= .4


@pytest.mark.parametrize('practice', [2, 3, 4, 5])
def test_sampled_task_matches_its_real_rules(monkeypatch, practice):
    mixed = CompletionEnv()
    monkeypatch.setattr(mixed, '_sample_practice_stage', lambda: practice)
    env = MissionObservation(mixed)
    reference = MissionObservation(GuidedLessonEnv(stage=practice), single_pass=practice < 4)
    actual, info = env.reset(seed=2)
    expected, _ = reference.reset(seed=2)
    assert actual.shape == (OBSERVATION_DIM,)
    np.testing.assert_array_equal(actual, expected)
    assert actual[-1] == (practice < 4)
    assert info['focus_stage'] == 4
    for _ in range(600):
        action = np.zeros(4)
        a, b = env.step(action), reference.step(action)
        np.testing.assert_array_equal(a[0], b[0])
        assert a[1:4] == b[1:4]
        assert a[4]['outcome'] == b[4]['outcome']
        if a[2] or a[3]:
            break
    assert a[4]['steps'] == (150 if practice < 4 else 600)


def test_task_choice_and_memory_reset_reproduce_from_seed():
    env = MissionObservation(CompletionEnv())
    for seed in range(20):
        before, info = env.reset(seed=seed)
        env.step([1, 1, 1, 1])
        after, other_info = env.reset(seed=seed)
        np.testing.assert_array_equal(before, after)
        assert info['stage'] == other_info['stage']
