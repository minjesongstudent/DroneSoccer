from types import SimpleNamespace

import numpy as np
import pytest

from src.completion_curriculum import Curriculum
from src.guided_curriculum import GuidedLessonEnv
from src.mission_observation import MissionObservation
from src import return_mixture


class FakePractice(MissionObservation):
    def __init__(self, library):
        super().__init__(GuidedLessonEnv(stage=4), single_pass=False)
        self.library = SimpleNamespace(manifest={'split': library})


def test_mixture_has_one_unchanged_observation_contract_and_exact_tasks(monkeypatch):
    monkeypatch.setattr(return_mixture, 'ReturnPracticeEnv', FakePractice)
    env = return_mixture.ReturnMixtureEnv('train')
    seen = set()
    for seed in range(50):
        obs, info = env.reset(seed=seed)
        first = obs.copy()
        chosen = (info['stage'], info['return_practice'])
        seen.add(chosen)
        assert obs.shape == (100,)
        assert obs[99] == (info['stage'] < 4)
        assert env.unwrapped is env.active.unwrapped
        env.step(np.zeros(4))
        again, restored = env.reset(seed=seed)
        np.testing.assert_array_equal(first, again)
        assert chosen == (restored['stage'], restored['return_practice'])
    assert seen == {(2, False), (3, False), (4, False), (5, False), (4, True)}
    env.close()


def test_development_bank_cannot_be_used_for_training(monkeypatch):
    monkeypatch.setattr(return_mixture, 'ReturnPracticeEnv', FakePractice)
    with pytest.raises(ValueError, match='development'):
        return_mixture.ReturnMixtureEnv('development')


def test_short_return_trials_never_promote_whole_mission_curriculum():
    curriculum = Curriculum()
    original = curriculum.state()
    for _ in range(200):
        assert not curriculum.record({'stage': 4, 'return_practice': True,
                                      'passes': 3, 'outcome': 'timeout'})
    assert curriculum.state() == original
    curriculum.record({'stage': 4, 'return_practice': False, 'passes': 2, 'outcome': 'timeout'})
    assert curriculum.rate == 1 and len(curriculum.recent) == 1


def test_public_mix_has_complete_single_pass_rehearsal():
    mix = return_mixture.display_mix(4)
    assert sum(x['probability'] for x in mix) == pytest.approx(1)
    assert mix[0]['stage'] == 2 and mix[1]['stage'] == 3
    assert mix[0]['probability']+mix[1]['probability'] == pytest.approx(.4)
    assert mix[-1]['name'] == 'Authentic mid-flight return'
