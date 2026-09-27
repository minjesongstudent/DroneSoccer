import numpy as np
import pytest

from src import live_evaluation
from src.curriculum import LessonEnv


class StraightAhead:
    num_timesteps = 42

    def predict(self, obs, deterministic):
        assert deterministic
        return np.array([0, .5, 0, 0]), None


def test_fixed_original_benchmark_is_independent_of_guided_lesson():
    close = live_evaluation.evaluate(StraightAhead(), 0, episodes=3)
    varied = live_evaluation.evaluate(StraightAhead(), 3, episodes=3)
    assert close['suites']['original_centered'] == varied['suites']['original_centered']
    assert close['protocols']['original_centered'] == varied['protocols']['original_centered']
    assert close['suites']['original_centered']['successful_episodes'] == 3
    assert close['protocols']['lesson']['id'] != varied['protocols']['lesson']['id']
    # Closer starts may finish sooner, but cannot redefine the original test.
    assert sum(r['steps'] for r in close['suites']['lesson']['records']) < sum(
        r['steps'] for r in close['suites']['original_centered']['records'])
    env = LessonEnv(stage=0)
    model = StraightAhead()
    for record in close['suites']['original_centered']['records']:
        obs, _ = env.reset(seed=record['seed'])
        while True:
            obs, _, term, trunc, info = env.step(model.predict(obs, True)[0])
            if term or trunc:
                break
        assert record == {key: info[key] for key in ('outcome', 'passes', 'steps')} | {'seed': record['seed']}
    env.close()


@pytest.mark.parametrize('stage', [2, 3, 4, 8])
def test_guided_single_pass_criterion_uses_guided_stage_not_parent_stage(monkeypatch, stage):
    class PassedEnv:
        def __init__(self, stage=0):
            self.__dict__.update(LessonEnv().__dict__)

        def reset(self, seed):
            return np.zeros(84), {}

        def step(self, action):
            return np.zeros(84), 30, True, False, {'outcome': 'passed', 'passes': 1, 'steps': 10}

        def close(self):
            pass

    monkeypatch.setattr(live_evaluation, 'GuidedLessonEnv', PassedEnv)
    report = live_evaluation.evaluate(StraightAhead(), stage, episodes=1)
    assert report['suites']['lesson']['success_rate'] == (1 if stage < 4 else 0)
    assert report['suites']['full_task']['success_rate'] == 0


def test_invalid_evaluation_episode_count_rejected():
    with pytest.raises(ValueError):
        live_evaluation.evaluate(StraightAhead(), 0, episodes=0)
