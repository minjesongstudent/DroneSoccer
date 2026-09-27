from dataclasses import asdict

import numpy as np
import pytest

from src import guided_curriculum as guided
from src.curriculum import LessonEnv


@pytest.mark.parametrize('stage', range(len(guided.STAGES)))
def test_guided_stage_mapping_reset_and_rewards_match_reference(stage):
    env = guided.GuidedLessonEnv(stage=stage)
    reference = LessonEnv(stage=guided.reference_stage(stage))
    obs, info = env.reset(seed=91)
    reference_obs, _ = reference.reset(seed=91)
    assert info['stage'] == stage
    assert info['reference_stage'] == guided.reference_stage(stage)
    assert env.stage == guided.reference_stage(stage)
    for name in ('cage', 'drone', 'hoop', 'camera', 'pad'):
        assert asdict(getattr(env, name)) == asdict(getattr(reference, name))
    if stage >= 2:
        np.testing.assert_array_equal(obs, reference_obs)
        for _ in range(15):
            actual = env.step([0, .5, 0, 0])
            expected = reference.step([0, .5, 0, 0])
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]
            assert actual[4]['stage'] == stage
            assert actual[4]['outcome'] == expected[4]['outcome']
            if actual[2] or actual[3]:
                break


def test_guided_stage_changes_only_at_reset_and_maps_delay_lesson():
    env = guided.GuidedLessonEnv(stage=0)
    env.reset(seed=4)
    env.set_stage(8)
    assert env.guided_stage == 0 and env.stage == 0
    assert env.step([0, 0, 0, 0])[4]['stage'] == 0
    _, info = env.reset(seed=4)
    assert env.guided_stage == 8 and env.stage == 6
    assert info['stage'] == 8 and env.randomise
    assert env.ranges.delay_steps == (0, 3)
    assert env.ranges.rate_hz == (10., 10.)


@pytest.mark.parametrize('stage, bounds', [(0, (.25, .35)), (1, (.35, .50))])
def test_close_lessons_start_outside_ring_and_require_full_body_pass(stage, bounds):
    env = guided.GuidedLessonEnv(stage=stage)
    for seed in range(30):
        env.reset(seed=seed)
        gap = -(env.pos-env.hoop_centre) @ env.hoop_normal
        assert bounds[0] <= gap <= bounds[1]
        assert not env._hit() and not env._hits_hanger() and not env._left_cage()
        crossed = False
        for _ in range(150):
            _, _, terminated, truncated, info = env.step([0, .5, 0, 0])
            along = (env.pos-env.hoop_centre) @ env.hoop_normal
            if 0 < along < .1:
                crossed = True
                assert info['passes'] == 0 and not terminated
            if terminated or truncated:
                break
        assert crossed
        assert terminated and not truncated and info['outcome'] == 'passed'
        assert along > env.drone.length/2+env.hoop.thickness/2


@pytest.mark.parametrize('stage', range(4))
def test_first_four_lessons_require_single_pass_window(stage):
    curriculum = guided.GuidedCurriculum({'stage': stage})
    for _ in range(99):
        assert not curriculum.record({'stage': stage, 'outcome': 'passed', 'passes': 1})
    assert curriculum.record({'stage': stage, 'outcome': 'passed', 'passes': 1})
    assert curriculum.stage == stage+1
    assert not curriculum.recent
    assert guided.GuidedCurriculum(curriculum.state()).state() == curriculum.state()


def test_guided_laps_require_multiple_passes_and_survival():
    curriculum = guided.GuidedCurriculum({'stage': 4})
    for _ in range(100):
        assert not curriculum.record({'stage': 4, 'outcome': 'hit', 'passes': 3})
    for _ in range(70):
        advanced = curriculum.record({'stage': 4, 'outcome': 'timeout', 'passes': 2})
    assert advanced and curriculum.stage == 5
    assert not curriculum.record({'stage': 4, 'outcome': 'timeout', 'passes': 2})
    assert not curriculum.recent


def test_demonstrations_are_real_successful_transitions_with_separate_seeds():
    progress = []
    transitions, summary = guided.collect_demonstrations(
        episodes=12, stages=(0, 1, 2, 3), progress=lambda n, total: progress.append((n, total)))
    assert summary['successful'] == summary['attempted'] == 12
    assert summary['steps'] == len(transitions)
    assert summary['seed_start'] == 1_000_000 and summary['seed_end'] == 1_000_011
    assert all(1_000_000 <= seed < 3_000_000 for seed in summary['seeds'])
    assert set(summary['per_stage']) == {'0', '1', '2', '3'}
    assert progress[-1] == (12, 12)
    env = None
    for transition in transitions:
        if env is None:
            env = guided.GuidedLessonEnv(stage=transition['stage'])
            obs, _ = env.reset(seed=transition['seed'])
        np.testing.assert_array_equal(obs, transition['obs'])
        obs, reward, terminated, truncated, info = env.step(transition['action'])
        np.testing.assert_array_equal(obs, transition['next_obs'])
        assert reward == transition['reward']
        assert (terminated, truncated) == (transition['terminated'], transition['truncated'])
        assert info['outcome'] == transition['info']['outcome']
        if terminated or truncated:
            assert info['outcome'] == 'passed' and info['passes'] == 1
            env.close()
            env = None
    assert env is None


def test_failed_demonstration_episodes_are_filtered(monkeypatch):
    original_reset = guided.GuidedLessonEnv.reset
    teacher = guided._demonstration_action

    def reset_with_seed(self, *, seed=None, options=None):
        self.test_seed = seed
        return original_reset(self, seed=seed, options=options)

    monkeypatch.setattr(guided.GuidedLessonEnv, 'reset', reset_with_seed)
    monkeypatch.setattr(guided, '_demonstration_action',
                        lambda env: np.zeros(4, np.float32) if env.test_seed == 1_000_001 else teacher(env))
    transitions, summary = guided.collect_demonstrations(episodes=4, stages=(0,), min_success_rate=.75)
    assert summary['successful'] == 3 and summary['attempted'] == 4
    assert summary['attempted_steps'] == summary['steps']+150
    assert 1_000_001 not in summary['seeds']
    assert all(t['seed'] != 1_000_001 for t in transitions)
    with pytest.raises(RuntimeError, match='Insufficient successful demonstrations'):
        guided.collect_demonstrations(episodes=4, stages=(0,), min_success_rate=1.)


@pytest.mark.parametrize('kwargs', [
    {'seed_start': 3_000_000}, {'seed_start': 2_999_999, 'episodes': 3},
    {'seed_start': 0}, {'stages': (4,)}, {'episodes': 0},
])
def test_invalid_or_evaluation_overlapping_demonstrations_fail(kwargs):
    with pytest.raises(ValueError):
        guided.collect_demonstrations(**kwargs)
