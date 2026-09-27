from dataclasses import asdict

import numpy as np
import pytest

from src.guided_curriculum import GuidedLessonEnv, STAGES, reference_stage
from src.unified_curriculum import Curriculum, MixedLessonEnv, training_mix


@pytest.mark.parametrize('stage', range(len(STAGES)))
def test_mix_has_only_available_layouts_and_keeps_frontier_probability(stage):
    probabilities = training_mix(stage)
    assert len(probabilities) == stage + 1
    assert sum(probabilities) == pytest.approx(1.)
    assert all(p > 0 for p in probabilities)
    assert probabilities[-1] == (1. if stage == 0 else .6)
    if 0 < stage <= 3:
        np.testing.assert_allclose(probabilities[:-1], .4 / stage)
    if stage == 4:
        np.testing.assert_allclose(probabilities, [.05, .05, .15, .15, .6])


@pytest.mark.parametrize('stage', range(len(STAGES)))
def test_reset_seed_repeats_selection_geometry_and_transitions(stage):
    env = MixedLessonEnv(stage=stage)
    other = MixedLessonEnv(stage=stage)
    for seed in range(12):
        obs, info = env.reset(seed=seed)
        expected, other_info = other.reset(seed=seed)
        np.testing.assert_array_equal(obs, expected)
        assert 0 <= info['stage'] <= stage
        assert info['stage'] == other_info['stage']
        assert info['focus_stage'] == stage
        assert env.guided_stage == stage and env.stage == reference_stage(stage)
        for _ in range(3):
            actual = env.step([.1, .3, 0., -.1])
            expected = other.step([.1, .3, 0., -.1])
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:4] == expected[1:4]
            assert actual[4]['outcome'] == expected[4]['outcome']
    env.close()
    other.close()


@pytest.mark.parametrize('focus, practice', [(2, 0), (4, 0), (4, 3), (8, 0), (8, 5)])
def test_mixed_start_uses_existing_geometry_without_changing_focus_dynamics(monkeypatch, focus, practice):
    env = MixedLessonEnv(stage=focus)
    fixed = GuidedLessonEnv(stage=practice)
    monkeypatch.setattr(env, '_sample_practice_stage', lambda: practice)
    obs, info = env.reset(seed=91)
    expected, _ = fixed.reset(seed=91)
    np.testing.assert_array_equal(obs, expected)
    for name in ('cage', 'drone', 'hoop', 'camera', 'pad'):
        assert asdict(getattr(env, name)) == asdict(getattr(fixed, name))
    assert env.guided_stage == focus and env.stage == reference_stage(focus)
    assert info['stage'] == practice and info['focus_stage'] == focus
    assert info['reference_stage'] == reference_stage(focus)
    assert env.randomise == (focus >= 6)
    if focus >= 6:
        assert env.ranges.delay_steps == (0, focus - 5)
        assert env.ranges.rate_hz == (10., 10.)
    env.close()
    fixed.close()


def test_easy_start_keeps_lap_rewards_and_full_horizon(monkeypatch):
    env = MixedLessonEnv(stage=4)
    fixed = GuidedLessonEnv(stage=4)
    monkeypatch.setattr(env, '_sample_practice_stage', lambda: 0)
    obs, _ = env.reset(seed=2)
    fixed.reset(seed=2)
    diameter = 2 * np.array([np.linalg.norm(env._hoop_h), np.linalg.norm(env._hoop_v)])
    expected = fixed.place(env.pos, env.heading, env.hoop_centre, env.hoop_normal, diameter)
    np.testing.assert_array_equal(obs, expected)
    passed = False
    for step in range(600):
        action = [0., .4, 0., 0.] if step < 32 else np.zeros(4)
        actual = env.step(action)
        expected = fixed.step(action)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1:4] == expected[1:4]
        assert actual[4]['outcome'] == expected[4]['outcome']
        assert actual[4]['passes'] == expected[4]['passes']
        assert actual[4]['stage'] == 0 and actual[4]['focus_stage'] == 4
        if actual[4]['outcome'] == 'passed':
            passed = True
            assert not actual[2] and not actual[3]
        assert not actual[2]
        assert actual[3] == (step == 599)
    assert passed and actual[4]['passes'] == 1
    assert actual[4]['outcome'] == 'timeout'
    env.close()
    fixed.close()


def test_focus_changes_only_at_reset(monkeypatch):
    env = MixedLessonEnv(stage=2)
    monkeypatch.setattr(env, '_sample_practice_stage', lambda: env.focus_stage)
    env.reset(seed=7)
    env.set_stage(8)
    info = env.step([0., 0., 0., 0.])[4]
    assert info['focus_stage'] == info['stage'] == 2
    assert not env.randomise
    _, info = env.reset(seed=7)
    assert info['focus_stage'] == info['stage'] == 8
    assert env.randomise and env.stage == 6
    env.close()


def test_only_frontier_start_outcomes_count_towards_promotion():
    curriculum = Curriculum()
    assert curriculum.stage == 4
    for _ in range(150):
        assert not curriculum.record({'stage': 2, 'focus_stage': 4,
                                      'outcome': 'timeout', 'passes': 3})
    assert not curriculum.recent
    for _ in range(99):
        assert not curriculum.record({'stage': 4, 'focus_stage': 4,
                                      'outcome': 'timeout', 'passes': 2})
    assert curriculum.record({'stage': 4, 'focus_stage': 4,
                              'outcome': 'timeout', 'passes': 2})
    assert curriculum.stage == 5 and not curriculum.recent
    assert Curriculum(curriculum.state()).state() == curriculum.state()


@pytest.mark.parametrize('stage', [-1, 9, 1.5])
def test_invalid_focus_stage_fails(stage):
    with pytest.raises(ValueError):
        training_mix(stage)
    with pytest.raises(ValueError):
        MixedLessonEnv(stage=stage)
