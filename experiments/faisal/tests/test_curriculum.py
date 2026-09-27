import numpy as np
import pytest

from src.curriculum import Curriculum, LessonEnv, R_LESSON_TIMEOUT, WINDOW
from src.live_evaluation import evaluate
from src.sim import CRASHES, HoopEnv, R_CRASH, R_PASS


def test_centered_lesson_is_solvable_and_requires_full_body_clearance():
    env = LessonEnv()
    env.reset(seed=2)
    crossing_seen = False
    for _ in range(100):
        _, _, done, truncated, info = env.step([0, 1, 0, 0])
        axial = (env.pos-env.hoop_centre) @ env.hoop_normal
        if 0 < axial < .10:
            crossing_seen = True
            assert info['passes'] == 0 and not done
        if done or truncated:
            break
    assert crossing_seen
    assert done and info['outcome'] == 'passed' and info['passes'] == 1
    assert axial > env.drone.length/2 + env.hoop.thickness/2


@pytest.mark.parametrize('stage', [0, 1])
def test_single_pass_forward_progress_stays_positive_through_clearance(stage):
    env = LessonEnv(stage=stage)
    env.reset(seed=2)
    # Identical aligned path isolates reward shaping from the harder stage's
    # initial lateral/yaw perturbations and includes the blind final approach.
    env.place([.65, 1., 1.], 0., [1.2, 1., 1.], [1., 0., 0.])
    crossed = False
    for _ in range(100):
        _, reward, done, truncated, info = env.step([0, .5, 0, 0])
        assert reward > 0
        assert not truncated
        if env.pos[0] > env.hoop_centre[0] and not done:
            crossed = True
            assert info['passes'] == 0
        if done:
            break
    assert crossed
    assert done and info['outcome'] == 'passed'
    assert reward > R_PASS


@pytest.mark.parametrize('stage', [0, 1])
def test_single_pass_target_is_static_and_beyond_any_yaw_clearance(stage):
    env = LessonEnv(stage=stage)
    env.reset(seed=2)
    target = env._progress_target(env.pos)
    distance = (target-env.hoop_centre) @ env.hoop_normal
    for heading in np.linspace(-np.pi, np.pi, 33):
        env.heading = heading
        forward = np.array([np.cos(heading), np.sin(heading), 0.])
        side = np.array([-forward[1], forward[0], 0.])
        clearance = (env.drone.length/2*abs(forward @ env.hoop_normal)
                     + env.drone.width/2*abs(side @ env.hoop_normal)
                     + env.hoop.thickness/2 + .01)
        assert distance > clearance
        for axial in [-.1, .1]:
            np.testing.assert_allclose(
                env._progress_target(env.hoop_centre + axial*env.hoop_normal), target)


@pytest.mark.parametrize('stage', [0, 1])
def test_single_pass_timeout_cost_is_smaller_than_unchanged_crash_cost(stage):
    env = LessonEnv(stage=stage)
    env.reset(seed=2)
    env.place([.65, 1., 1.], 0., [1.2, 1., 1.], [1., 0., 0.])
    total = 0.
    for step in range(150):
        _, reward, done, truncated, info = env.step(np.zeros(4))
        total += reward
        assert not done
        assert truncated == (step == 149)
    assert info['outcome'] == 'timeout'
    assert total == pytest.approx(-R_LESSON_TIMEOUT)
    assert 0 < R_LESSON_TIMEOUT < R_CRASH

    # A stationary box intersecting the rim still incurs the base crash cost.
    env.reset(seed=2)
    env.place([1.2, 1.225, 1.], 0., [1.2, 1., 1.], [1., 0., 0.])
    _, reward, done, truncated, info = env.step(np.zeros(4))
    assert done and not truncated and info['outcome'] in CRASHES
    assert reward <= -R_CRASH


@pytest.mark.parametrize('stage', range(2, 7))
def test_lap_lesson_rewards_and_horizon_match_base_environment(stage):
    env = LessonEnv(stage=stage)
    env.reset(seed=2)
    base = HoopEnv()
    base.reset(seed=2)
    # Exact placement and nominal dynamics let us compare the whole transition
    # contract, including pass reward at crossing and the 600-step timeout.
    env.randomise = False
    layout = ([.65, 1., 1.], 0., [1.2, 1., 1.], [1., 0., 0.])
    env.place(*layout)
    base.place(*layout)
    for step in range(600):
        action = [0, .5, 0, 0] if step < 40 else np.zeros(4)
        obs, reward, done, truncated, info = env.step(action)
        expected_obs, expected_reward, expected_done, expected_truncated, expected_info = base.step(action)
        np.testing.assert_array_equal(obs, expected_obs)
        assert reward == expected_reward
        assert (done, truncated) == (expected_done, expected_truncated)
        assert info['outcome'] == expected_info['outcome']
        assert info['passes'] == expected_info['passes']
        if done or truncated:
            break
    assert step == 599 and truncated and not done
    assert info['outcome'] == 'timeout' and info['passes'] == 1


def test_lesson_change_waits_for_reset_and_delays_increase():
    env = LessonEnv()
    env.reset(seed=0)
    env.set_stage(6)
    assert env.stage == 0 and not env.randomise
    env.reset(seed=0)
    assert env.stage == 6 and env.randomise
    assert env.ranges.delay_steps == (0, 3)
    assert env.ranges.rate_hz == (10, 10)


def test_curriculum_requires_complete_window_and_laps_after_passes():
    lesson = Curriculum()
    for _ in range(WINDOW-1):
        assert not lesson.record({'stage': 0, 'outcome': 'passed', 'passes': 1})
    assert lesson.record({'stage': 0, 'outcome': 'passed', 'passes': 1})
    assert lesson.stage == 1 and not lesson.recent
    lesson = Curriculum({'stage': 2})
    for _ in range(WINDOW):
        assert not lesson.record({'stage': 2, 'outcome': 'hit', 'passes': 3})
    assert lesson.stage == 2
    for _ in range(WINDOW):
        lesson.record({'stage': 2, 'outcome': 'timeout', 'passes': 2})
    assert lesson.stage == 3
    assert Curriculum(lesson.state()).state() == lesson.state()


@pytest.mark.parametrize('stage', range(7))
def test_lesson_layouts_start_valid(stage):
    env = LessonEnv(stage=stage)
    for seed in range(30):
        obs, _ = env.reset(seed=seed)
        assert obs.shape == (84,) and np.isfinite(obs).all()
        assert not env._hit() and not env._left_cage() and not env._hits_hanger()


def test_evaluation_does_not_count_easy_passes_as_full_task_success():
    class StraightAhead:
        num_timesteps = 42
        def predict(self, obs, deterministic):
            return np.array([0, 1, 0, 0]), None
    a = evaluate(StraightAhead(), 0, episodes=2)
    b = evaluate(StraightAhead(), 0, episodes=2)
    assert a == b
    assert a['suites']['lesson']['success_rate'] == 1
    assert a['suites']['full_task']['success_rate'] == 0
