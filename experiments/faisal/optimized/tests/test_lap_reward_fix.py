"""Behavioural regression checks for clearance and target-switch reward farming."""
import numpy as np
import pytest

from src.sim import HoopEnv, R_PASS, R_HIDDEN, PROGRESS_UNIT
from src.params import Cage, Drone


def placed(**kwargs):
    env = HoopEnv(**kwargs)
    env.place([.65, 1., 1.], 0., [1.2, 1., 1.], [1., 0., 0.])
    return env


@pytest.mark.parametrize('heading', [0., .25, -.25])
def test_laps_pay_once_only_after_whole_body_clears(heading):
    env = placed()
    env.heading = heading
    initial_target = env._progress_target(env.pos).copy()
    crossed = False
    bonuses = 0
    for _ in range(110):
        # World +x at constant heading; this is a geometry test, not a teacher.
        _, reward, done, truncated, info = env.step([.5*np.sin(heading), .5*np.cos(heading), 0, 0])
        axial = (env.pos-env.hoop_centre) @ env.hoop_normal
        assert not done and not truncated
        if 0 < axial <= env._clearance_reach():
            crossed = True
            assert info['passes'] == 0
            np.testing.assert_allclose(env._progress_target(env.pos), initial_target)
            assert reward > 0
        if info['outcome'] == 'passed':
            assert axial > env._clearance_reach()
            bonuses += 1
        if axial > .35:
            break
    assert crossed and bonuses == 1 and info['passes'] == 1


def test_forward_clearance_reward_beats_sideways_at_diagnostic_position():
    env = placed()
    pos = np.array([1.22, 1., 1.])
    target = env._progress_target(pos)
    forward = (np.linalg.norm(pos-target)-np.linalg.norm(pos+[.01, 0, 0]-target))/PROGRESS_UNIT
    sideways = (np.linalg.norm(pos-target)-np.linalg.norm(pos+[0, .01, 0]-target))/PROGRESS_UNIT
    assert forward > 0 and sideways < 0


def test_original_zero_pass_loop_no_longer_earns_reward():
    env = placed()
    centre = env.hoop_centre.copy()
    corners = [centre+np.array([x, y, 0.]) for x, y in
               [(.03, .45), (.03, .60), (-.03, .60), (-.03, .45)]]
    env.place(corners[0], 0., centre, [1., 0., 0.])
    initial_distance = env._progress_distance(env.pos)
    goal, reached, total, shaping = 1, 0, 0., 0.
    for _ in range(600):
        delta = corners[goal]-env.pos
        if np.linalg.norm(delta) < .012:
            goal = (goal+1)%4
            reached += 1
            delta = corners[goal]-env.pos
        velocity = delta*2
        velocity *= min(1., .15/max(np.linalg.norm(velocity), 1e-9))
        _, reward, done, truncated, info = env.step(
            np.array([-velocity[1], velocity[0], velocity[2], 0.])/env.drone.max_speed)
        total += reward
        shaping += reward+R_HIDDEN*(1-env.observe()[6])
        assert not done and info['passes'] == 0
    final_distance = env._progress_distance(env.pos)
    assert reached//4 >= 13 and truncated
    assert shaping == pytest.approx((initial_distance-final_distance)/PROGRESS_UNIT, abs=1e-5)
    assert total < 0


def test_shaping_accounts_for_every_phase_switch_over_a_full_lap():
    env = HoopEnv(cage=Cage(length=10., width=10., height=4.))
    env.place([5., 5., 1.5], 0., [6., 5., 1.5], [1., 0., 0.])
    initial_distance = env._progress_distance(env.pos)
    shaping = 0.
    phases = {env._lap_phase}
    last_passes = 0
    # Follow the reward waypoints only to exercise the transition accounting.
    for _ in range(600):
        delta = env._progress_target(env.pos)-env.pos
        action = np.clip([-3*delta[1], 3*delta[0], 3*delta[2], 0.], -1, 1)
        _, reward, done, truncated, info = env.step(action)
        assert not done, info
        shaping += reward + R_HIDDEN*(1-env.observe()[6])-R_PASS*(info['passes']-last_passes)
        last_passes = info['passes']
        phases.add(env._lap_phase)
        if info['passes'] == 2:
            break
    assert info['passes'] == 2
    assert phases == {'approach', 'return_side', 'return_front', 'align_front'}
    final_distance = env._progress_distance(env.pos)
    assert shaping == pytest.approx((initial_distance-final_distance)/PROGRESS_UNIT, abs=1e-5)


def test_partial_crossing_then_collision_never_pays():
    env = placed()
    while env.pos[0] <= 1.21:
        info = env.step([0, .5, 0, 0])[4]
    assert info['passes'] == 0
    for _ in range(50):
        _, reward, done, _, info = env.step([-1, 0, 0, 0])
        assert info['passes'] == 0
        if done:
            assert info['outcome'] == 'hit' and reward < 0
            break
    assert done


def test_reset_clears_partial_pass_and_phase():
    env = placed()
    while env.pos[0] <= 1.21:
        env.step([0, .5, 0, 0])
    assert env._crossed_pending
    env.reset(seed=10)
    assert not env._crossed_pending and env.passes == 0 and env._lap_phase == 'approach'
