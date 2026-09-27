"""Return milestones must encourage completion without renewable waypoint rewards."""
import numpy as np
import pytest

from src.curriculum import LessonEnv, REWARD_VERSION
from src.sim import HoopEnv, PROGRESS_UNIT, R_HIDDEN, R_PASS


def placed(diameter=.45, heading=0.):
    env = HoopEnv()
    env.place([.50, 1., 1.], heading, [1.05, 1., 1.], [1., 0., 0.], diameter)
    return env


def action_for(env, world_velocity):
    forward = np.array([np.cos(env.heading), np.sin(env.heading), 0.])
    right = np.array([forward[1], -forward[0], 0.])
    return np.array([world_velocity @ right, world_velocity @ forward,
                     world_velocity[2], 0.]) / env.drone.max_speed


@pytest.mark.parametrize('phase,next_phase', [
    ('return_side', 'return_front'), ('return_front', 'align_front'),
    ('align_front', 'approach')])
@pytest.mark.parametrize('sign', [-1, 1])
def test_arriving_at_each_return_milestone_rewards_progress(phase, next_phase, sign):
    env = placed()
    gate = env.hoop_centre + sign*.575*env._axis_h
    env._lap_phase = phase
    env._return_gate = gate.copy() if phase != 'return_side' else None
    offset = env._clearance_target_offset()*env.hoop_normal
    if phase == 'return_side':
        target = gate+offset
        env.pos = target-sign*.052*env._axis_h
        velocity = sign*env.drone.max_speed*env._axis_h
    elif phase == 'return_front':
        target = gate-offset
        env.pos = target+.052*env.hoop_normal
        velocity = -env.drone.max_speed*env.hoop_normal
    else:
        target = env.hoop_centre-offset
        env.pos = target+sign*.052*env._axis_h
        velocity = -sign*env.drone.max_speed*env._axis_h
    before = env._progress_distance(env.pos)
    _, reward, terminated, truncated, info = env.step(action_for(env, velocity))
    assert not terminated and not truncated and info['passes'] == 0
    assert env._lap_phase == next_phase
    assert np.linalg.norm(env.pos-target) < .05
    assert reward > 0  # Previously approximately -5 despite reaching the waypoint.
    shaping = reward+R_HIDDEN*(1-env.observe()[6])
    assert shaping == pytest.approx((before-env._progress_distance(env.pos))/PROGRESS_UNIT)


@pytest.mark.parametrize('sign', [-1, 1])
def test_remaining_route_is_continuous_at_exact_milestones(sign):
    env = placed()
    gate = env.hoop_centre+sign*.575*env._axis_h
    offset = env._clearance_target_offset()*env.hoop_normal
    transitions = [('return_side', 'return_front', gate+offset),
                   ('return_front', 'align_front', gate-offset),
                   ('align_front', 'approach', env.hoop_centre-offset)]
    for phase, next_phase, point in transitions:
        env._lap_phase = phase
        env._return_gate = None if phase == 'return_side' else gate
        before = env._progress_distance(point)
        env._lap_phase = next_phase
        env._return_gate = gate
        assert env._progress_distance(point) == pytest.approx(before, abs=1e-12)


@pytest.mark.parametrize('sign', [-1, 1])
def test_either_feasible_side_can_be_chosen_then_latches_only_on_arrival(sign):
    env = placed()
    env._lap_phase = 'return_side'
    offset = env._clearance_target_offset()*env.hoop_normal
    chosen = env.hoop_centre+sign*.575*env._axis_h
    opposite = env.hoop_centre-sign*.575*env._axis_h
    # Looking at or moving toward the other side does not commit the episode.
    env.pos = opposite+offset-sign*.08*env._axis_h
    np.testing.assert_allclose(env._progress_target(env.pos), opposite+offset)
    assert env._return_gate is None
    env.pos = chosen+offset-sign*.052*env._axis_h
    np.testing.assert_allclose(env._progress_target(env.pos), chosen+offset)
    assert env._return_gate is None
    _, _, terminated, _, info = env.step(action_for(env, sign*.4*env._axis_h))
    assert not terminated and info['lap_phase'] == 'return_front'
    np.testing.assert_allclose(env._return_gate, chosen)
    np.testing.assert_allclose(env._progress_target(opposite+offset), chosen-offset)
    # A new episode has no stale side choice.
    env.reset(seed=3)
    assert env._return_gate is None


def test_only_a_feasible_gate_is_offered():
    env = HoopEnv()
    env.place([.5, .6, 1.], 0., [1.05, .6, 1.], [1., 0., 0.])
    assert len(env._gates) == 1  # Negative-y side lies outside the fence.
    env._lap_phase = 'return_side'
    blocked_side = env.hoop_centre-.575*env._axis_h
    expected = env._gates[0]+env._clearance_target_offset()*env.hoop_normal
    np.testing.assert_allclose(env._progress_target(blocked_side), expected)


def test_gate_choice_and_route_distance_cannot_be_changed_by_rotating_in_place():
    env = placed()
    env._lap_phase = 'return_side'
    env.pos = env.hoop_centre+np.array([.3, .15, 0.])
    target, distance = env._progress_target(env.pos).copy(), env._progress_distance(env.pos)
    for heading in np.linspace(-np.pi, np.pi, 25):
        env.heading = heading
        np.testing.assert_allclose(env._progress_target(env.pos), target)
        assert env._progress_distance(env.pos) == pytest.approx(distance)
    assert env._return_gate is None


@pytest.mark.parametrize('diameter', [(.42, .42), (.42, .48), (.48, .42), (.48, .48)])
@pytest.mark.parametrize('heading', [0., -.4, .4, np.pi/4])
def test_default_geometry_pass_bonus_exceeds_route_reset_cost(diameter, heading):
    env = placed(diameter, heading)
    action = action_for(env, np.array([.2, 0., 0.]))
    for _ in range(70):
        before = env._progress_distance(env.pos)
        _, reward, terminated, truncated, info = env.step(action)
        assert not terminated and not truncated
        if info['outcome'] == 'passed':
            assert info['passes'] == 1 and env._lap_phase == 'return_side'
            assert env._return_gate is None
            assert reward > 0
            assert (env.pos-env.hoop_centre) @ env.hoop_normal > env._clearance_reach()
            assert reward == pytest.approx(
                (before-env._progress_distance(env.pos))/PROGRESS_UNIT
                + R_PASS-R_HIDDEN*(1-env.observe()[6]))
            break
    else:
        pytest.fail('Expected one fully cleared pass')


def test_repeated_zero_pass_side_switching_does_not_farm_route_progress():
    env = placed()
    centre = env.hoop_centre.copy()
    corners = [centre+np.array([x, y, 0.]) for x, y in
               [(.29, -.20), (.40, -.20), (.40, .20), (.29, .20)]]
    env.place(corners[0], 0., centre, [1., 0., 0.])
    assert env._lap_phase == 'return_side' and env._return_gate is None
    initial = env._progress_distance(env.pos)
    goal, reached, reward_sum, shaping = 1, 0, 0., 0.
    sides = set()
    closed_circuit_shaping = []
    for _ in range(600):
        delta = corners[goal]-env.pos
        if np.linalg.norm(delta) < .012:
            if goal == 0:
                closed_circuit_shaping.append(shaping)
            goal = (goal+1) % len(corners)
            reached += 1
            delta = corners[goal]-env.pos
        velocity = delta*2
        velocity *= min(1., .2/max(np.linalg.norm(velocity), 1e-9))
        _, reward, terminated, truncated, info = env.step(action_for(env, velocity))
        assert not terminated and info['passes'] == 0
        assert env._lap_phase == 'return_side' and env._return_gate is None
        sides.add(int(np.sign((env._progress_target(env.pos)-centre) @ env._axis_h)))
        reward_sum += reward
        shaping += reward+R_HIDDEN*(1-float(env.observe()[6]))
    assert truncated and reached//4 >= 4
    assert sides == {-1, 1}
    assert shaping == pytest.approx((initial-env._progress_distance(env.pos))/PROGRESS_UNIT, abs=1e-5)
    assert len(closed_circuit_shaping) >= 4
    # Every completed circuit returns within 12 mm of its initial position.
    # Cumulative shaping remains bounded by that endpoint tolerance, regardless
    # of the number of circuits; it cannot pay repeatedly for changing sides.
    assert max(abs(value) for value in closed_circuit_shaping) <= .12+1e-5
    assert reward_sum < 0


def test_single_pass_reward_does_not_charge_for_an_unrequested_return():
    env = LessonEnv(stage=0)
    env.reset(seed=8)
    direction = env.hoop_normal.copy()
    for _ in range(80):
        before = env._progress_distance(env.pos)
        _, reward, terminated, truncated, info = env.step(action_for(env, .2*direction))
        assert not truncated
        if terminated:
            assert info['outcome'] == 'passed'
            assert reward > R_PASS
            assert env._progress_distance(env.pos) == pytest.approx(
                np.linalg.norm(env.pos-env._progress_target(env.pos)))
            assert reward == pytest.approx((before-env._progress_distance(env.pos))/PROGRESS_UNIT
                                           +R_PASS-R_HIDDEN*(1-env.observe()[6]))
            break
    else:
        pytest.fail('Expected the single-pass task to finish')
    assert REWARD_VERSION == 'lap-remaining-route-potential-v3'
