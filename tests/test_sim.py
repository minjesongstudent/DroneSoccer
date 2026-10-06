import dataclasses

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src import contract
from src.params import Cage, Camera, Drone, Hoop, Randomise
from src.contract import GAMMA, LAPS_MAX_STEPS, R_CRASH, R_PASS
from src.sim import HOOP_FROM_CENTRE, SIDE_GAP, HoopEnv

# A roomy cage and an exact hoop, so the geometry tests are not about the real layout.
BIG = Cage(length=10.0, width=10.0, height=4.0)
EXACT = Hoop(diameter_tolerance=0.0, marker_max_angle_deg=70.0)  # tests pin these, not the measured hoop
IDEAL = Drone(tau=0.3, delay_steps=0)  # scripted paths assume no lag, not the measured drone
U, V, MAJOR, RATIO, SIN, COS, VISIBLE, MARKER = range(8)


def make_env(**kwargs):
    kwargs.setdefault("cage", BIG)
    kwargs.setdefault("hoop", EXACT)
    kwargs.setdefault("drone", IDEAL)
    kwargs.setdefault("camera", Camera(pitch_deg=0.0))  # level: the geometry tests assume it
    return HoopEnv(**kwargs)


def observe_hoop_ahead(distance, hoop_angle_deg=0.0):
    """Drone at (2, 5, 1.5) facing +x; hoop `distance` ahead of the camera."""
    env = make_env()
    cam_x = 2.0 + env.camera.offset_forward
    a = np.deg2rad(hoop_angle_deg)
    env.place([2.0, 5.0, 1.5], 0.0, [cam_x + distance, 5.0, 1.5], [np.cos(a), np.sin(a), 0.0])
    return env.observe()


def fly(env, action, max_steps=200):
    for _ in range(max_steps):
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["outcome"]
    return "running"


def first_event(env, action, max_steps=200):
    """The first outcome that is not "running": a pass, or how the episode ended."""
    for _ in range(max_steps):
        info = env.step(action)[4]
        if info["outcome"] != "running":
            return info["outcome"]
    return "running"


def newest(policy_input):
    """The latest camera frame in a policy input: what contract.reward was given."""
    n = contract.OBS_STACK * contract.OBS_DIM
    return policy_input[n - contract.OBS_DIM:n]


# --- observation geometry ---------------------------------------------------

def test_hoop_straight_ahead_is_centred_and_round():
    obs = observe_hoop_ahead(1.0)
    assert obs[VISIBLE] == 1
    assert abs(obs[U]) < 0.01 and abs(obs[V]) < 0.01
    assert obs[RATIO] == pytest.approx(1.0, abs=0.01)


def test_45_degree_hoop_is_squashed():
    obs = observe_hoop_ahead(1.2, hoop_angle_deg=45.0)
    assert obs[RATIO] == pytest.approx(np.cos(np.deg2rad(45)), abs=0.03)


def test_major_scales_with_inverse_distance():
    near, far = observe_hoop_ahead(0.8), observe_hoop_ahead(1.6)
    assert near[MAJOR] / far[MAJOR] == pytest.approx(2.0, rel=0.02)
    assert near[MAJOR] == pytest.approx(EXACT.outer_diameter / 0.8, rel=0.02)


def test_hoop_behind_is_not_visible():
    assert observe_hoop_ahead(-1.0)[VISIBLE] == 0


def test_lean_is_continuous_across_vertical():
    def ellipse(theta_deg):
        t = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        pts = np.stack([0.2 * np.cos(t), 0.1 * np.sin(t)], axis=1)
        a = np.deg2rad(theta_deg)
        rot = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        return contract.observe(pts @ rot.T, (0.5, 0.4))

    for a, b in [(89, 91), (179, 181), (-1, 1)]:
        # 2 degrees apart, so allow sin(4 degrees) * lean of change, but not a jump.
        assert np.allclose(ellipse(a)[SIN:VISIBLE], ellipse(b)[SIN:VISIBLE], atol=0.05)


# --- dynamics ---------------------------------------------------------------

def test_commands_are_body_frame():
    env = make_env()
    env.place([5.0, 5.0, 1.5], np.pi / 2, [5.0, 8.0, 1.5], [0.0, 1.0, 0.0])  # facing +y
    for _ in range(10):
        env.step([0.0, 1.0, 0.0, 0.0])  # forward
    assert env.pos[1] > 5.2 and abs(env.pos[0] - 5.0) < 0.01

    env.place([5.0, 5.0, 1.5], np.pi / 2, [5.0, 8.0, 1.5], [0.0, 1.0, 0.0])
    for _ in range(10):
        env.step([1.0, 0.0, 0.0, 0.0])  # right, which is +x when facing +y
    assert env.pos[0] > 5.2 and abs(env.pos[1] - 5.0) < 0.01


def test_yaw_clockwise_decreases_heading():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, [8.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    for _ in range(5):
        env.step([0.0, 0.0, 0.0, 1.0])
    assert env.heading < -0.05


def test_command_delay():
    env = make_env(drone=Drone(delay_steps=2))
    env.place([5.0, 5.0, 1.5], 0.0, [8.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    env.step([0.0, 1.0, 0.0, 0.0])
    env.step([0.0, 1.0, 0.0, 0.0])
    assert env.pos[0] == 5.0
    env.step([0.0, 1.0, 0.0, 0.0])
    assert env.pos[0] > 5.0


# --- outcomes ---------------------------------------------------------------

@pytest.mark.parametrize("lateral, vertical, expected", [
    (0.0, 0.0, "passed"),
    (0.0, 0.10, "passed"),   # the drone is flat, so it has more room vertically...
    (0.10, 0.0, "hit"),      # ...than sideways, where a prop guard clips the noodle
    ((EXACT.outer_diameter - EXACT.thickness) / 2, 0.0, "hit"),
])
def test_outcomes(lateral, vertical, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5 + vertical], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    assert first_event(env, [0.0, 1.0, 0.0, 0.0]) == expected


@pytest.mark.parametrize("lateral, expected", [
    # Box side at 0.085 m, noodle's inner edge at 0.16 m: contact from 0.075 m.
    (0.07, "passed"),
    (0.08, "hit"),   # a shallow graze the old point grid could miss
])
def test_side_graze(lateral, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    assert first_event(env, [0.0, 1.0, 0.0, 0.0]) == expected


@pytest.mark.parametrize("lateral, vertical, expected", [
    # 0.40 m wide, 0.50 m tall. A round hoop of the average size would score both the other way.
    (0.06, 0.0, "hit"),       # contact from 0.05 m sideways, not 0.075 m
    (0.0, 0.124, "passed"),   # contact from ~0.127 m up, not ~0.12 m
])
def test_squashed_hoop_is_scored_as_seen(lateral, vertical, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5 + vertical], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0],
              hoop_diameter=(0.40, 0.50))
    assert first_event(env, [0.0, 1.0, 0.0, 0.0]) == expected


def test_touching_the_net_ends_the_episode():
    env = make_env()
    for action, axis, expected in [([0.0, 1.0, 0.0, 0.0], 0, BIG.length - IDEAL.length / 2),
                                   ([0.0, -1.0, 0.0, 0.0], 0, IDEAL.length / 2),
                                   ([1.0, 0.0, 0.0, 0.0], 1, IDEAL.width / 2),
                                   ([-1.0, 0.0, 0.0, 0.0], 1, BIG.width - IDEAL.width / 2),
                                   ([0.0, 0.0, 1.0, 0.0], 2, BIG.height - IDEAL.height / 2)]:
        # The hoop sits out of the way, so only the net can stop the drone.
        env.place([5.0, 5.0, 1.5], 0.0, [5.0, 9.0, 3.5], [0.0, 1.0, 0.0])
        for _ in range(300):
            before = env._to_go(env.pos)
            policy_input, reward, terminated, _, info = env.step(action)
            if terminated:
                break
        assert info["outcome"] == "net" and info["wall_steps"] == 1
        # Scored as a crash and a wall touch, whatever the reward makes of them.
        args = before, env._to_go(env.pos), True, newest(policy_input), np.float32(action)
        assert reward == pytest.approx(contract.reward("net", *args))
        assert reward < contract.reward("running", *args)
        # The drone's box is at the net, not its centre.
        assert env.pos[axis] == pytest.approx(expected)


def test_the_floor_holds_the_drone_without_ending_the_episode():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, [5.0, 9.0, 3.5], [0.0, 1.0, 0.0])
    down = [0.0, 0.0, -1.0, 0.0]
    for _ in range(300):
        before = env._to_go(env.pos)
        policy_input, reward, terminated, _, info = env.step(down)
        assert not terminated
    assert env.pos[2] == pytest.approx(BIG.min_altitude)
    # Held there, it is scored as a wall touch that costs something, not a crash.
    assert info["outcome"] == "running"
    after, obs = env._to_go(env.pos), newest(policy_input)
    assert reward == pytest.approx(contract.reward("running", before, after, True, obs, np.float32(down)))
    assert reward < contract.reward("running", before, after, False, obs, np.float32(down))


def test_hit_shortcut_agrees_with_the_full_check():
    """_hit skips the centreline check when the drone is far away. It must never
    skip a real contact, so compare it with the full check around the hoop."""
    env = make_env()
    env.place([2.0, 5.0, 1.5], 0.3, [2.5, 5.0, 1.5], [1.0, 0.0, 0.0])
    rng = np.random.default_rng(0)
    hits = 0
    for _ in range(5000):
        env.pos = env.hoop_centre + rng.uniform(-1, 1, 3) * env._hit_reach * 1.2
        env.heading = rng.uniform(-np.pi, np.pi)
        full = bool((env._box_distance(env._centreline) < env.hoop.thickness / 2).any())
        assert env._hit() == full
        hits += full
    assert hits > 0


# --- laps -------------------------------------------------------------------
# Hoop at (6, 5, 1.5) with its back facing +x, so the front (and the marker)
# faces a drone coming from x < 6.

HOOP_AT, BACK = [6.0, 5.0, 1.5], [1.0, 0.0, 0.0]


def goto(env, x, y, z=1.5, max_steps=300):
    """Fly to (x, y) facing +x with a proportional controller. Returns the last step."""
    for _ in range(max_steps):
        dx, dy, dz = x - env.pos[0], y - env.pos[1], z - env.pos[2]
        step = env.step(np.clip([-3 * dy, 3 * dx, 3 * dz, 0.0], -1, 1))
        if step[2] or step[3] or np.hypot(dx, dy) < 0.02:
            return step
    return step


def test_lap_pass_scores_and_keeps_flying():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step([0.0, 1.0, 0.0, 0.0])
        if info["outcome"] == "passed":
            break
    assert info["outcome"] == "passed" and reward > 10.0
    assert not terminated and not truncated and info["passes"] == 1


def test_a_crash_costs_less_than_a_pass_pays():
    """Otherwise the safest policy is to hover out of reach and never try."""
    assert R_CRASH < R_PASS
    env = make_env()
    env.place([5.0, 5.0 + 0.3, 1.5], 0.0, HOOP_AT, BACK)   # aimed at the rim
    total = 0.0
    for _ in range(100):
        _, reward, terminated, _, info = env.step([0.0, 1.0, 0.0, 0.0])
        total += reward
        if terminated:
            break
    assert info["outcome"] == "hit"
    assert total < 0.0 and -total < R_PASS


def test_crashing_is_never_the_cheap_way_out_of_an_episode():
    """The drone that flies at the rim must lose to the one that waits, or the
    per-step costs teach it to crash. Timeouts bootstrap and the policy cannot see
    the clock, so it values hovering as endless: its step reward / (1 - GAMMA)."""
    assert R_CRASH < R_PASS
    env = make_env()
    env.place([5.0, 5.0 + 0.3, 1.5], 0.0, HOOP_AT, BACK)   # aimed at the rim
    hovering = env.step([0.0, 0.0, 0.0, 0.0])[1] / (1 - GAMMA)

    env.place([5.0, 5.0 + 0.3, 1.5], 0.0, HOOP_AT, BACK)
    crashing, terminated = 0.0, False
    for k in range(LAPS_MAX_STEPS):
        _, reward, terminated, _, info = env.step([0.0, 1.0, 0.0, 0.0])
        crashing += GAMMA ** k * reward
        if terminated:
            break
    assert terminated and info["outcome"] == "hit"
    assert crashing < hovering


def test_holding_still_never_pays():
    """Anywhere in the real cage, a step of hovering must cost something, or the
    drone earns by waiting. The shaping pays (1 - GAMMA) * R_PROGRESS * to_go a step
    for holding still, so the per-step costs must stay above it."""
    env = HoopEnv()
    rng = np.random.default_rng(0)
    cage = env.cage
    for seed in range(100):
        env.reset(seed=seed)
        spots = rng.uniform([0.1, 0.1, cage.min_altitude], [cage.length - 0.1, cage.width - 0.1,
                                                            cage.height - 0.1], (300, 3))
        front = [p for p in spots if (p - env.hoop_centre) @ env.hoop_normal < 0.0]
        # The furthest spot earns the most for holding still. In front, facing the
        # hoop, the marker can be in view too, so nothing else is charged there.
        for pos in (max(spots, key=env._to_go), max(front, key=env._to_go), *spots[:5]):
            to_hoop = env.hoop_centre - pos
            for heading in (np.arctan2(to_hoop[1], to_hoop[0]), rng.uniform(-np.pi, np.pi)):
                env.pos, env.heading = pos.copy(), heading
                assert env.step([0.0, 0.0, 0.0, 0.0])[1] < 0.0


def test_crossing_outside_the_ring_never_raises_the_distance_to_go():
    """Only a pass may cost distance to go, or going round would be punished."""
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for y in np.linspace(5.25, 6.2, 12):
        for z in (1.0, 1.5):
            behind, front = np.array([6.001, y, z]), np.array([5.999, y, z])
            assert env._to_go(front) <= env._to_go(behind) + 0.01

def test_back_through_the_ring_is_wrong_way():
    env = make_env()
    env.place([7.0, 5.0, 1.5], np.pi, HOOP_AT, BACK)  # behind the hoop, facing it
    assert fly(env, [0.0, 1.0, 0.0, 0.0]) == "wrong_way"


def test_crossing_outside_the_ring_is_neutral():
    env = make_env()
    env.place([7.0, 5.8, 1.5], np.pi, HOOP_AT, BACK)
    for _ in range(40):  # 1.6 m at 0.4 m/s: well past the plane
        _, _, terminated, truncated, info = env.step([0.0, 1.0, 0.0, 0.0])
        assert not terminated and not truncated
    assert env.pos[0] < 6.0 and info["passes"] == 0


def test_a_full_lap_scores_twice():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for x, y in [(6.6, 5.0), (6.6, 5.9), (5.4, 5.9), (5.4, 5.0), (6.6, 5.0)]:
        _, _, terminated, truncated, info = goto(env, x, y)
        assert not terminated and not truncated, info["outcome"]
    assert info["passes"] == 2


def test_route_records_the_way_round():
    """Back to front outside the ring, at +y: the hoop's left, seen from the front."""
    env = make_env()
    env.place([6.6, 5.9, 1.5], 0.0, HOOP_AT, BACK)
    routes = [env.step([0.0, -1.0, 0.0, 0.0])[4]["route"] for _ in range(20)]
    assert [r for r in routes if r] == ["left"]


@pytest.mark.parametrize("y, z, expected", [
    (5.0, 2.0, "hit"),       # straight over the top: into the hanger
    (5.0, 1.0, "under"),     # below the hoop: open
    (5.4, 2.0, "over"),      # over, but clear of the strip's side
])
def test_hanger_blocks_going_over_the_top(y, z, expected):
    env = make_env()
    env.place([6.6, y, z], 0.0, HOOP_AT, BACK)   # behind the hoop, reversing towards the front
    for _ in range(20):
        _, _, terminated, _, info = env.step([0.0, -1.0, 0.0, 0.0])
        if terminated or info["route"]:
            break
    assert (info["outcome"] if terminated else info["route"]) == expected


def _downhill(env, pos, e=1e-4):
    return -np.array([env._to_go(pos + e * d) - env._to_go(pos - e * d) for d in np.eye(3)])


def test_the_shaping_leads_away_from_the_rim():
    """Straight at the centre from the side, or at a gate from just behind the
    ring, runs into the noodle. The route must lead out first."""
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    # In front, beside the rim: out front (-x), not sideways into it.
    assert _downhill(env, np.array([5.97, 5.5, 1.5]))[0] < 0.0
    # Just through the ring: backing off (+x) as well as sideways.
    step = _downhill(env, np.array([6.05, 5.05, 1.5]))
    assert step[0] > 0.0 and step[1] > 0.0
    # Well beside the rim behind it: forwards to the gate, to come round.
    assert _downhill(env, np.array([6.3, 5.6, 1.5]))[0] < 0.0


def test_following_the_shaping_flies_clean_laps():
    """An oracle that only descends the distance to go laps without a crash."""
    env = make_env()
    env.place([5.0, 5.3, 1.3], 0.0, HOOP_AT, BACK)
    for _ in range(LAPS_MAX_STEPS):
        v = _downhill(env, env.pos)
        v /= max(np.linalg.norm(v), 1e-9)
        _, _, terminated, _, info = env.step([-v[1], v[0], v[2], 0.0])
        assert not terminated, info["outcome"]
    assert info["passes"] >= 8


@pytest.mark.parametrize("angle_deg, expected", [(0, 1), (60, 1), (75, 0), (180, 0)])
def test_marker_is_seen_only_from_the_front(angle_deg, expected):
    """Drone 1 m from the hoop at angle_deg off its front normal, facing the hoop."""
    env = make_env()
    a = np.deg2rad(angle_deg)
    drone = np.array(HOOP_AT) + [-np.cos(a), np.sin(a), 0.0]
    env.place(drone, np.arctan2(-np.sin(a), np.cos(a)), HOOP_AT, BACK)
    obs = env.observe()
    assert obs[VISIBLE] == 1   # the hoop itself is in view every time
    assert obs[MARKER] == expected


def test_marker_needs_the_hoop():
    assert contract.observe(np.zeros((2, 2)), (0.5, 0.4), marker=True)[MARKER] == 0


def test_laps_run_to_the_cut_off():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for _ in range(LAPS_MAX_STEPS):
        _, _, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
        assert not terminated
    assert truncated and info["outcome"] == "timeout"


def test_pad_layouts_fit_the_real_cage():
    env = HoopEnv(camera=Camera())
    for seed in range(300):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        assert env._in_fence(env.pos)
        assert (env.pos - env.hoop_centre) @ env.hoop_normal < 0   # starts in front
        assert len(env._gates) == 2                                 # room to go round either side
        middle = np.array([env.cage.length, env.cage.width]) / 2
        assert np.linalg.norm(env.hoop_centre[:2] - middle) <= HOOP_FROM_CENTRE + 1e-9
        # Level 1 is the real start: take-off height, above a pad in a corner.
        assert env.pos[2] == pytest.approx(env.drone.takeoff_height)
        inset = env.pad.inset
        assert {round(min(c, env.cage.length - c), 6) for c in env.pos[:2]} == {inset}


def test_level_zero_starts_with_hoop_and_marker_in_view():
    env = HoopEnv(level=0.0, camera=Camera())
    seen = []
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        seen.append(obs[(contract.OBS_STACK - 1) * contract.OBS_DIM + MARKER])
    assert np.mean(seen) > 0.95


def test_randomised_dynamics_stay_in_range_and_fixed_per_episode():
    env = HoopEnv(randomise=True, camera=Camera())
    r = Randomise()
    delays = set()
    for seed in range(50):
        env.reset(seed=seed)
        dyn = env.dyn
        assert r.delay_steps[0] <= dyn.delay_steps <= r.delay_steps[1]
        assert r.tau[0] <= dyn.tau <= r.tau[1]
        delays.add(dyn.delay_steps)
        env.step([0.0, 0.5, 0.0, 0.0])
        assert env.dyn is dyn
    assert delays == set(range(r.delay_steps[0], r.delay_steps[1] + 1))



def test_measurement_error_stays_in_range_and_leaves_the_measured_values():
    env = HoopEnv(randomise=True, camera=Camera())
    measured = {name: dataclasses.replace(obj) for name, obj in env._measured.items()}
    fx = set()
    for seed in range(30):
        env.reset(seed=seed)
        for key, error in Randomise().error.items():
            name, attr = key.split(".")
            assert abs(getattr(getattr(env, name), attr) - getattr(measured[name], attr)) <= error
        fx.add(env.camera.fx)
    assert len(fx) == 30
    assert env._measured["camera"] == measured["camera"]


def test_hanger_bar_is_drawn_per_episode():
    env = HoopEnv(camera=Camera())
    low, high = Hoop().hanger_half_width_range
    widths = []
    for seed in range(30):
        env.reset(seed=seed)
        widths.append(env.hanger_half_width)
    assert low <= min(widths) and max(widths) <= high and len(set(widths)) == 30

def test_gymnasium_api():
    check_env(HoopEnv(randomise=True, camera=Camera()), skip_render_check=True)


def test_hanger_can_be_left_out():
    env = make_env(hanger=False)
    env.place([6.6, 5.0, 2.0], 0.0, HOOP_AT, BACK)
    routes = [env.step([0.0, -1.0, 0.0, 0.0])[4]["route"] for _ in range(20)]
    assert [r for r in routes if r] == ["over"]


def test_seen_hoop_flag_reads_1_until_the_first_sighting_then_0_for_good():
    """The policy input's last number: blind at takeoff must not look like blind mid-lap."""
    env = make_env()
    x = env.place([5.0, 5.0, 1.5], np.pi, HOOP_AT, BACK)   # facing away from the hoop
    assert x[-1] == 1.0 and not newest(x)[VISIBLE]
    seen = []
    for _ in range(200):   # turn on the spot: the hoop comes into view, then leaves it
        x = env.step([0.0, 0.0, 0.0, 1.0])[0]
        seen.append(bool(newest(x)[VISIBLE]))
        if seen[-1]:
            assert x[-1] == 0.0
    assert any(seen) and not seen[-1]
    assert x[-1] == 0.0
