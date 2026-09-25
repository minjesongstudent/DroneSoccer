import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src import contract
from src.params import Cage, Camera, Drone, Hoop, Randomise
from src.sim import LAPS_MAX_STEPS, SIDE_GAP, HoopEnv

# A roomy cage and an exact hoop, so the geometry tests are not about the real layout.
BIG = Cage(length=10.0, width=10.0, height=4.0)
EXACT = Hoop(diameter_tolerance=0.0)
U, V, MAJOR, RATIO, SIN, COS, VISIBLE, MARKER = range(8)


def make_env(**kwargs):
    kwargs.setdefault("cage", BIG)
    kwargs.setdefault("hoop", EXACT)
    kwargs.setdefault("camera", Camera())
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


def test_yaw_clockwise_decreases_heading_and_is_masked_by_default():
    env = make_env(yaw_enabled=True)
    env.place([5.0, 5.0, 1.5], 0.0, [8.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    for _ in range(5):
        env.step([0.0, 0.0, 0.0, 1.0])
    assert env.heading < -0.05

    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, [8.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    for _ in range(5):
        env.step([0.0, 0.0, 0.0, 1.0])
    assert env.heading == 0.0


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
    (0.6, 0.0, "missed"),
])
def test_outcomes(lateral, vertical, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5 + vertical], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    assert fly(env, [0.0, 1.0, 0.0, 0.0]) == expected


@pytest.mark.parametrize("lateral, expected", [
    # Box side at 0.085 m, noodle's inner edge at 0.16 m: contact from 0.075 m.
    (0.07, "passed"),
    (0.08, "hit"),   # a shallow graze the old point grid could miss
])
def test_side_graze(lateral, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    assert fly(env, [0.0, 1.0, 0.0, 0.0]) == expected


@pytest.mark.parametrize("lateral, vertical, expected", [
    # 0.40 m wide, 0.50 m tall. A round hoop of the average size would score both the other way.
    (0.06, 0.0, "hit"),       # contact from 0.05 m sideways, not 0.075 m
    (0.0, 0.124, "passed"),   # contact from ~0.127 m up, not ~0.12 m
])
def test_squashed_hoop_is_scored_as_seen(lateral, vertical, expected):
    env = make_env()
    env.place([5.0, 5.0 + lateral, 1.5 + vertical], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0],
              hoop_diameter=(0.40, 0.50))
    assert fly(env, [0.0, 1.0, 0.0, 0.0]) == expected


def test_leaving_cage_ends_episode():
    env = make_env()
    env.place([5.0, 5.0, 1.5], 0.0, [6.0, 5.0, 1.5], [1.0, 0.0, 0.0])
    assert fly(env, [0.0, 0.0, 1.0, 0.0]) == "left_cage"


# --- the real cage ----------------------------------------------------------

def test_real_cage_layouts_start_with_hoop_in_view():
    env = HoopEnv(camera=Camera())
    visible = []
    for seed in range(200):
        obs, info = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        visible.append(obs[(contract.OBS_STACK - 1) * contract.OBS_DIM + VISIBLE])
    assert np.mean(visible) > 0.95


def test_gymnasium_api():
    check_env(HoopEnv(camera=Camera()), skip_render_check=True)


# --- laps -------------------------------------------------------------------
# Hoop at (6, 5, 1.5) with its back facing +x, so the front (and the marker)
# faces a drone coming from x < 6.

HOOP_AT, BACK = [6.0, 5.0, 1.5], [1.0, 0.0, 0.0]


def laps_env(**kwargs):
    return make_env(task="laps", **kwargs)


def goto(env, x, y, z=1.5, max_steps=300):
    """Fly to (x, y) facing +x with a proportional controller. Returns the last step."""
    for _ in range(max_steps):
        dx, dy, dz = x - env.pos[0], y - env.pos[1], z - env.pos[2]
        step = env.step(np.clip([-3 * dy, 3 * dx, 3 * dz, 0.0], -1, 1))
        if step[2] or step[3] or np.hypot(dx, dy) < 0.02:
            return step
    return step


def test_lap_pass_scores_and_keeps_flying():
    env = laps_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for _ in range(100):
        _, reward, terminated, truncated, info = env.step([0.0, 1.0, 0.0, 0.0])
        if info["outcome"] == "passed":
            break
    assert info["outcome"] == "passed" and reward > 20.0
    assert not terminated and not truncated and info["passes"] == 1


def test_back_through_the_ring_is_wrong_way():
    env = laps_env()
    env.place([7.0, 5.0, 1.5], np.pi, HOOP_AT, BACK)  # behind the hoop, facing it
    assert fly(env, [0.0, 1.0, 0.0, 0.0]) == "wrong_way"


def test_crossing_outside_the_ring_is_neutral():
    env = laps_env()
    env.place([7.0, 5.8, 1.5], np.pi, HOOP_AT, BACK)
    for _ in range(40):  # 1.6 m at 0.4 m/s: well past the plane
        _, _, terminated, truncated, info = env.step([0.0, 1.0, 0.0, 0.0])
        assert not terminated and not truncated
    assert env.pos[0] < 6.0 and info["passes"] == 0


def test_a_full_lap_scores_twice():
    env = laps_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for x, y in [(6.6, 5.0), (6.6, 5.9), (5.4, 5.9), (5.4, 5.0), (6.6, 5.0)]:
        _, _, terminated, truncated, info = goto(env, x, y)
        assert not terminated and not truncated, info["outcome"]
    assert info["passes"] == 2


def test_route_records_the_way_round():
    """Back to front outside the ring, at +y: the hoop's left, seen from the front."""
    env = laps_env()
    env.place([6.6, 5.9, 1.5], 0.0, HOOP_AT, BACK)
    routes = [env.step([0.0, -1.0, 0.0, 0.0])[4]["route"] for _ in range(20)]
    assert [r for r in routes if r] == ["left"]


@pytest.mark.parametrize("y, z, expected", [
    (5.0, 2.0, "hit"),       # straight over the top: into the hanger
    (5.0, 1.0, "under"),     # below the hoop: open
    (5.4, 2.0, "over"),      # over, but clear of the strip's side
])
def test_hanger_blocks_going_over_the_top(y, z, expected):
    env = laps_env()
    env.place([6.6, y, z], 0.0, HOOP_AT, BACK)   # behind the hoop, reversing towards the front
    for _ in range(20):
        _, _, terminated, _, info = env.step([0.0, -1.0, 0.0, 0.0])
        if terminated or info["route"]:
            break
    assert (info["outcome"] if terminated else info["route"]) == expected


def test_progress_targets_a_side_waypoint_from_behind():
    env = laps_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    assert np.allclose(env._progress_target(np.array([5.5, 5.2, 1.5])), HOOP_AT)
    gate = env._progress_target(np.array([6.5, 5.2, 1.5]))
    assert np.allclose(gate, [6.0, 5.0 + EXACT.outer_diameter / 2 + SIDE_GAP, 1.5])


@pytest.mark.parametrize("angle_deg, expected", [(0, 1), (60, 1), (75, 0), (180, 0)])
def test_marker_is_seen_only_from_the_front(angle_deg, expected):
    """Drone 1 m from the hoop at angle_deg off its front normal, facing the hoop."""
    env = laps_env()
    a = np.deg2rad(angle_deg)
    drone = np.array(HOOP_AT) + [-np.cos(a), np.sin(a), 0.0]
    env.place(drone, np.arctan2(-np.sin(a), np.cos(a)), HOOP_AT, BACK)
    obs = env.observe()
    assert obs[VISIBLE] == 1   # the hoop itself is in view every time
    assert obs[MARKER] == expected


def test_marker_needs_the_hoop():
    assert contract.observe(np.zeros((2, 2)), (0.5, 0.4), marker=True)[MARKER] == 0


def test_laps_run_to_the_cut_off():
    env = laps_env()
    env.place([5.0, 5.0, 1.5], 0.0, HOOP_AT, BACK)
    for _ in range(LAPS_MAX_STEPS):
        _, _, terminated, truncated, info = env.step([0.0, 0.0, 0.0, 0.0])
        assert not terminated
    assert truncated and info["outcome"] == "timeout"


def test_pad_layouts_fit_the_real_cage():
    env = HoopEnv(task="laps", camera=Camera())
    for seed in range(300):
        obs, _ = env.reset(seed=seed)
        assert env.observation_space.contains(obs)
        assert env._in_fence(env.pos)
        assert (env.pos - env.hoop_centre) @ env.hoop_normal < 0   # starts in front
        assert all(env._in_fence(g) for g in env._gates)           # room to go round
        # Level 1 is the real start: take-off height, above a pad in a corner.
        assert env.pos[2] == pytest.approx(env.drone.takeoff_height)
        inset = env.pad.inset
        assert {round(min(c, env.cage.length - c), 6) for c in env.pos[:2]} == {inset}


def test_level_zero_starts_with_hoop_and_marker_in_view():
    env = HoopEnv(task="laps", level=0.0, camera=Camera())
    seen = []
    for seed in range(200):
        obs, _ = env.reset(seed=seed)
        seen.append(obs[(contract.OBS_STACK - 1) * contract.OBS_DIM + MARKER])
    assert np.mean(seen) > 0.95


def test_randomised_dynamics_stay_in_range_and_fixed_per_episode():
    env = HoopEnv(task="laps", randomise=True, camera=Camera())
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


def test_laps_gymnasium_api():
    check_env(HoopEnv(task="laps", randomise=True, camera=Camera()), skip_render_check=True)


def test_hanger_can_be_left_out():
    env = laps_env(hanger=False)
    env.place([6.6, 5.0, 2.0], 0.0, HOOP_AT, BACK)
    routes = [env.step([0.0, -1.0, 0.0, 0.0])[4]["route"] for _ in range(20)]
    assert [r for r in routes if r] == ["over"]
