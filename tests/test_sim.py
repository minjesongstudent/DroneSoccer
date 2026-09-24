import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from src import contract
from src.params import Cage, Camera, Drone, Hoop
from src.sim import HoopEnv

# A roomy cage and an exact hoop, so the geometry tests are not about the real layout.
BIG = Cage(length=10.0, width=10.0, height=4.0)
EXACT = Hoop(diameter_tolerance=0.0)
U, V, MAJOR, RATIO, SIN, COS, VISIBLE = range(7)


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
