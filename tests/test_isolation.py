"""The policy that flies the drone must never see the sim's true state.

The true state (HoopEnv.privileged(), positions, the hoop's pose, the episode's draw of
the dynamics) may reach the critic and a privileged teacher, both of which stay in the
sim. The student actor, the one saved for labcheck, reads only contract.policy_input:
the readings the real drone has too.
"""
import dataclasses
import inspect

import numpy as np
import pytest
import torch

from src import actor as actor_module
from src import contract, evaluate, ppo
from src.sim import PRIVILEGED_DIM, HoopEnv
from src.teacher import LearnedTeacher


def test_policy_input_is_built_from_readings_only():
    """Its only arguments are the histories the drone has: camera, height, motion,
    commands and the flag computed from the camera."""
    names = list(inspect.signature(contract.policy_input).parameters)
    assert names == ["frames", "heights", "motion", "actions", "unseen"]


def test_moving_the_true_state_leaves_the_policy_input_unchanged():
    """Rewrite everything true about the episode without new readings: the input must
    not move, so nothing true leaks into it except through the sensors."""
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    for _ in range(15):
        x, *_ = env.step(np.array([0.1, 0.5, 0.2, 0.0], np.float32))
    before = env._policy_input()
    env.pos = env.pos + 0.3
    env.vel = env.vel * -2.0
    env.heading += 1.0
    env.hoop_centre = env.hoop_centre + 0.2
    env.dyn = dataclasses.replace(env.dyn, max_speed=0.9, up_exponent=2.0)
    assert np.array_equal(env._policy_input(), before)
    assert np.array_equal(x, before)


def test_same_readings_from_different_worlds_give_the_same_input():
    """Two sims whose true states differ but whose sensor histories are copied across
    produce identical inputs: the input is the readings and nothing else."""
    a, b = HoopEnv(randomise=True), HoopEnv(randomise=True)
    a.reset(seed=1)
    b.reset(seed=2)
    for _ in range(20):
        a.step(np.array([0.2, 0.6, 0.1, 0.3], np.float32))
    for name in ("_frames", "_heights", "_motion", "_actions", "_unseen"):
        setattr(b, name, getattr(a, name))
    assert not np.allclose(a.privileged(), b.privileged())
    assert np.array_equal(a._policy_input(), b._policy_input())


def test_motion_readings_are_sensor_like():
    """The motion slots carry the Tello's packet as labcheck reads it: speed from two
    axes in whole 0.1 m/s steps and heading in whole degrees from the start, never the
    true velocity."""
    env = HoopEnv(randomise=True)
    env.reset(seed=1)
    for _ in range(30):
        x, *_ = env.step(np.array([0.0, 1.0, 0.0, 0.5], np.float32))
    motion = contract.split_motion(x)
    speeds = motion[:, 0]
    squares = (speeds / 0.1) ** 2   # vgx^2 + vgy^2 in whole dm/s
    assert np.allclose(squares, np.round(squares), atol=1e-3)
    turned = np.degrees(np.arctan2(motion[:, 1], motion[:, 2]))
    assert np.allclose(turned, np.round(turned), atol=1e-3)


def test_a_students_actor_reads_the_policy_input_only():
    """In a student's training run the actor's input is fleet.obs, whatever the true state."""
    fleet = ppo.Fleet(2, 1.0, True, 0.0, 0, np.random.default_rng(0))
    assert not fleet.privileged_actor
    x = fleet.inputs()
    assert x.shape == (2, contract.POLICY_INPUT_DIM)
    fleet.priv[:] = 123.0
    assert np.array_equal(fleet.inputs(), x)


def test_a_privileged_actor_reads_the_true_state_after_the_input():
    fleet = ppo.Fleet(2, 1.0, True, 0.0, 0, np.random.default_rng(0), privileged_actor=True)
    assert fleet.inputs().shape == (2, contract.POLICY_INPUT_DIM + PRIVILEGED_DIM)


def test_saved_students_are_not_privileged_and_teachers_are(tmp_path):
    """The flag saved with the actor is what labcheck refuses on, and what evaluate binds on."""
    student, teacher = tmp_path / "student.zip", tmp_path / "teacher.zip"
    actor_module.save_actor(actor_module.Actor(), student)
    actor_module.save_actor(actor_module.Actor(contract.POLICY_INPUT_DIM + PRIVILEGED_DIM), teacher, privileged=True)
    assert not actor_module.is_privileged(student) and actor_module.is_privileged(teacher)
    s, t = evaluate.load_policy(str(student)), evaluate.load_policy(str(teacher))
    assert not hasattr(s, "bind")           # nothing hands a student the sim
    assert s.observation_size == contract.POLICY_INPUT_DIM
    with pytest.raises(AttributeError):     # a teacher flies only with a sim bound to it
        t(np.zeros(contract.POLICY_INPUT_DIM, np.float32))


def test_labcheck_refuses_privileged_actors():
    """labcheck itself needs djitellopy, so check its policy loader's source for the guard."""
    from pathlib import Path
    source = (Path(__file__).resolve().parent.parent / "src" / "labcheck.py").read_text()
    run_policy = source[source.index("def run_policy"):source.index("def seen_once")]
    assert "is_privileged(args.model)" in run_policy and "sys.exit" in run_policy


def test_a_learned_teacher_must_be_privileged(tmp_path):
    student = tmp_path / "student.zip"
    actor_module.save_actor(actor_module.Actor(), student)
    with pytest.raises(ValueError):
        LearnedTeacher(student)


def test_a_learned_teacher_starts_every_episode_from_a_clear_memory(tmp_path):
    """Its GRU's memory is cleared on reset, so one flight's memory never steers the next."""
    path = tmp_path / "teacher.zip"
    actor_module.save_actor(actor_module.Actor(contract.POLICY_INPUT_DIM + PRIVILEGED_DIM), path, privileged=True)
    teacher, env = LearnedTeacher(path), HoopEnv(randomise=True)
    obs, _ = env.reset(seed=4)
    teacher.reset(env)
    first = teacher.act(env, obs)
    for _ in range(10):
        teacher.act(env, obs)
    teacher.reset(env)
    assert np.allclose(teacher.act(env, obs), first)


def test_the_dynamics_block_is_where_its_names_say():
    """teacher_check hides dynamics by name; the names must find the values."""
    from src.sim import DYNAMICS_FIELDS, DYNAMICS_START
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    p = env.privileged()
    for name, want in (("tau", env.dyn.tau), ("max_speed", env.dyn.max_speed),
                       ("telemetry_lag", env.telemetry_lag), ("yaw_drift", env.yaw_drift),
                       ("height_step", env.height_step)):
        assert p[DYNAMICS_START + DYNAMICS_FIELDS.index(name)] == pytest.approx(want, rel=1e-5)


def test_learned_teachers_batched_match_one_at_a_time(tmp_path):
    """The fleet's one batched teacher gives each simulator what its own teacher would."""
    from src.teacher import LearnedTeachers
    path = tmp_path / "teacher.zip"
    torch.manual_seed(0)
    actor_module.save_actor(actor_module.Actor(contract.POLICY_INPUT_DIM + PRIVILEGED_DIM), path, privileged=True)
    envs = [HoopEnv(randomise=True) for _ in range(3)]
    obs = np.stack([e.reset(seed=i)[0] for i, e in enumerate(envs)])
    batch, singles = LearnedTeachers(path, 3), [LearnedTeacher(path) for _ in envs]
    for t, e in zip(singles, envs):
        t.reset(e)
    for _ in range(4):
        priv = np.stack([e.privileged() for e in envs])
        together = batch.act(obs, priv)
        alone = np.stack([t.act(e, o) for t, e, o in zip(singles, envs, obs)])
        assert np.allclose(together, alone, atol=1e-5)
        obs = np.stack([e.step(a)[0] for e, a in zip(envs, together)])
    batch.reset(1)
    singles[1].reset(envs[1])
    priv = np.stack([e.privileged() for e in envs])
    assert np.allclose(batch.act(obs, priv)[1], singles[1].act(envs[1], obs[1]), atol=1e-5)


def test_an_actor_saved_before_the_motion_readings_still_flies(tmp_path):
    """evaluate, replay and teacher_check hand it today's input; it gets the old layout."""
    path = tmp_path / "old.zip"
    actor_module.save_actor(actor_module.Actor(contract.SLOTS_V1_DIM), path)
    policy = actor_module.load_policy(path)
    obs, _ = HoopEnv(randomise=True).reset(seed=0)
    assert policy(obs).shape == (contract.ACTION_DIM,)


# --- the gated teacher (train.py --privileged-actor gated) ------------------------------

def _seen() -> np.ndarray:
    frame = contract.NOT_VISIBLE.copy()
    frame[contract.VISIBLE] = 1.0
    return frame


def test_the_gated_view_is_blank_until_the_hoop_is_reported():
    from src.sim import GATE_HELD, GATE_NEVER, GATED_DIM
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    g = env.gated()
    assert g.shape == (GATED_DIM,)
    assert not g[GATE_HELD].any() and not g[GATE_NEVER].any()
    assert g[-2] == 0.0 and g[-1] == 1.0   # not reported yet; the time since saturated


def test_the_gated_view_follows_the_true_state_while_reported(monkeypatch):
    from src import sim
    monkeypatch.setattr(sim, "GATE_BLUR", np.zeros_like(sim.GATE_BLUR))
    monkeypatch.setattr(sim, "GATE_BLUR_FACING", (0.0,) * len(sim.GATE_BLUR_DISTANCE))
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    for _ in range(5):
        env.step(np.array([0.1, 0.4, 0.1, 0.2], np.float32))
    env._update_gate(_seen())
    g = env.gated()
    assert np.allclose(g[sim.GATE_HELD], env.privileged()[sim.GATE_HELD], atol=1e-5)
    assert g[-2] == 1.0 and g[-1] == 0.0


def test_the_gated_view_holds_while_the_hoop_is_out_of_view():
    """Once the camera stops reporting the hoop, moving the drone, the hoop or the
    heading changes nothing in the gated view: a student would not know either."""
    from src.sim import GATE_NEVER
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    env._update_gate(_seen())
    env._update_gate(contract.NOT_VISIBLE.copy())
    before = env.gated()
    env.pos = env.pos + 0.3
    env.vel = env.vel * -2.0 + 0.1
    env.heading += 1.0
    env.yaw_rate += 0.5
    env.hoop_centre = env.hoop_centre + 0.2
    after = env.gated()
    assert np.array_equal(after, before)
    assert not after[GATE_NEVER].any()   # the cage and the lap's metres to go, never


def test_the_gated_blur_is_kept_for_a_sighting_and_drawn_afresh_after(monkeypatch):
    from src import sim
    monkeypatch.setattr(sim, "GATE_BLUR", np.full_like(sim.GATE_BLUR, 0.05))
    env = HoopEnv(randomise=True)
    env.reset(seed=3)
    env._update_gate(_seen())
    first = env._held - env._relative()
    env._update_gate(_seen())
    assert np.allclose(env._held - env._relative(), first, atol=1e-6)   # same sighting, same error
    for _ in range(sim.GATE_REACQUIRE):
        env._update_gate(contract.NOT_VISIBLE.copy())
    env._update_gate(_seen())
    assert not np.allclose(env._held - env._relative(), first, atol=1e-3)   # a new sighting


def test_a_gated_actor_reads_the_gated_view_and_the_critic_the_true_state():
    from src.sim import GATED_DIM
    fleet = ppo.Fleet(2, 1.0, True, 0.0, 0, np.random.default_rng(0), privileged_actor="gated")
    x = fleet.inputs()
    assert x.shape == (2, contract.POLICY_INPUT_DIM + GATED_DIM)
    assert np.array_equal(x[:, contract.POLICY_INPUT_DIM:], np.stack([e.gated() for e in fleet.envs]))
    assert np.array_equal(fleet.priv, np.stack([e.privileged() for e in fleet.envs]))
    fleet.priv[:] = 123.0     # what the critic reads never reaches the actor
    assert np.array_equal(fleet.inputs(), x)


def test_a_saved_gated_teacher_flies_and_teaches_on_the_gated_view(tmp_path):
    from src.sim import GATED_DIM
    from src.teacher import LearnedTeachers
    path = tmp_path / "gated.zip"
    torch.manual_seed(0)
    actor_module.save_actor(actor_module.Actor(contract.POLICY_INPUT_DIM + GATED_DIM), path, privileged="gated")
    assert actor_module.is_privileged(path)
    assert actor_module.load(path)[1] == "gated"
    policy = evaluate.load_policy(str(path))
    assert isinstance(policy, actor_module.GatedPolicy)
    env = HoopEnv(randomise=True)
    obs, _ = env.reset(seed=1)
    policy.bind(env)
    assert policy(obs).shape == (contract.ACTION_DIM,)
    assert LearnedTeachers(path, 1).gated
    teacher = LearnedTeacher(path)
    teacher.reset(env)
    assert teacher.act(env, obs).shape == (contract.ACTION_DIM,)
    fleet = ppo.Fleet(2, 1.0, True, 0.0, 0, np.random.default_rng(0), teacher=str(path))
    assert fleet.gating and fleet.labels().shape == (2, contract.ACTION_DIM)
