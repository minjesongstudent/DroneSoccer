import numpy as np
import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from src import contract, train
from src.contract import LAPS_MAX_STEPS
from src.demo_buffer import DemoReplayBuffer
from src.sim import HoopEnv
from src.teacher import Teacher, run


def test_teacher_flies_the_mission_from_the_real_start():
    """From the blind take-off, with the drone and detector randomised, it finds the
    hoop and laps it for the minute. 200 flights: 3.1 passes each, 7% crashed."""
    out = run(10, level=1.0, seed=0)
    assert out["passes"] >= 20
    assert out["timeout"] >= 8


def test_teacher_starts_every_run_in_with_the_hoop_in_view():
    """The policy can only copy a run in whose start it saw. The first teacher's
    began blind 78% of the time, and its students flew into the rim. teacher3 only
    starts one on a frame that reports the hoop, so run() no longer counts it; this
    checks the frame the teacher acted on whenever its run_ins went up."""
    env, teacher = HoopEnv(level=1.0, randomise=True), Teacher()
    run_ins = seen = 0
    for i in range(10):
        obs, _ = env.reset(seed=i)
        teacher.reset(env)
        done = False
        while not done:
            before = teacher.run_ins
            action = teacher.act(env, obs)
            if teacher.run_ins > before:
                run_ins += 1
                seen += bool(contract.split_input(obs)[0][-1][contract.VISIBLE])
            obs, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
    assert run_ins > 0
    assert seen == run_ins


def demo_model(envs: int, demos: int, share: float = 0.25) -> SAC:
    env = DummyVecEnv([lambda: Monitor(HoopEnv(level=0.0, randomise=True))] * envs)
    return SAC("MlpPolicy", env, seed=0, device="cpu", buffer_size=5000, **train.demo_buffer(demos, share))


def test_demos_fill_the_teacher_buffer_with_passes_and_timeouts():
    """Two minutes of flying: passes in the rewards, and the first minute's end
    stored as a timeout, which bootstraps, not a crash. The agent's buffer stays empty."""
    model = demo_model(1, 2 * LAPS_MAX_STEPS)
    summary = train.add_demos(model, 2 * LAPS_MAX_STEPS, randomise=True, seed=0)
    demos = model.replay_buffer.demos
    assert demos.size() == 2 * LAPS_MAX_STEPS
    assert model.replay_buffer.size() == 0
    assert "2 flights" in summary
    assert demos.rewards[:demos.size()].max() > 20   # a pass nets ~+31
    # A flight that lasted the minute is stored as a timeout, a crash is not.
    ends = np.flatnonzero(demos.dones[:demos.size(), 0])
    lengths = np.diff(np.concatenate([[-1], ends]))
    assert np.array_equal(demos.timeouts[ends, 0] == 1, lengths == LAPS_MAX_STEPS)
    # Each stored step follows on from the one before, within a flight.
    assert np.array_equal(demos.next_observations[10, 0], demos.observations[11, 0])


def test_demos_fill_every_environment():
    model = demo_model(2, 200)
    train.add_demos(model, 200, randomise=True, seed=0)
    demos = model.replay_buffer.demos
    assert demos.size() == 100   # rows, each one step of both environments
    assert not np.array_equal(demos.observations[50, 0], demos.observations[50, 1])


@pytest.mark.parametrize("share", [0.25, 0.5])
def test_every_batch_takes_its_share_from_the_teacher(share, tmp_path):
    """However few of its own steps the agent has, and after a --resume reload."""
    model = demo_model(1, 300, share)
    train.add_demos(model, 300, randomise=True, seed=0)
    obs = np.zeros((1, *model.observation_space.shape), np.float32)
    for _ in range(10):   # the agent's own steps, marked by their reward
        model.replay_buffer.add(obs, obs, np.zeros((1, 4), np.float32), np.array([-999.0]),
                                np.array([False]), [{}])
    model.save_replay_buffer(tmp_path / "buffer.pkl")
    model.load_replay_buffer(tmp_path / "buffer.pkl")
    assert isinstance(model.replay_buffer, DemoReplayBuffer)
    batch = model.replay_buffer.sample(256)
    own = int((batch.rewards == -999.0).sum())
    assert len(batch.rewards) == 256
    assert own == 256 - round(256 * share)


def test_no_demos_keeps_the_ordinary_buffer():
    assert train.demo_buffer(0, 0.25) == {}
