import pytest
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src import train
from src.params import Camera
from src.contract import LAPS_MAX_STEPS
from src.sim import HoopEnv


def test_curriculum_raises_the_level(monkeypatch):
    """With the bar at zero, every full window of episodes steps the level up."""
    monkeypatch.setattr(train, "LEVEL_WINDOW", 1)
    monkeypatch.setattr(train, "LEVEL_UP", 0.0)
    monkeypatch.setattr(train, "LEVEL_PASSES", 0)
    env = Monitor(HoopEnv(level=0.0, camera=Camera()))
    model = SAC("MlpPolicy", env, seed=0, device="cpu", learning_starts=10_000)
    callback = train.Progress(total=2 * LAPS_MAX_STEPS)
    model.learn(total_timesteps=2 * LAPS_MAX_STEPS, callback=callback)
    assert callback.level > 0.0
    assert env.get_wrapper_attr("level") == callback.level


def test_curriculum_counts_passes_however_the_episode_ended():
    """The level changes only the start, so passes before a crash still count."""
    callback = train.Progress(total=1)
    callback.recent.extend([{"outcome": "net", "passes": train.LEVEL_PASSES}] * 3
                           + [{"outcome": "timeout", "passes": 0}])
    assert callback._ready() == pytest.approx(0.75)


def test_passing_once_then_crashing_does_not_outscore_lapping():
    """Per minute flown, one pass and a crash at 8 s scored 7.5 against 5 for
    lapping the whole minute. Per episode it scores 1 against 5."""
    crash_fast = [{"outcome": "net", "passes": 1, "steps": 80}] * 10
    laps = [{"outcome": "timeout", "passes": 5, "steps": LAPS_MAX_STEPS}] * 10
    assert train.scores(crash_fast)["passes_per_ep"] == pytest.approx(1.0)
    assert train.scores(laps)["passes_per_ep"] == pytest.approx(5.0)


def test_resume_carries_on_from_the_snapshot(monkeypatch, tmp_path, capsys):
    """A run cut off after its last snapshot continues to the full step count,
    at the level it had reached, with its replay buffer."""
    monkeypatch.setattr(train, "RUNS", tmp_path)
    monkeypatch.setattr(train, "CHECKPOINT_EVERY", 200)
    monkeypatch.setattr(train, "BUFFER_SIZE", 2000)   # the real one saves 700 MB
    train.train(600, "r", seed=0, level=0.5, randomise=False)
    buffered = SAC.load(tmp_path / "r" / "resume" / "model.zip")
    buffered.load_replay_buffer(tmp_path / "r" / "resume" / "replay_buffer.pkl")
    # SB3 calls back before it stores the step, so the newest transition is not in yet.
    assert buffered.replay_buffer.size() == 599

    path = train.train(1000, "r", seed=0, level=0.0, randomise=False, resume=True)
    assert "resumed  from step 600 at level 0.50" in capsys.readouterr().out
    assert SAC.load(path).num_timesteps == 1000
    state = (tmp_path / "r" / "resume" / "state.json").read_text()
    assert '"steps": 1000' in state and '"level": 0.5' in state
    assert not list((tmp_path / "r" / "resume").glob("*.tmp"))


def test_init_grows_an_older_policy_without_changing_what_it_does(tmp_path):
    """A model saved for the 84-number input, before the seen-hoop flag, starts out
    acting exactly as before, whatever the flag reads."""
    import gymnasium as gym
    import numpy as np
    env = Monitor(HoopEnv(camera=Camera()))
    old_space = gym.spaces.Box(-np.inf, np.inf, (env.observation_space.shape[0] - 1,), np.float32)

    class Old(gym.Wrapper):
        observation_space = old_space

    old = SAC("MlpPolicy", Old(env), seed=1, device="cpu")
    old.save(tmp_path / "old")
    new = train.grown(str(tmp_path / "old.zip"), env, seed=0, envs=1)
    x = np.random.default_rng(0).normal(size=(50, old_space.shape[0])).astype(np.float32)
    for flag in (0.0, 1.0):
        xn = np.concatenate([x, np.full((50, 1), flag, np.float32)], axis=1)
        assert np.array_equal(new.predict(xn, deterministic=True)[0], old.predict(x, deterministic=True)[0])
    assert new.learning_starts == train.INIT_WARMUP
