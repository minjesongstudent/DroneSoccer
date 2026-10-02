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
