from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor

from src import train
from src.params import Camera
from src.sim import LAPS_MAX_STEPS, HoopEnv


def test_curriculum_raises_the_level(monkeypatch):
    """With the bar at zero, every full window of episodes steps the level up."""
    monkeypatch.setattr(train, "LEVEL_WINDOW", 1)
    monkeypatch.setattr(train, "LEVEL_UP", 0.0)
    monkeypatch.setattr(train, "LEVEL_PASSES", 0)
    env = Monitor(HoopEnv(task="laps", level=0.0, camera=Camera()))
    model = SAC("MlpPolicy", env, seed=0, device="cpu", learning_starts=10_000)
    callback = train.Progress(total=2 * LAPS_MAX_STEPS, task="laps")
    model.learn(total_timesteps=2 * LAPS_MAX_STEPS, callback=callback)
    assert callback.level > 0.0
    assert env.get_wrapper_attr("level") == callback.level
