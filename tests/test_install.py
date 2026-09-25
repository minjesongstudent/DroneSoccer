"""Checks that everything training needs is installed and works together.

Run this first on a new machine: pytest tests/test_install.py
djitellopy is left out: only the laptop that flies the drone needs it.
"""
import importlib

import pytest

TRAINING_PACKAGES = ["numpy", "cv2", "gymnasium", "torch", "stable_baselines3", "tensorboard", "matplotlib"]


@pytest.mark.parametrize("name", TRAINING_PACKAGES)
def test_package_imports(name):
    importlib.import_module(name)


def test_short_training_run(tmp_path):
    """A few hundred SAC steps with TensorBoard logging, then save, reload and fly."""
    from stable_baselines3 import SAC

    from src.evaluate import load_policy, run_episodes
    from src.sim import HoopEnv

    model = SAC("MlpPolicy", HoopEnv(), seed=0, device="cpu", learning_starts=100,
                tensorboard_log=str(tmp_path))
    model.learn(total_timesteps=300)
    model.save(tmp_path / "model")

    results = run_episodes(load_policy(str(tmp_path / "model.zip")), episodes=2)
    assert len(results) == 2
