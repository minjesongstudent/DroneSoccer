"""Standard recurrent PPO baseline using complete public-observation episodes."""
import numpy as np
import gymnasium as gym
from src.guided_curriculum import GuidedLessonEnv
from src.return_reset_curriculum import wrap

VERSION = 'recurrent-ppo-complete-flights-v1'


class RecurrentPredictor:
    """Stateful single-flight inference; explicitly reset on every new episode."""
    def __init__(self, model):
        self.model = model
        self.reset_episode()

    @property
    def num_timesteps(self):
        return self.model.num_timesteps

    def reset_episode(self):
        self.state = None
        self.episode_start = np.ones(1, dtype=bool)

    def predict(self, observation, deterministic=True):
        action, self.state = self.model.predict(observation, state=self.state,
            episode_start=self.episode_start, deterministic=deterministic)
        self.episode_start[:] = False
        return action, self.state


class CompleteFlightMixture(gym.Env):
    """Equal mixture: close, near, centered, varied passes, and full front laps.

    Every reset is a genuine new episode; no mid-flight snapshots or invented
    recurrent histories. No demonstration controller supplies any actions.
    """
    def __init__(self):
        self.envs = [wrap(GuidedLessonEnv(stage=k), single=k<4,
                         metric_ring=True, geometric_return=True) for k in range(5)]
        self.active = self.envs[0]
        self.observation_space = self.active.observation_space
        self.action_space = self.active.action_space
        self.rng = np.random.default_rng()

    def reset(self, *, seed=None, options=None):
        if options:raise ValueError('Only complete natural starts supported')
        if seed is not None:self.rng = np.random.default_rng(seed)
        self.active = self.envs[int(self.rng.integers(5))]
        return self.active.reset(seed=int(self.rng.integers(1_000_000)))

    def step(self, action):
        return self.active.step(action)

    def close(self):
        for env in self.envs:env.close()
