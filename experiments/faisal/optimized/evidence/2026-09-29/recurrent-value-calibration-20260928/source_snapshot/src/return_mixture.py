"""One actor rehearses passing, whole flights, and authentic return continuations."""
import gymnasium as gym
import numpy as np

from src.guided_curriculum import GuidedLessonEnv, STAGES
from src.mission_observation import MissionObservation
from src.return_practice import ReturnPracticeEnv

VERSION = 'authentic-return-start-mixture-v1'


def display_mix(stage):
    return [{'stage': value, 'name': STAGES[value], 'probability': .2}
            for value in (2, 3, 4, max(5, stage))] + [
                {'stage': None, 'name': 'Authentic mid-flight return', 'probability': .2}]


class ReturnMixtureEnv(gym.Env):
    """Choose one complete observation-producing environment per episode.

    Restored return observations bypass a new memory reset. Only the restored
    policy history determines them; this mixture contributes no actor inputs.
    """
    def __init__(self, library, stage=4, observation_wrapper=None):
        self.practice = ReturnPracticeEnv(library)
        if self.practice.library.manifest.get('split') != 'train':
            raise ValueError('Training requires a training snapshot bank, never the development bank.')
        self.regular = [MissionObservation(GuidedLessonEnv(stage=value), single_pass=value < 4)
                        for value in (2, 3, 4, max(5, stage))]
        if observation_wrapper is not None:
            self.regular = [observation_wrapper(env) for env in self.regular]
        self.active = self.regular[0]
        self.observation_space = self.active.observation_space
        self.action_space = self.active.action_space
        if self.practice.observation_space != self.observation_space:
            raise ValueError('Return practice and whole flights must share the observation contract.')
        self.focus_stage = self.next_focus_stage = int(stage)
        self.set_stage(stage)
        self._choice_rng = np.random.default_rng()

    @property
    def unwrapped(self):
        # Viewer and diagnostics need the actual active physical environment.
        return self.active.unwrapped

    def set_stage(self, stage):
        if int(stage) != stage or not 4 <= stage < len(STAGES):
            raise ValueError('Return training focus must be a lap stage.')
        self.next_focus_stage = int(stage)

    def _info(self, info):
        return dict(info, focus_stage=self.focus_stage,
                    return_practice=self.active is self.practice)

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._choice_rng = np.random.default_rng(seed)
        self.focus_stage = self.next_focus_stage
        self.regular[3].unwrapped.set_stage(max(5, self.focus_stage))
        choice = int(self._choice_rng.integers(5))
        self.active = self.practice if choice == 4 else self.regular[choice]
        episode_seed = int(self._choice_rng.integers(1_000_000))
        obs, info = self.active.reset(seed=episode_seed, options=options)
        return obs, self._info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active.step(action)
        return obs, reward, terminated, truncated, self._info(info)

    def close(self):
        for env in [*self.regular, self.practice]:
            env.close()
