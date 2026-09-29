"""FIT-only backward practice, with passing and whole-flight rehearsal."""
import gymnasium as gym
import numpy as np
from src.success_history import SuccessHistoryLibrary, SuccessHistoryPractice, LANDMARKS, FIT
from src.return_reset_curriculum import wrap
from src.guided_curriculum import GuidedLessonEnv

VERSION = 'authentic-success-backward-curriculum-v1'


class BackwardScheduler:
    def __init__(self):
        self.level = 0
        self.streak = 0
        self.last_step = -1
        self.history = []

    def observe(self, step, records):
        if type(step) is not int or step <= self.last_step:
            raise ValueError('Steps must increase')
        expected = {(seed, mark, rep) for seed in FIT for mark in LANDMARKS[:self.level+1] for rep in range(9)}
        keys = [(r['seed'], r['landmark'], r['replicate']) for r in records]
        if len(keys) != len(expected) or set(keys) != expected or any(type(r['success']) is not bool for r in records):
            raise ValueError('Complete FIT-only deterministic and stochastic checks required')
        competent = all(
            all(r['success'] for r in records if r['landmark'] == mark and r['replicate'] == 0)
            and sum(r['success'] for r in records if r['landmark'] == mark and r['replicate'] > 0) >= 24
            for mark in LANDMARKS[:self.level+1])
        self.streak = self.streak + 1 if competent else 0
        before = self.level
        if self.streak >= 2 and self.level < len(LANDMARKS)-1:
            self.level += 1
            self.streak = 0
        self.last_step = step
        self.history.append(dict(step=step, level_before=before, level=self.level, competent=competent))
        return self.level != before


class SuccessMixture(gym.Env):
    """40% passing,20% whole flights,20% current and20% earlier authentic starts."""
    def __init__(self, library, scheduler):
        if not isinstance(library, SuccessHistoryLibrary) or library.split != 'fit' or library.evaluation_only:
            raise ValueError('Only FIT snapshots can enter training')
        if library.fit_seeds != FIT:
            raise ValueError('This scheduler is preregistered for the original FIT layouts only')
        self.scheduler = scheduler
        self.practice = SuccessHistoryPractice(library)
        self.regular = [wrap(GuidedLessonEnv(stage=k), single=k < 4,
                             metric_ring=True, geometric_return=True) for k in (2,3,4)]
        self.active = self.regular[0]
        self.rng = np.random.default_rng()
        self.observation_space = self.active.observation_space
        self.action_space = self.active.action_space
        if self.practice.observation_space != self.observation_space:
            raise ValueError('Mixed observation contracts')

    @property
    def unwrapped(self):
        return self.active.unwrapped

    def reset(self, *, seed=None, options=None):
        if options:
            raise ValueError('External reset overrides are not allowed')
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        choice = int(self.rng.integers(5))
        episode_seed = int(self.rng.integers(1_000_000))
        if choice < 3:
            self.active = self.regular[choice]
            obs, info = self.active.reset(seed=episode_seed)
        else:
            self.active = self.practice
            level = self.scheduler.level if choice == 3 else int(self.rng.integers(max(1,self.scheduler.level)))
            obs, info = self.active.reset(seed=episode_seed, options={'phase': LANDMARKS[level]})
        return obs, dict(info, focus_stage=4, success_curriculum=VERSION)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.active.step(action)
        return obs, reward, terminated, truncated, dict(info, focus_stage=4)

    def close(self):
        for env in [*self.regular, self.practice]:
            env.close()
