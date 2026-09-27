"""One policy rehearses complete pass lessons as well as front/corner laps."""
import numpy as np

from src.guided_curriculum import GuidedLessonEnv, GuidedCurriculum

VERSION = 'pass-retention-and-laps-v1'


def training_mix(stage):
    if not 4 <= int(stage) <= 8 or int(stage) != stage:
        raise ValueError('Completion focus must be between 4 and 8')
    weights = np.zeros(max(6, stage+1))
    weights[2:4] = .2  # Actual single-pass tasks, including their termination.
    if stage == 4:
        weights[4:6] = [.4, .2]
    else:
        weights[4] = .2
        weights[stage] += .4
    return tuple(weights)


class CompletionEnv(GuidedLessonEnv):
    def __init__(self, stage=4, **kwargs):
        training_mix(stage)
        self.focus_stage = self.next_focus_stage = int(stage)
        self.practice_stage = int(stage)
        self._mixture_rng = np.random.default_rng()
        super().__init__(stage=stage, **kwargs)

    def set_stage(self, stage):
        training_mix(stage)
        self.next_focus_stage = int(stage)

    def _sample_practice_stage(self):
        p = training_mix(self.focus_stage)
        return int(self._mixture_rng.choice(len(p), p=p))

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self._mixture_rng = np.random.default_rng(seed)
        self.focus_stage = self.next_focus_stage
        self.practice_stage = self._sample_practice_stage()
        self.next_guided_stage = self.practice_stage
        return super().reset(seed=seed, options=options)

    def _guided_info(self, info):
        info = super()._guided_info(info)
        info['focus_stage'] = self.focus_stage
        info['practice_stage'] = self.practice_stage
        # Requested task is public context, not estimated hidden flight state.
        info['single_pass_task'] = self.practice_stage < 4
        return info


class Curriculum(GuidedCurriculum):
    def __init__(self, state=None):
        super().__init__({'stage': 4, **(state or {})})

    def record(self, info):
        # A short mid-flight continuation cannot promote a 60-second mission.
        return False if info.get('return_practice', False) else super().record(info)
