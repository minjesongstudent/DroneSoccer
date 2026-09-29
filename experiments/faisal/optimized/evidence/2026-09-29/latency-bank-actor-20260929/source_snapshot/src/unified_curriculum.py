"""Practice earlier start layouts while keeping one current training task.

The focus stage determines rewards, termination, dynamics and promotion. Only
the initial layout is mixed: 60% use the focus layout and 40% use earlier ones.
In particular, easier starts remain 600-step lap episodes once focus reaches
stage 4; they do not reintroduce single-pass terminals into the lap task.
"""
from __future__ import annotations

import numpy as np

from src.guided_curriculum import GuidedCurriculum, GuidedLessonEnv, STAGES, reference_stage

UNIFIED_VERSION = 'mixed-start-layouts-current-task-v1'
DEFAULT_STAGE = 4


def training_mix(stage):
    """Return probabilities indexed by start-layout stage, up to the focus.

    Early single-pass lessons share rehearsal probability equally. Thereafter
    centered/varied approaches get more weight, with every earlier lap layout
    still represented. Focus 4 gives (5%, 5%, 15%, 15%, 60%).
    """
    reference_stage(stage)
    stage = int(stage)
    if stage == 0:
        return (1.,)
    weights = (np.ones(stage) if stage <= 3 else
               np.asarray([1., 1., 3., 3., 2., 2., 2., 2.][:stage]))
    return tuple(np.append(.4 * weights / weights.sum(), .6))


class MixedLessonEnv(GuidedLessonEnv):
    """A fixed focus task with a separately sampled practice start layout.

    Internal ``guided_stage``/``stage`` remain the focus task's indices. Public
    info['stage'] identifies the sampled layout so only frontier-layout trials
    enter the existing curriculum's promotion window. info['focus_stage'] is
    the task actually governing rewards, horizon and dynamics.
    """

    def __init__(self, stage=DEFAULT_STAGE, **kwargs):
        training_mix(stage)
        self.focus_stage = self.next_focus_stage = int(stage)
        self.practice_stage = int(stage)
        self._mixture_rng = np.random.default_rng()
        super().__init__(stage=stage, **kwargs)

    def set_stage(self, stage):
        training_mix(stage)
        self.next_focus_stage = int(stage)

    def _sample_practice_stage(self):
        probabilities = training_mix(self.focus_stage)
        return int(self._mixture_rng.choice(len(probabilities), p=probabilities))

    def reset(self, *, seed=None, options=None):
        # A separate RNG keeps a seed's layout identical to the corresponding
        # fixed lesson. Unseeded resets continue both streams independently.
        if seed is not None:
            self._mixture_rng = np.random.default_rng(seed)
        self.focus_stage = self.next_focus_stage
        self.practice_stage = self._sample_practice_stage()
        self.next_guided_stage = self.focus_stage
        return super().reset(seed=seed, options=options)

    def _sample_layout(self):
        # Parent layout code dispatches through these two indices. Temporarily
        # change only layout selection, then restore before episode dynamics,
        # observations or any reward/termination code can run.
        focus_guided, focus_reference = self.guided_stage, self.stage
        try:
            self.guided_stage = self.practice_stage
            self.stage = reference_stage(self.practice_stage)
            super()._sample_layout()
        finally:
            self.guided_stage, self.stage = focus_guided, focus_reference

    def _guided_info(self, info):
        info = super()._guided_info(info)
        info['stage'] = self.practice_stage
        info['practice_stage'] = self.practice_stage
        info['focus_stage'] = self.focus_stage
        return info


class Curriculum(GuidedCurriculum):
    """Promote on frontier-layout outcomes; ignore easier practice starts."""

    def __init__(self, state=None):
        super().__init__({'stage': DEFAULT_STAGE, **(state or {})})


UnifiedCurriculum = Curriculum
