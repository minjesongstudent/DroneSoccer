"""Experimental fresh-start return lessons; not enabled in production training.

Reset geometry defines a shorter remaining route. Histories are initialized
from the actual first camera frame, never from an invented earlier flight.
"""
import numpy as np
from src.guided_curriculum import GuidedLessonEnv
from src.sim import CRASHES

VERSION = 'fresh-return-segment-resets-v1'
LEVELS = ('near alignment', 'wide alignment', 'short front return',
          'half front return', 'whole front return', 'side departure')


class ReturnResetLesson(GuidedLessonEnv):
    def __init__(self, lesson=0, horizon=200):
        if isinstance(lesson, bool) or int(lesson)!=lesson or not 0<=lesson<len(LEVELS):
            raise ValueError('Invalid return reset lesson')
        if isinstance(horizon,bool) or int(horizon)!=horizon or not 1<=horizon<=600:
            raise ValueError('Invalid return lesson horizon')
        self.lesson=int(lesson)
        self.horizon=int(horizon)
        super().__init__(stage=4)

    def reset(self, *, seed=None, options=None):
        if options: raise ValueError('Return reset geometry is versioned, not caller supplied')
        super().reset(seed=seed)
        centre=self.hoop_centre.copy();normal=self.hoop_normal.copy()
        side=self._axis_h.copy();offset=self._clearance_target_offset()
        gate=self._gates[int(self.np_random.integers(len(self._gates)))].copy()
        front=centre-offset*normal
        if self.lesson==0:
            pos=front-.40*normal+.04*(gate-centre)/np.linalg.norm(gate-centre)
            phase='align_front'
        elif self.lesson==1:
            pos=gate-offset*normal;phase='align_front'
        elif self.lesson in (2,3,4):
            fraction={2:.20,3:.50,4:1.0}[self.lesson]
            pos=gate-offset*normal+2*offset*fraction*normal
            phase='return_front'
        else:
            pos=centre+offset*normal;phase='return_side'
        # Aim the actual camera at the ring at reset, not during any step.
        heading=float(np.arctan2(centre[1]-pos[1],centre[0]-pos[0]))
        self._set_layout(pos,heading,centre,normal,np.linalg.norm(self._hoop_h),np.linalg.norm(self._hoop_v))
        observation=self._start_episode()
        self._lap_phase=phase
        self._return_gate=gate if phase=='return_front' else None
        self.finished=False
        if not self._in_fence(self.pos) or self._outcome(self.pos.copy()) in CRASHES:
            raise ValueError('Reset lesson geometry is not collision-free')
        return observation,self._lesson_info(self._guided_info(self._info('running')))

    def _lesson_info(self,info):
        return dict(info,reset_lesson=self.lesson,reset_lesson_version=VERSION,
                    fresh_reset=True,historical_passes=0,full_mission_success=False,
                    reset_lesson_success=bool(self.passes>=1 and info['outcome']=='timeout'),
                    focus_stage=4,single_pass_task=False)

    def step(self,action):
        if self.finished:raise RuntimeError('Reset the completed lesson')
        obs,reward,term,trunc,info=super().step(action)
        if not term and self.steps>=self.horizon:
            trunc=True;info=dict(info,outcome='timeout')
        self.finished=bool(term or trunc)
        return obs,reward,term,trunc,self._lesson_info(info)
