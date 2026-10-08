"""Training reset distribution only; every subsequent movement remains learned."""
import numpy as np
from src.sim import HoopEnv


class ApproachCurriculumEnv(HoopEnv):
    def reset(self, *, seed=None, options=None):
        if seed is None or not 11240000 <= seed < 11245000:
            raise ValueError('Explicit fresh FIT curriculum seed required')
        obs,info=super().reset(seed=seed,options=options)
        self.curriculum_start=False
        if seed%2:
            return obs,info
        rng=np.random.default_rng(seed+30000000)
        centre=self.hoop_centre.copy();normal=self.hoop_normal.copy()
        diameters=(2*np.linalg.norm(self._hoop_h),2*np.linalg.norm(self._hoop_v))
        for _ in range(64):
            pos=(centre-rng.uniform(.45,.65)*normal
                 +rng.uniform(-.18,.18)*self._axis_h
                 +np.array([0,0,rng.uniform(-.06,.06)]))
            if self._in_fence(pos[:2]) and self.cage.min_altitude+.05<pos[2]<self.cage.height-.1:
                heading=np.arctan2(centre[1]-pos[1],centre[0]-pos[0])+rng.uniform(-np.pi/12,np.pi/12)
                obs=self.place(pos,heading,centre,normal,hoop_diameter=diameters,
                               hanger_half_width=self.hanger_half_width)
                self.curriculum_start=True
                return obs,self._info('running')
        raise RuntimeError('Cannot construct curriculum reset inside existing cage')
