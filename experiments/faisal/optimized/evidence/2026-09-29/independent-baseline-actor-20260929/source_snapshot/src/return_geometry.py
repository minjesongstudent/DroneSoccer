"""Optional geometric reward-state correction.

Recognize an actual outside-plane return followed by full front clearance.
Simulator geometry affects reward state only, never observations or actions.
"""
import gymnasium as gym
from src.sim import CRASHES, R_PROGRESS, PROGRESS_UNIT

VERSION='outside-return-geometric-progress-v1'

class GeometricReturnProgress(gym.Wrapper):
    def __init__(self,env):
        super().__init__(env)
        self.outside_return_seen=False

    def reset(self,**kwargs):
        self.outside_return_seen=False
        return self.env.reset(**kwargs)

    def step(self,action):
        base=self.unwrapped
        previous=base.pos.copy();phase=base._lap_phase;passes=base.passes
        obs,reward,term,trunc,info=self.env.step(action)
        crossing=base._crossing(previous)
        advanced=False;correction=0.
        if base.passes!=passes:
            self.outside_return_seen=False
        elif info['outcome'] not in CRASHES:
            if phase in ('return_side','return_front') and crossing is not None and crossing[0]==-1 and not crossing[1]:
                self.outside_return_seen=True
            front_clear=(base.pos-base.hoop_centre)@base.hoop_normal < -base._clearance_reach()
            if self.outside_return_seen and front_clear and base._lap_phase in ('return_side','return_front'):
                before=base._progress_distance(base.pos)
                base._lap_phase='align_front';base._return_gate=None
                correction=R_PROGRESS*(before-base._progress_distance(base.pos))/PROGRESS_UNIT
                reward+=correction
                self.outside_return_seen=False;advanced=True
        return obs,float(reward),term,trunc,dict(info,lap_phase=base._lap_phase,
            geometric_return_advanced=advanced,geometric_reward_correction=correction)
