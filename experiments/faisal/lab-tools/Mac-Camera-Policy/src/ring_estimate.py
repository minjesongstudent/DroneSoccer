"""Optional camera-derived ring-centre estimate for the metric-ring trial.

Weak perspective plus nominal command odometry is approximate, not measured
pose. No ring normal, unseen wall position or safe action is inferred here.
"""
from dataclasses import replace
import gymnasium as gym
import numpy as np
from src.params import Camera, Hoop

VERSION='public-approximate-ring-centre-v1'
INPUT_DIM=115
OUTPUT_DIM=119
SCALE_METRES=3.0


class RingCentreEstimate:
    def __init__(self, camera=None, diameter=None):
        self.camera=replace(camera or Camera())
        self.diameter=float(Hoop().outer_diameter if diameter is None else diameter)
        values=(self.camera.fx,self.camera.fy,self.camera.width,self.camera.height,self.diameter)
        if not all(np.isfinite(v) and v>0 for v in values):raise ValueError('Positive finite nominal calibration required')
        if self.camera.pitch_deg!=0 or not np.isfinite(self.camera.offset_forward):
            raise ValueError('Estimator supports a level nominal camera with finite offset')

    def encode(self,observation):
        public=np.asarray(observation,dtype=np.float32)
        if public.shape!=(INPUT_DIM,) or not np.isfinite(public).all():
            raise ValueError('Finite115D public input required')
        extra=np.zeros(4,np.float32)
        u,v,major=public[100:103]
        # Clipped fits or clipped odometry cannot support this estimate.
        valid=(public[108]>.5 and public[106]>.5 and public[107]>.5
               and 0<major<2.999 and abs(u)<2.999 and abs(v)<2.999
               and np.all(np.abs(public[110:113])<.999))
        if valid:
            depth=self.diameter/major;hx,hy=self.camera.half_extent
            relative=np.array([depth+self.camera.offset_forward,-u*hx*depth,-v*hy*depth])-public[110:113]*SCALE_METRES
            sine,cosine=public[113:115]
            body=np.array([cosine*relative[0]+sine*relative[1],-sine*relative[0]+cosine*relative[1],relative[2]])
            if np.isfinite(body).all() and abs(np.linalg.norm([sine,cosine])-1.0)<.01:
                extra[:3]=np.clip(body/SCALE_METRES,-1,1)
                extra[3]=1.0
        return np.concatenate([public,extra])


class MetricRingObservation(gym.ObservationWrapper):
    """Stateless append: authentic115D reset histories can be reused unchanged."""
    def __init__(self,env,*,nominal_camera=None,nominal_diameter=None):
        super().__init__(env)
        if env.observation_space.shape!=(INPUT_DIM,):raise ValueError('Original115D contract required')
        self.estimator=RingCentreEstimate(nominal_camera,nominal_diameter)
        self.observation_space=gym.spaces.Box(np.r_[env.observation_space.low,[-1,-1,-1,0]].astype(np.float32),np.r_[env.observation_space.high,[1,1,1,1]].astype(np.float32))

    def observation(self,observation):return self.estimator.encode(observation)
