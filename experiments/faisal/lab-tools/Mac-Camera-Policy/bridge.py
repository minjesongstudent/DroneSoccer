"""Real detection -> exact public observation history; no simulator state."""
from collections import deque
import cv2
import numpy as np
from src import contract
from src.observation_memory import CameraCommandMemory
from src.front_memory import FrontSightingMemory
from src.ring_estimate import RingCentreEstimate

class ObservationBridge:
    def __init__(self):
        self.any_memory=CameraCommandMemory()
        self.front_memory=FrontSightingMemory()
        # Preserve the trained estimator's nominal calibration, not a hidden pose.
        self.estimator=RingCentreEstimate()
        self.ready=False

    def base(self):
        return np.concatenate([*[self.frames[-1-lag] for lag in contract.OBS_LAGS],*self.actions]).astype(np.float32)

    def update(self, detection, issued_action=None):
        detection=np.asarray(detection,dtype=np.float32)
        if detection.shape!=(8,) or not np.isfinite(detection).all():
            raise ValueError('Eight finite camera features required')
        if not self.ready:
            self.frames=deque([detection.copy() for _ in range(25)],maxlen=25)
            self.actions=deque([np.zeros(4,np.float32) for _ in range(3)],maxlen=3)
            base=self.base()
            self.any_memory.reset(base);self.front_memory.reset(base)
            self.ready=True
        else:
            if issued_action is None:
                raise ValueError('Actual issued command required, not an unsent policy suggestion')
            action=np.asarray(issued_action,dtype=np.float32)
            if action.shape!=(4,) or not np.isfinite(action).all() or np.any(np.abs(action)>1):
                raise ValueError('Invalid issued action')
            self.frames.append(detection.copy());self.actions.append(action.copy())
            base=self.base()
            self.any_memory.advance(action,base);self.front_memory.advance(action,base)
        original=np.r_[self.any_memory.encode(base),np.float32(1)]
        public=np.r_[original,self.front_memory.encode(base)[84:]].astype(np.float32)
        return self.estimator.encode(public)

def detect(frame, config):
    """Colour-segment outer contour; preview thresholds require real lab validation."""
    h,w=frame.shape[:2]
    if [w,h]!=config['image_size']:
        raise ValueError(f'Frame is {w}x{h}; configure matching intrinsics, do not resize silently')
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,np.array(config['ring_hsv_low'],np.uint8),np.array(config['ring_hsv_high'],np.uint8))
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    candidates=[c for c in contours if len(c)>=20 and cv2.contourArea(c)>=config['min_area_px']]
    if not candidates:
        return contract.NOT_VISIBLE.copy(),None,mask
    contour=max(candidates,key=cv2.contourArea)
    ellipse=cv2.fitEllipse(contour)
    # Restrict marker detection to a narrow band near the candidate outer ring.
    marker=False
    if config['marker_hsv_low'] is not None:
        marker_mask=cv2.inRange(hsv,np.array(config['marker_hsv_low'],np.uint8),np.array(config['marker_hsv_high'],np.uint8))
        band=np.zeros((h,w),np.uint8)
        cv2.ellipse(band,ellipse,255,20)
        marker=cv2.countNonZero(cv2.bitwise_and(marker_mask,band))>=config['marker_min_area_px']
    K=np.array(config['camera_matrix'],dtype=float)
    points=cv2.undistortPoints(contour.astype(np.float32),K,np.array(config['distortion'],float)).reshape(-1,2)
    observation=contract.observe(points,(w/(2*K[0,0]),h/(2*K[1,1])),marker=marker)
    return observation,ellipse,mask
