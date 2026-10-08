"""Explicit experimental twelve-command policy evaluation; 121 actor inputs."""
import numpy as np
from stable_baselines3 import SAC
from src.camera_history import CameraHistory
from src.policy_runtime import SPECS
from src.twelve_action_history import TwelveHistory
from tools.evaluate_four_actions import evaluate_four_actions


class TwelveActionPolicy:
    def __init__(self,path):
        self.model=SAC.load(path,device='cpu')
        assert self.model.observation_space.shape==(121,)
        self.reset()

    def reset(self):
        self.camera=CameraHistory(SPECS['seenflag-s1-level1'].lags)
        self.history=TwelveHistory()

    def predict(self,detection,previous):
        obs=self.history.update(self.camera.update(detection,previous))
        action,_=self.model.predict(obs,deterministic=True)
        assert action.shape==(4,) and np.isfinite(action).all() and np.all(abs(action)<=1)
        return obs,action


def evaluate_twelve_actions(checkpoint,folder,count):
    return evaluate_four_actions(checkpoint,folder,count,policy_class=TwelveActionPolicy,
                                 inputs=121,action_history=12)
