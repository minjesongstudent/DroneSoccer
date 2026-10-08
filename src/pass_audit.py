"""Simulator-only full-body passage diagnostics. Never supplied to an actor."""
import numpy as np


class PassageAudit:
    def __init__(self):
        self.armed = False
        self.crossed = False
        self.count = 0
        self.steps = []
        self.returns = 0
        self.waiting_return = False

    def update(self, env, outcome='running', unsafe=False):
        # Project the yawed drone box onto the hoop normal, then include the
        # noodle's extent either side of the centre plane. Centre crossing alone
        # cannot score here; the rear of the entire box must clear the far side.
        c, s = np.cos(env.heading), np.sin(env.heading)
        axes = np.array([[c,s,0],[-s,c,0],[0,0,1]])
        reach = float(np.abs(axes @ env.hoop_normal) @ env._half_box)
        margin = reach + env.hoop.thickness/2
        along = float((env.pos-env.hoop_centre) @ env.hoop_normal)
        if unsafe:
            self.armed = self.crossed = False
            return
        if along < -margin:
            self.armed = True
            self.crossed = False
            if self.waiting_return:
                self.returns += 1
                self.waiting_return = False
        if outcome == 'passed' and self.armed:
            self.crossed = True
        if self.armed and self.crossed and along > margin:
            self.count += 1
            self.steps.append(env.steps)
            self.waiting_return = True
            self.armed = self.crossed = False
