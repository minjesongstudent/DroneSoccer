"""Pass first, then laps, corner starts and progressively delayed dynamics.

Only training uses these easier starts. Full-task evaluation uses HoopEnv(level=1).
Stages change on reset, never halfway through an episode.
"""
from collections import deque

import numpy as np

from src.params import Randomise
from src.sim import HoopEnv, R_PASS

STAGES = ('Centered pass', 'Varied approach', 'Laps from the front',
          'Laps from corners', 'Laps + 100 ms delay', 'Laps + 200 ms delay',
          'Laps + 300 ms delay')
WINDOW = 100
REWARD_VERSION = 'single-pass-clearance-v1'
R_LESSON_TIMEOUT = 5.0


class LessonEnv(HoopEnv):
    def __init__(self, stage=0, **kwargs):
        self.next_stage = int(stage)
        self.stage = int(stage)
        self._pending_pass = False
        super().__init__(**kwargs)

    def set_stage(self, stage):
        if not 0 <= stage < len(STAGES):
            raise ValueError('Invalid curriculum stage')
        self.next_stage = int(stage)

    def reset(self, *, seed=None, options=None):
        self.stage = self.next_stage
        self.set_level(1.0 if self.stage >= 3 else 0.0)
        self.randomise = self.stage >= 4
        if self.randomise:
            amount = self.stage - 3
            # Fixed control interval: delay and response randomisation without
            # making elapsed-time metrics depend on a guessed nominal dt.
            spread = (.05, .10, .20)[amount-1]
            self.ranges = Randomise(delay_steps=(0, amount),
                                    tau=(max(.15, .3-spread), .3+spread),
                                    rate_hz=(10.0, 10.0))
        self._pending_pass = False
        return super().reset(seed=seed, options=options)

    def _sample_layout(self):
        if self.stage >= 3:
            return super()._sample_layout()
        rng = self.np_random
        easy = self.stage == 0
        # Keep room on both sides for passing and the later go-round lesson.
        centre = np.array([self.cage.length*.60, self.cage.width*.5,
                           rng.uniform(.9, 1.15)])
        distance = rng.uniform(.50, .65) if easy else rng.uniform(.55, .90)
        bearing = rng.uniform(-.08, .08) if easy else rng.uniform(-.25, .25)
        normal = np.array([np.cos(bearing), np.sin(bearing), 0.])
        side = np.array([-normal[1], normal[0], 0.])
        pos = centre - distance*normal
        pos += side*rng.uniform(-.01 if easy else -.06, .01 if easy else .06)
        pos[2] += rng.uniform(-.01 if easy else -.06, .01 if easy else .06)
        heading = bearing + rng.uniform(-.02 if easy else -.30, .02 if easy else .30)
        radii = (self.hoop.outer_diameter + rng.uniform(-1, 1, 2)*self.hoop.diameter_tolerance)/2
        self._set_layout(pos, heading, centre, normal, *radii)

    def _progress_target(self, pos):
        if self.stage >= 2:
            return super()._progress_target(pos)
        # Aim through the opening until the whole body clears. The laps target
        # turns sideways at the centre plane, before a single pass is complete.
        # Half the horizontal diagonal bounds the box's projected reach at any
        # yaw. Keep the target beyond it, independent of the current heading,
        # so turning alone cannot move the reward target.
        reach = np.hypot(self.drone.length, self.drone.width)/2
        return self.hoop_centre + self.hoop_normal*(reach + self.hoop.thickness/2 + .10)

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if self.stage < 2:
            # A clean pass requires the back of the collision box to clear the
            # ring. The base laps simulator scores at centre-plane crossing.
            if info['outcome'] == 'passed':
                self.passes -= 1
                reward -= R_PASS
                self._pending_pass = True
                info['outcome'] = 'running'
            heading = np.array([np.cos(self.heading), np.sin(self.heading), 0.])
            side = np.array([-heading[1], heading[0], 0.])
            reach = (self.drone.length/2*abs(heading @ self.hoop_normal)
                     + self.drone.width/2*abs(side @ self.hoop_normal)
                     + self.hoop.thickness/2 + .01)
            cleared = (self.pos-self.hoop_centre) @ self.hoop_normal > reach
            if self._pending_pass and cleared and not terminated:
                self.passes = 1
                reward += R_PASS
                terminated, truncated = True, False
                info['outcome'] = 'passed'
            elif not terminated and self.steps >= 150:
                truncated = True
                info['outcome'] = 'timeout'
                # Failure to finish the short lesson has a modest cost. Preserve
                # truncation semantics and keep this much smaller than a crash.
                reward -= R_LESSON_TIMEOUT
            info['passes'] = self.passes
        info['stage'] = self.stage
        return obs, reward, terminated, truncated, info


class Curriculum:
    def __init__(self, state=None):
        state = state or {}
        self.stage = state.get('stage', 0)
        self.recent = deque(state.get('recent', []), maxlen=WINDOW)

    def record(self, info):
        if info['stage'] != self.stage:
            return False
        success = (info['outcome'] == 'passed' if self.stage < 2 else
                   info['outcome'] == 'timeout' and info['passes'] >= 2)
        self.recent.append(bool(success))
        threshold = .8 if self.stage < 2 else .7
        if len(self.recent) == WINDOW and self.rate >= threshold and self.stage < len(STAGES)-1:
            self.stage += 1
            self.recent.clear()
            return True
        return False

    @property
    def rate(self):
        return sum(self.recent)/len(self.recent) if self.recent else 0.

    def state(self):
        return {'stage': self.stage, 'recent': list(self.recent)}
