"""Shorter pass lessons and successful simulation examples for training only.

The reference lessons, geometry, rewards and held-out evaluation stay separate.
The privileged demonstration controller is never used for policy evaluation.
"""
from __future__ import annotations

import math

import numpy as np

from src.curriculum import Curriculum, LessonEnv, STAGES as REFERENCE_STAGES, WINDOW

GUIDANCE_VERSION = 'close-lessons-demonstrations-v1'
STAGES = ('Close aligned pass', 'Near aligned pass', *REFERENCE_STAGES)
DEMONSTRATION_SEED_START = 1_000_000
EVALUATION_SEED_START = 3_000_000


def reference_stage(stage):
    """Map a guided lesson index to the frozen reference lesson implementation."""
    if int(stage) != stage or not 0 <= stage < len(STAGES):
        raise ValueError('Invalid guided curriculum stage')
    return max(0, int(stage)-2)


class GuidedLessonEnv(LessonEnv):
    """Parent ``stage`` stays a reference index; ``guided_stage`` is public.

    Keeping these two fields distinct is essential: the parent's reward,
    termination and randomisation branches all depend on its reference index.
    Both reset and step expose the guided index through info['stage'].
    """

    def __init__(self, stage=0, **kwargs):
        mapped = reference_stage(stage)
        self.guided_stage = int(stage)
        self.next_guided_stage = int(stage)
        super().__init__(stage=mapped, **kwargs)

    def set_stage(self, stage):
        reference_stage(stage)
        self.next_guided_stage = int(stage)

    def reset(self, *, seed=None, options=None):
        self.guided_stage = self.next_guided_stage
        self.next_stage = reference_stage(self.guided_stage)
        obs, info = super().reset(seed=seed, options=options)
        return obs, self._guided_info(info)

    def _sample_layout(self):
        if self.guided_stage >= 2:
            return super()._sample_layout()
        rng = self.np_random
        close = self.guided_stage == 0
        centre = np.array([self.cage.length*.60, self.cage.width*.5,
                           rng.uniform(.9, 1.15)])
        distance = rng.uniform(.25, .35) if close else rng.uniform(.35, .50)
        bearing = rng.uniform(-.08, .08)
        normal = np.array([np.cos(bearing), np.sin(bearing), 0.])
        side = np.array([-normal[1], normal[0], 0.])
        offset = .003 if close else .015
        pos = centre-distance*normal+side*rng.uniform(-offset, offset)
        pos[2] += rng.uniform(-offset, offset)
        yaw = .008 if close else .03
        heading = bearing+rng.uniform(-yaw, yaw)
        radii = (self.hoop.outer_diameter
                 + rng.uniform(-1, 1, 2)*self.hoop.diameter_tolerance)/2
        self._set_layout(pos, heading, centre, normal, *radii)

    def _guided_info(self, info):
        info['reference_stage'] = self.stage
        info['stage'] = self.guided_stage
        return info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return obs, reward, terminated, truncated, self._guided_info(info)


class GuidedCurriculum(Curriculum):
    def __init__(self, state=None):
        super().__init__(state)
        reference_stage(self.stage)

    def record(self, info):
        if info['stage'] != self.stage:
            return False
        success = (info['outcome'] == 'passed' if self.stage < 4 else
                   info['outcome'] == 'timeout' and info['passes'] >= 2)
        self.recent.append(bool(success))
        threshold = .8 if self.stage < 4 else .7
        if len(self.recent) == WINDOW and self.rate >= threshold and self.stage < len(STAGES)-1:
            self.stage += 1
            self.recent.clear()
            return True
        return False


def _demonstration_action(env):
    """Privileged, geometry-based teacher for single passes, never evaluation.

    Keep modest forward speed while correcting lateral/vertical displacement
    and yaw. Observations stored for the student remain the normal 84 values.
    """
    normal = env.hoop_normal
    side = env._axis_h
    relative = env.pos-env.hoop_centre
    velocity = (.22*normal-1.5*(relative @ side)*side
                -np.array([0., 0., 1.5*relative[2]]))
    heading = np.array([np.cos(env.heading), np.sin(env.heading), 0.])
    right = np.array([heading[1], -heading[0], 0.])
    error = (np.arctan2(normal[1], normal[0])-env.heading+np.pi) % (2*np.pi)-np.pi
    action = np.array([velocity @ right/env.dyn.max_speed,
                       velocity @ heading/env.dyn.max_speed,
                       velocity[2]/env.dyn.max_speed,
                       -1.5*error/np.deg2rad(env.dyn.max_yaw_rate_deg)], dtype=np.float32)
    return np.clip(action, -1., 1.)


def collect_demonstrations(episodes=96, stages=(0, 1, 2),
                           seed_start=DEMONSTRATION_SEED_START,
                           min_success_rate=.9, progress=None):
    """Return (successful transitions, JSON-safe summary) from actual env.step.

    ``episodes`` is the number of attempts, cycling through ``stages``. Every
    transition of a failed attempt is discarded. Raise if fewer than the
    requested minimum succeed or any requested stage yields no example.
    ``progress(completed_attempts, total_attempts)`` is optional.

    Transition keys: obs, action, next_obs (float32 arrays), reward, terminated,
    truncated, info, seed, stage. Seeds occupy the reserved [1M, 3M) range,
    disjoint from normal low training seeds and held-out seeds starting at 3M.
    """
    if int(episodes) != episodes or episodes <= 0:
        raise ValueError('episodes must be a positive integer')
    episodes = int(episodes)
    stages = tuple(stages)
    if not stages or episodes < len(stages) or len(set(stages)) != len(stages):
        raise ValueError('Provide unique stages and at least one attempt per stage')
    if any(reference_stage(stage) >= 2 for stage in stages):
        raise ValueError('Demonstrations support single-pass guided stages 0 through 3 only')
    if not 0 < min_success_rate <= 1:
        raise ValueError('min_success_rate must be in (0, 1]')
    if int(seed_start) != seed_start or not DEMONSTRATION_SEED_START <= seed_start < EVALUATION_SEED_START:
        raise ValueError('Demonstration seeds must start in the reserved [1000000, 3000000) range')
    seed_start = int(seed_start)
    if seed_start+episodes > EVALUATION_SEED_START:
        raise ValueError('Demonstration seeds overlap held-out evaluation seeds')

    transitions, successful_seeds = [], []
    attempted_steps = 0
    per_stage = {str(stage): {'attempted': 0, 'successful': 0, 'steps': 0}
                 for stage in stages}
    for attempt in range(episodes):
        stage = int(stages[attempt % len(stages)])
        seed = seed_start+attempt
        env = GuidedLessonEnv(stage=stage)
        trajectory = []
        try:
            obs, _ = env.reset(seed=seed)
            while True:
                action = _demonstration_action(env)
                next_obs, reward, terminated, truncated, info = env.step(action)
                trajectory.append({'obs': obs.copy(), 'action': action.copy(),
                                   'next_obs': next_obs.copy(), 'reward': float(reward),
                                   'terminated': bool(terminated), 'truncated': bool(truncated),
                                   'info': info, 'seed': seed, 'stage': stage})
                obs = next_obs
                if terminated or truncated:
                    break
        finally:
            env.close()
        attempted_steps += len(trajectory)
        per_stage[str(stage)]['attempted'] += 1
        if info['outcome'] == 'passed' and info['passes'] == 1 and terminated and not truncated:
            transitions.extend(trajectory)
            successful_seeds.append(seed)
            per_stage[str(stage)]['successful'] += 1
            per_stage[str(stage)]['steps'] += len(trajectory)
        if progress:
            progress(attempt+1, episodes)
    minimum = math.ceil(episodes*min_success_rate)
    if len(successful_seeds) < minimum or any(v['successful'] == 0 for v in per_stage.values()):
        raise RuntimeError(f'Insufficient successful demonstrations: {len(successful_seeds)}/{episodes}; '
                           f'need at least {minimum} and one per stage. Per stage: {per_stage}')
    summary = {'guidance_version': GUIDANCE_VERSION, 'successful': len(successful_seeds),
               'attempted': episodes, 'steps': len(transitions), 'attempted_steps': attempted_steps,
               'seed_start': seed_start, 'seed_end': seed_start+episodes-1,
               'seeds': successful_seeds, 'per_stage': per_stage,
               'controller': 'privileged geometry teacher, training examples only'}
    return transitions, summary
