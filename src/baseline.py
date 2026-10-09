"""Hand-written controllers on the same eight numbers the policy sees.

They prove the mission is solvable and are the comparison line for the RL results.
    act     a proportional controller that flies at the hoop and through it.
    Lapper  act for each pass, lined up with the hoop's axis first, plus a
            scripted search and go-round.
Run: python -m src.evaluate --baseline
"""
from __future__ import annotations

import numpy as np

from src import contract
from src.params import Camera, Drone

K_CENTRE = 1.5     # command per unit of u or v
COMMIT = 0.5       # forward command above which a lost hoop means "keep going"
# The camera looks down by pitch_deg, so a hoop level with the drone sits above
# the image centre. Steering v to 0 instead flies ~tan(12.8 deg) high: 23 cm at 1 m,
# a whole hoop radius, into the top of the rim.
V_LEVEL = -np.tan(np.deg2rad(Camera().pitch_deg)) / Camera().half_extent[1]

# Lining up. Flown straight at the centre from the side, the drone crosses the
# ring at an angle and clips the noodle. A hoop seen off its axis is an ellipse,
# so the Lapper strafes until it is round, turning to keep it centred, which
# orbits the hoop. The ellipse does not say which side the drone is on, so it
# strafes one way and turns back if the ellipse gets flatter.
ROUND = 0.95       # ratio above which the hoop counts as faced: cos 18 degrees
K_ALIGN = 0.8      # strafe command when far off the axis
K_YAW = 1.0        # turn command per unit of u while lining up
# The real ratio jitters by ~0.1 per frame, so compare averages of recent frames
# and, after a flip, hold the new way for a while. Lags 0-3 are single-step slots
# of the policy input; lags 4 and 8 read the 4-step slots that start there, whose
# frame is the newest sighting in them, so lag 4 itself whenever it saw the hoop.
FLATTER = 0.02     # drop in mean ratio, NOW_LAGS against THEN_LAGS, that flips the strafe
NOW_LAGS = (0, 1, 2)
THEN_LAGS = (4, 8)
# Only fly at a hoop whose front marker was seen within MARKER_LAG steps: 6 s,
# as far back as the old 12-frame history reached.
MARKER_LAG = 60
HOLD_STEPS = 10    # steps after a flip before the next may happen

# Lapper. After a pass the drone is only ~0.3 m past the hoop, too close to see
# it, so the way round is a fixed box, flown blind on timing: sidestep, reverse
# past the hoop's plane, sidestep back, turn to face it. Right is a guess; the
# baseline cannot know which side has room, so some laps end in the net.
SEARCH_YAW = 0.5       # turning speed while the hoop is out of view
# The camera looks down, so a hoop above the drone stays out of view however it
# turns. After each full turn without a sighting, climb while turning.
TURN_STEPS = round(360 / (SEARCH_YAW * Drone().max_yaw_rate_deg) / Drone().dt)   # 120
CLIMB_STEPS = 10       # 0.4 m in the sim; ~0.26 m on the Tello, whose up is slower
MAX_CLIMBS = 2         # then keep turning: +0.8 m sim, ~0.5 m real, under labcheck's 170 cm
GO_ROUND = [           # (steps, action): ~0.6 m right, ~0.8 m back, ~0.6 m left, ~180 degrees
    (15, (0.0, 1.0, 0.0, 0.0)),    # finish carrying the body through
    (18, (1.0, 0.0, 0.0, 0.0)),
    (22, (0.0, -1.0, 0.0, 0.0)),
    (18, (-1.0, 0.0, 0.0, 0.0)),
    (35, (0.0, 0.0, 0.0, 1.0)),
]


def _split(policy_input: np.ndarray):
    """Per slot, oldest first, the frames and mean commands. The newest slot is one
    step, so frames[-1] is the newest frame and actions[-1] the last command."""
    frames, _heights, actions, _unseen = contract.split_input(policy_input)
    return frames, actions


def act(policy_input: np.ndarray) -> np.ndarray:
    frames, actions = _split(policy_input)
    u, v, _major, _ratio, _s, _c, visible, _marker = frames[-1]
    last_forward = actions[-1][1]

    if not visible:
        # The hoop leaves the frame in the last ~40 cm. Keep going if we were committed.
        forward = last_forward if last_forward > COMMIT else 0.0
        return np.array([0.0, forward, 0.0, 0.0], dtype=np.float32)

    right = np.clip(K_CENTRE * u, -1, 1)
    up = np.clip(-K_CENTRE * (v - V_LEVEL), -1, 1)   # v is positive downwards in the image
    forward = np.clip(1.0 - 2.0 * max(abs(u), abs(v - V_LEVEL)), 0.0, 1.0)
    return np.array([right, forward, up, 0.0], dtype=np.float32)


class Lapper:
    """Search, pass with `act`, fly the GO_ROUND box, repeat.

    Needs memory across a lap, so it is a small state machine rather than a function.
    Call reset() at the start of every episode.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.script = []   # remaining (steps, action) of the go-round, empty when approaching
        self.strafe = 1.0  # which way lining up goes: +1 right, -1 left
        self.since_flip = HOLD_STEPS
        self.searching = 0  # steps since the hoop was last seen, outside the go-round

    def __call__(self, policy_input: np.ndarray) -> np.ndarray:
        frames, actions = _split(policy_input)
        visible = frames[-1][6]

        if not self.script and not visible and actions[-1][1] > COMMIT:
            self.script = [list(leg) for leg in GO_ROUND]   # committed and blind: the pass
        if self.script:
            leg = self.script[0]
            leg[0] -= 1
            if leg[0] <= 0:
                self.script.pop(0)
            return np.array(leg[1], dtype=np.float32)
        if not visible:
            return self._search()
        self.searching = 0
        action = self._line_up(frames, actions, act(policy_input))
        # Only fly at a hoop whose front marker was seen in the last MARKER_LAG steps.
        # Blue junk has no marker, and nor does the hoop's back, which is a wrong way.
        if not frames[contract.slot_of(MARKER_LAG):, 7].any():
            action[1] = 0.0
        return action

    def _search(self) -> np.ndarray:
        turns, into = divmod(self.searching, TURN_STEPS + CLIMB_STEPS)
        self.searching += 1
        climbing = 1 <= turns + 1 <= MAX_CLIMBS and into >= TURN_STEPS
        return np.array([0.0, 0.0, 1.0 if climbing else 0.0, SEARCH_YAW], dtype=np.float32)

    def _line_up(self, frames, actions, action):
        u, ratio = frames[-1][0], frames[-1][3]
        now, then = ([frames[contract.slot_of(lag)] for lag in lags] for lags in (NOW_LAGS, THEN_LAGS))
        self.since_flip += 1
        if (all(f[6] for f in now + then) and self.since_flip >= HOLD_STEPS
                and np.mean([f[3] for f in now]) < np.mean([f[3] for f in then]) - FLATTER):
            self.strafe, self.since_flip = -self.strafe, 0
        off = np.clip((ROUND - ratio) / (1 - ROUND), 0.0, 1.0)   # 0 once round
        action[0] = np.clip(self.strafe * K_ALIGN * off + 0.3 * K_CENTRE * u, -1, 1)
        action[1] *= 1 - off   # hold off going forward until lined up
        action[3] = np.clip(K_YAW * u, -1, 1)
        return action
