"""The whole MDP: the states, the actions, the reward, the discount, and the ends.

This file is the one place the learning problem is defined. sim.py is only the
physics that carries it out, and params.py holds the measurements of the real
setup. If a number changes what the agent is asked to do, it belongs here. If it
describes the world, it belongs in params.py.

The state: eight numbers, and nothing else. Seven describe the hoop's ellipse.
The eighth, `marker`, says whether the front marker is in view: a disc on the
noodle's front face that the noodle hides from behind. A hoop looks the same from
both sides, so without it the policy cannot tell which way through is forwards.

The detector (real camera) and the simulator (projected geometry) both produce
points on the hoop's outer edge in normalised image coordinates, then call
`observe`. Sharing this one function removes a whole class of sim-to-real
mismatch. Changing anything here means retraining, so treat it as frozen.

Normalised image coordinates are x/z and y/z in the camera frame: x right,
y down, z forward. The detector gets them from `cv2.undistortPoints(pts, K, dist)`.
"""
from __future__ import annotations

import cv2
import numpy as np

# Reward per step = R_PASS on a pass, - R_CRASH on a crash,
#                 - R_WALL each step on the floor, - R_TIME always.
#
# No shaping: with free hovering, the route-shaped run settled on hovering.
# The policy cannot see the clock and timeouts bootstrap, so hovering is worth
# -R_TIME / (1 - GAMMA) = -20. That beats crashing (-30), so the drone is never
# paid to crash, but trying beats hovering once a lap passes ~20% of the time.
R_PASS = 40.0
R_CRASH = 30.0      # R_PASS 90 vs 15 taught pass-and-die
R_WALL = 5.0        # each step held at min_altitude; the net is a crash
R_TIME = 0.1        # keep R_TIME / (1 - GAMMA) below R_CRASH, or crashing beats hovering

GAMMA = 0.995   # a lap is ~100 steps, so the next pass must still be worth chasing


def reward(outcome: str, to_go_before: float, to_go_after: float, on_wall: bool,
           obs: np.ndarray, action: np.ndarray) -> float:
    """The reward for one step. sim.HoopEnv.step calls this and nothing else scores.

    outcome: "passed", "running", or a crash in CRASHES ("hit", "wrong_way", "net").
    to_go_before, to_go_after: metres left round the lap to the next pass, before
        and after the step (sim.HoopEnv._to_go). Unused while there is no shaping.
    on_wall: the drone touched the min_altitude floor or the net this step.
    obs: the eight numbers the camera reports after the step (FIELDS).
    action: the command the policy just gave (ACTION_FIELDS), each in [-1, 1].
    """
    return (R_PASS * (outcome == "passed")
            - R_CRASH * (outcome in CRASHES)
            - R_TIME
            - R_WALL * on_wall)

# "net": the drone touched the net. The net is not in the state, but the hoop
# is, and the hoop sits well in from it.
CRASHES = ("hit", "wrong_way", "net")
LAPS_MAX_STEPS = 600   # 60 s cut-off, not a failure

FIELDS = ("u", "v", "major", "ratio", "sin_2t", "cos_2t", "visible", "marker")
OBS_DIM = len(FIELDS)
NOT_VISIBLE = np.zeros(OBS_DIM, dtype=np.float32)

# Same order and signs as djitellopy's send_rc_control(left_right, forward_backward,
# up_down, yaw), each in [-1, 1]. fly.py only has to scale, never reorder.
ACTION_FIELDS = ("right", "forward", "up", "yaw_cw")
ACTION_DIM = len(ACTION_FIELDS)

# The policy sees a strided window of past observations and its own last 3
# actions, enough to cover every command still in flight at up to 3 steps of
# delay. OBS_LAGS holds the lags in control steps, oldest first, so the newest
# frame stays last and frames[-1] still means "now".
#
# The hoop leaves the frame about 0.40 m out, and the drone needs about 0.5 m to
# carry its body through. At 0.2 m/s that is 25 steps at 10 Hz, so the window
# spans 24 lags. Close lags give velocity, far lags hold the last sighting
# through the blind final approach. Striding covers 2.5 s with 9 frames instead
# of 25, which keeps POLICY_INPUT_DIM at 84 rather than 212.
OBS_LAGS = (24, 20, 16, 12, 8, 4, 2, 1, 0)
OBS_STACK = len(OBS_LAGS)
FRAME_HISTORY = max(OBS_LAGS) + 1
ACTION_HISTORY = 3
POLICY_INPUT_DIM = OBS_STACK * OBS_DIM + ACTION_HISTORY * ACTION_DIM

MIN_ARC_FRACTION = 0.25  # share of the hoop's edge the detector needs in frame
MIN_POINTS = 5     # fitEllipse needs at least five
MIN_RATIO = 0.2    # UNMEASURED: flattest ellipse the detector still finds (edge-on hoop), B3
CLIP = 3.0         # bound u, v, major when a partial arc puts the centre off-frame
_FIT_SCALE = 1000.0  # fitEllipse is tuned for pixel-scale input


def observe(points: np.ndarray, half_extent: tuple[float, float],
            marker: bool = False) -> np.ndarray:
    """Eight numbers from hoop edge points and whether the front marker is seen.

    points: (N, 2) normalised image coordinates of the hoop's outer edge.
    half_extent: half the frame width and height in normalised coordinates,
        so that u and v are +-1 at the frame edges.
    marker: the front marker is in view. The detector looks for it next to the
        fitted ellipse, so without a hoop it reads 0 like everything else.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if len(points) < MIN_POINTS:
        return NOT_VISIBLE.copy()

    (xc, yc), (w, h), angle = cv2.fitEllipse((points * _FIT_SCALE).astype(np.float32))
    if w >= h:
        major, minor, theta = w, h, angle
    else:
        major, minor, theta = h, w, angle + 90.0
    if not (np.isfinite([xc, yc, major, minor]).all() and major > 0):
        return NOT_VISIBLE.copy()

    ratio = minor / major
    if ratio < MIN_RATIO:
        return NOT_VISIBLE.copy()

    # An ellipse at theta and theta + 180 is the same ellipse, so encode 2*theta.
    # A near-circle has no meaningful lean, so scale it down as ratio -> 1.
    two_t = np.deg2rad(2.0 * theta)
    lean = 1.0 - ratio
    hx, hy = half_extent
    obs = np.array([
        np.clip(xc / _FIT_SCALE / hx, -CLIP, CLIP),
        np.clip(yc / _FIT_SCALE / hy, -CLIP, CLIP),
        np.clip(major / _FIT_SCALE, 0.0, CLIP),
        ratio,
        lean * np.sin(two_t),
        lean * np.cos(two_t),
        1.0,
        float(marker),
    ], dtype=np.float32)
    return obs
