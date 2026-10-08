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

# Reward per step = R_PROGRESS * (to_go_before - to_go_after)
#                 + R_PASS on a pass, - R_CRASH on a crash,
#                 - R_TIME always, - R_WALL each step on the floor.
#
# Shaping: R_PROGRESS per metre flown along the route round the lap to the next
# pass (sim.HoopEnv._to_go), with no GAMMA. With GAMMA inside the difference
# (netcrash-routeshaping) the shaping was potential-based: holding still earned
# (1 - GAMMA) * R_PROGRESS * to_go a step, which over the bootstrapped horizon
# repaid exactly the progress made, so parking anywhere was worth the same and
# every seed parked 0.5-0.8 m in front of the hoop. Without GAMMA, holding still
# earns nothing and progress made is kept, so parking closer to the next pass is
# worth R_PROGRESS per metre more. Any loop still sums to zero, so it cannot be
# farmed. The pass takes the route back: to_go jumps ~2.2 m there, so the pass
# step nets ~+31 and the way round pays the 8.8 back.
#
# The policy cannot see the clock and timeouts bootstrap, so hovering anywhere is
# worth -R_TIME / (1 - GAMMA) = -20. A crash is -30, 10 worse than hovering at the
# same spot (progress made is kept either way), so crashing is never a cheap way
# out. Pass then crash nets ~+1, pass then hover ~+11: no pass-and-die.
# Raising the crash-vs-hover gap stopped learning before the first pass:
# R_CRASH 50 (crash50-nogamma-progress) and R_CRASH 50 with R_TIME 0.15
# (time15-crash50-nogamma, cancelled at 450k) both settled into hovering.
#
# R_HIDDEN is gone: with every parking spot equal it alone ranked them, and it
# picked spots where the marker stays in view, outside ~0.45 m. The last
# 0.35-0.45 m of every pass is blind, so it charged the pass itself.
R_PASS = 40.0
R_CRASH = 30.0      # R_PASS 90 vs 15 taught pass-and-die
R_PROGRESS = 4.0    # per metre flown along the route
R_TIME = 0.1        # R_TIME / (1 - GAMMA) must stay below R_CRASH
R_WALL = 1.0        # each step held at min_altitude, which is no crash; the net is one.
                    # 5 cost an eighth of a pass a step, and low hoops are 10 cm above it

GAMMA = 0.995   # a lap is ~100 steps, so the next pass must still be worth chasing


def reward(outcome: str, to_go_before: float, to_go_after: float, on_wall: bool,
           obs: np.ndarray, action: np.ndarray) -> float:
    """The reward for one step. sim.HoopEnv.step calls this and nothing else scores.

    outcome: "passed", "running", or a crash in CRASHES ("hit", "wrong_way",
        "roof", "side").
    to_go_before, to_go_after: metres left round the lap to the next pass, before
        and after the step (sim.HoopEnv._to_go).
    on_wall: the drone touched the min_altitude floor or the net this step.
    obs: the eight numbers the camera reports after the step (FIELDS).
    action: the command the policy just gave (ACTION_FIELDS), each in [-1, 1].
    """
    return (R_PROGRESS * (to_go_before - to_go_after)
            + R_PASS * (outcome == "passed")
            - R_CRASH * (outcome in CRASHES)
            - R_TIME
            - R_WALL * on_wall)

# "roof" and "side": the drone touched the net, above or beside it. The net is
# not in the state, but the hoop is, and the hoop sits well in from it. Both
# faces cost the same R_CRASH. They carry separate names only so the logs say
# which one the drone met: the old single "net" hid whether a run died on the
# ceiling in the blind climb or in the walls on the way round.
CRASHES = ("hit", "wrong_way", "roof", "side")
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
# Close lags give velocity, far lags hold the last sighting through blind
# stretches. On a lap the hoop leaves view 0.8 s before the pass and is back
# about 3.7 s after it (seenflag-s1-level1: median 45 steps, max 53), so the
# window spans 60 lags: the old 24 left the go-round with no hoop frame for
# ~2 s of every lap. Striding covers 6 s with 12 frames instead of 61.
OBS_LAGS = (60, 50, 40, 32, 24, 16, 12, 8, 4, 2, 1, 0)
OBS_STACK = len(OBS_LAGS)
FRAME_HISTORY = max(OBS_LAGS) + 1
ACTION_HISTORY = 3
#
# Last comes the not-yet-through flag: 1 until the hoop has left the frame
# while big and centred, which is what flying at it looks like, then 0 for good. It separates takeoff
# (net 0.3 m behind) from mid-lap (hoop behind, just passed): without it the
# lap's turn-back reversed into the net at takeoff. It used to drop at the first
# sighting (seenflag-s1-level1), but then the first sighting after a blind climb
# looked like finding the hoop again after a pass, and every 2026-10-06 flight
# flew the lap's go-round (strafe left, turn right) from the corner, lost the
# hoop and backed into the net. A size alone does not work: with the detector
# needing ~60% of the rim in frame, a hoop approached level is last reported at
# 0.57-1.08 across, depending on the detector and the height. But on every
# straight approach it leaves the frame at least PASS_MAJOR across and within
# PASS_U of the centre, while the flights' losses before any pass were off the
# top or side at 0.42 or less. It reads 1 at takeoff, not 0, so that a policy
# grown from an 84-input one with a zero weight on it keeps the lap's inputs
# unchanged.
PASS_MAJOR = 0.55   # ~0.8 m from a 0.45 m hoop
PASS_U = 0.5
POLICY_INPUT_DIM = OBS_STACK * OBS_DIM + ACTION_HISTORY * ACTION_DIM + 1
VISIBLE = FIELDS.index("visible")


def not_yet_through(unseen: bool, frames) -> bool:
    """The flag after the newest frame: 0 for good once the hoop leaves the frame
    having last been seen at least PASS_MAJOR across within PASS_U of the centre."""
    before, now = frames[-2], frames[-1]
    flew_at_it = (before[VISIBLE] and not now[VISIBLE]
                  and before[FIELDS.index("major")] >= PASS_MAJOR and abs(before[FIELDS.index("u")]) < PASS_U)
    return bool(unseen) and not flew_at_it


def policy_input(frames, actions, unseen: bool) -> np.ndarray:
    """What the policy sees, in the simulator and on the drone alike.

    frames: the last FRAME_HISTORY observations, newest last.
    actions: the last ACTION_HISTORY actions, newest last.
    unseen: the hoop has not yet been flown at until it left the frame (not_yet_through).
    """
    return np.concatenate([*(frames[-1 - lag] for lag in OBS_LAGS), *actions,
                           [1.0 if unseen else 0.0]]).astype(np.float32)

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
