"""The whole MDP: the states, the actions, the reward, the discount, and the ends.

This file is the one place the learning problem is defined. sim.py is only the
physics that carries it out, and params.py holds the measurements of the real
setup. If a number changes what the agent is asked to do, it belongs here. If it
describes the world, it belongs in params.py.

The state, each step: eight numbers from the camera and one from the height
sensor. Seven camera numbers describe the hoop's ellipse. The eighth, `marker`,
says whether the front marker is in view: a disc on the noodle's front face that
the noodle hides from behind. A hoop looks the same from both sides, so without
it the policy cannot tell which way through is forwards. The height is the
Tello's downward time-of-flight reading, in metres. The policy sees 12 s of these
with the commands it gave (policy_input).

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
#                 + R_FOUND once per episode, the first time the front is reported,
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
# R_FOUND: in the 2026-10-09 flights a policy climbed blind, turning ~6 deg/s, to the
# height limit without ever seeing the hoop. Paid once per episode (sim.HoopEnv tracks
# it), on the first detector report with the front marker, so it cannot be farmed by
# losing and refinding the hoop. Small next to R_PASS, so it guides the search and
# nothing else: finding then crashing still nets R_FOUND - R_CRASH, worse than hovering.
R_FOUND = 5.0

GAMMA = 0.995   # a lap is ~100 steps, so the next pass must still be worth chasing


def reward(outcome: str, to_go_before: float, to_go_after: float, on_wall: bool,
           obs: np.ndarray, action: np.ndarray) -> float:
    """The reward for one step. sim.HoopEnv.step calls this and nothing else scores.

    outcome: "passed", "running", "found" (a running step on which the front is
        reported for the first time this episode), or a crash in CRASHES ("hit",
        "wrong_way", "roof", "side").
    to_go_before, to_go_after: metres left round the lap to the next pass, before
        and after the step (sim.HoopEnv._to_go).
    on_wall: the drone touched the min_altitude floor or the net this step.
    obs: the eight numbers the camera reports after the step (FIELDS).
    action: the command the policy just gave (ACTION_FIELDS), each in [-1, 1].
    """
    return (R_PROGRESS * (to_go_before - to_go_after)
            + R_PASS * (outcome == "passed")
            + R_FOUND * (outcome == "found")
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

# The policy sees the last HISTORY_STEPS steps, each one a camera frame, a height
# reading and the command that led to them, grouped into slots. SLOT_STEPS holds
# how many steps each slot covers, oldest first, so the newest slot stays last.
# Each slot gives, for its steps:
#   the camera numbers of the newest frame that saw the hoop, with `visible` as
#       the share of its steps that saw it (all 0 if none did),
#   the newest height reading,
#   the mean command.
# So the slots tile the window with nothing left out between them: every sighting
# and every command is in some slot, and the commands sum to how far the drone was
# told to fly since any sighting. That is what it has to go on while blind.
#
# The newest four slots are single steps: the commands still in flight at up to
# 4 steps of delay, and the frame-to-frame motion of the hoop to steer on. Older
# slots are 0.4 s each, evenly spaced. 12 s covers the teacher's longest blind
# stretch, from losing the hoop on the run in round to seeing it again in front
# (teacher2, 150 flights: median 9.2 s, 90th percentile 10.9 s). The old window
# was 6 s of frames and 0.3 s of commands, and 65% of the teacher2 students'
# crashes came after 6 s or more without the hoop, when its input never changed.
SLOT_STEPS = (4,) * 29 + (1, 1, 1, 1)
HISTORY_STEPS = sum(SLOT_STEPS)
N_SLOTS = len(SLOT_STEPS)
SLOT_DIM = OBS_DIM + 1 + ACTION_DIM   # camera, height, command
#
# Last comes the not-yet-through flag: 1 until the hoop has left the frame
# while big and centred, which is what flying at it looks like, then 0 until the
# front (the marker) is reported again, then 1. It used to stay 0 for good, but
# then circling at 0.6-0.8 m dropped it with no pass, and the 2026-10-09 flights
# (teacher2-level1) flew the after-pass turn-back from in front of the hoop and
# into the net. Re-armed by the front only, so the back seen just after a pass
# keeps it 0. It separates takeoff
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
POLICY_INPUT_DIM = N_SLOTS * SLOT_DIM + 1
VISIBLE = FIELDS.index("visible")

# The height reading: metres above the floor from the Tello's downward
# time-of-flight sensor (state field "tof", in cm). On the drone,
# labcheck.fly_policy drops its glitches (6553 with no return, single-packet
# jumps); sim.HoopEnv adds the offset, noise and coarse steps params.Altimeter
# allows for. It keeps the blind climb off the roof, which the camera cannot see.


def not_yet_through(unseen: bool, frames) -> bool:
    """The flag after the newest frame: 0 once the hoop leaves the frame having last
    been seen at least PASS_MAJOR across within PASS_U of the centre, and 1 again
    once the hoop is reported with its front marker."""
    before, now = frames[-2], frames[-1]
    if now[VISIBLE] and now[FIELDS.index("marker")]:
        return True
    flew_at_it = (before[VISIBLE] and not now[VISIBLE]
                  and before[FIELDS.index("major")] >= PASS_MAJOR and abs(before[FIELDS.index("u")]) < PASS_U)
    return bool(unseen) and not flew_at_it


def policy_input(frames, heights, actions, unseen: bool) -> np.ndarray:
    """What the policy sees, in the simulator and on the drone alike.

    frames, heights, actions: the last HISTORY_STEPS steps, newest last. Step i is
        the camera frame and height reading after it, and the command that led to
        them: actions[-1] is the command just sent, frames[-1] what came after.
    unseen: the hoop has not been flown at until it left the frame since its front
        was last seen (not_yet_through).
    """
    frames = np.asarray(frames, np.float32)[-HISTORY_STEPS:]
    heights = np.asarray(heights, np.float32)[-HISTORY_STEPS:]
    actions = np.asarray(actions, np.float32)[-HISTORY_STEPS:]
    slots, start = [], 0
    for width in SLOT_STEPS:
        f, a = frames[start:start + width], actions[start:start + width]
        seen = np.flatnonzero(f[:, VISIBLE])
        frame = f[seen[-1]].copy() if len(seen) else NOT_VISIBLE.copy()
        frame[VISIBLE] = len(seen) / width
        slots.append(np.concatenate([frame, heights[start + width - 1:start + width], a.mean(axis=0)]))
        start += width
    return np.concatenate([*slots, [1.0 if unseen else 0.0]]).astype(np.float32)


def split_input(x: np.ndarray):
    """The policy input taken apart: per slot, oldest first, the frames (N_SLOTS,
    OBS_DIM), heights (N_SLOTS,) and mean commands (N_SLOTS, ACTION_DIM), and the flag."""
    slots = np.asarray(x[:N_SLOTS * SLOT_DIM]).reshape(N_SLOTS, SLOT_DIM)
    return slots[:, :OBS_DIM], slots[:, OBS_DIM], slots[:, OBS_DIM + 1:], bool(x[-1] > 0.5)


def slot_of(lag: int) -> int:
    """The slot that holds the step `lag` steps back (0 is now)."""
    back = 0
    for i in range(N_SLOTS - 1, -1, -1):
        back += SLOT_STEPS[i]
        if lag < back:
            return i
    raise ValueError(f"lag {lag} is beyond the {HISTORY_STEPS}-step window")

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
