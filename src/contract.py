from __future__ import annotations

import cv2
import numpy as np

R_PASS = 40.0
R_CRASH = 35.0
R_PROGRESS = 4.0
R_TIME = 0.1
R_WALL = 1.0
R_FOUND = 5.0
R_CENTRE = 20.0

GAMMA = 0.995


def reward(outcome: str, to_go_before: float, to_go_after: float, on_wall: bool,
           obs: np.ndarray, action: np.ndarray, centred: float = 0.0) -> float:
    return (R_PROGRESS * (to_go_before - to_go_after)
            + (R_PASS + R_CENTRE * centred) * (outcome == "passed")
            + R_FOUND * (outcome == "found")
            - R_CRASH * (outcome in CRASHES)
            - R_TIME
            - R_WALL * on_wall)

CRASHES = ("hit", "wrong_way", "roof", "side")
LAPS_MAX_STEPS = 600

FIELDS = ("u", "v", "major", "ratio", "sin_2t", "cos_2t", "visible", "marker")
OBS_DIM = len(FIELDS)
NOT_VISIBLE = np.zeros(OBS_DIM, dtype=np.float32)

ACTION_FIELDS = ("right", "forward", "up", "yaw_cw")
ACTION_DIM = len(ACTION_FIELDS)

SLOT_STEPS = (4,) * 29 + (1, 1, 1, 1)
HISTORY_STEPS = sum(SLOT_STEPS)
N_SLOTS = len(SLOT_STEPS)
MOTION_FIELDS = ("speed", "sin_turned", "cos_turned")
MOTION_DIM = len(MOTION_FIELDS)
NO_MOTION = np.array([0.0, 0.0, 1.0], np.float32)
SLOT_DIM = OBS_DIM + 1 + MOTION_DIM + ACTION_DIM
PASS_MAJOR = 0.55
PASS_U = 0.5
POLICY_INPUT_DIM = N_SLOTS * SLOT_DIM + 1
VISIBLE = FIELDS.index("visible")



def not_yet_through(unseen: bool, frames) -> bool:
    before, now = frames[-2], frames[-1]
    if now[VISIBLE] and now[FIELDS.index("marker")]:
        return True
    flew_at_it = (before[VISIBLE] and not now[VISIBLE]
                  and before[FIELDS.index("major")] >= PASS_MAJOR and abs(before[FIELDS.index("u")]) < PASS_U)
    return bool(unseen) and not flew_at_it


def motion_reading(speed: float, turned_rad: float) -> np.ndarray:
    return np.array([speed, np.sin(turned_rad), np.cos(turned_rad)], np.float32)


def policy_input(frames, heights, motion, actions, unseen: bool) -> np.ndarray:
    frames = np.asarray(frames, np.float32)[-HISTORY_STEPS:]
    heights = np.asarray(heights, np.float32)[-HISTORY_STEPS:]
    motion = np.asarray(motion, np.float32).reshape(-1, MOTION_DIM)[-HISTORY_STEPS:]
    actions = np.asarray(actions, np.float32)[-HISTORY_STEPS:]
    slots, start = [], 0
    for width in SLOT_STEPS:
        f, a = frames[start:start + width], actions[start:start + width]
        seen = np.flatnonzero(f[:, VISIBLE])
        frame = f[seen[-1]].copy() if len(seen) else NOT_VISIBLE.copy()
        frame[VISIBLE] = len(seen) / width
        end = start + width - 1
        slots.append(np.concatenate([frame, heights[end:end + 1], motion[end], a.mean(axis=0)]))
        start += width
    return np.concatenate([*slots, [1.0 if unseen else 0.0]]).astype(np.float32)


def split_input(x: np.ndarray):
    slots = np.asarray(x[:N_SLOTS * SLOT_DIM]).reshape(N_SLOTS, SLOT_DIM)
    return slots[:, :OBS_DIM], slots[:, OBS_DIM], slots[:, OBS_DIM + 1 + MOTION_DIM:], bool(x[-1] > 0.5)


SLOTS_V1_DIM = N_SLOTS * (SLOT_DIM - MOTION_DIM) + 1


def slots_v1(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, np.float32)
    slots = x[:N_SLOTS * SLOT_DIM].reshape(N_SLOTS, SLOT_DIM)
    keep = np.r_[0:OBS_DIM + 1, OBS_DIM + 1 + MOTION_DIM:SLOT_DIM]
    return np.append(slots[:, keep].ravel(), x[N_SLOTS * SLOT_DIM]).astype(np.float32)


def for_model(size: int):
    return {POLICY_INPUT_DIM: lambda x: x, SLOTS_V1_DIM: slots_v1}.get(size)


def split_motion(x: np.ndarray) -> np.ndarray:
    slots = np.asarray(x[:N_SLOTS * SLOT_DIM]).reshape(N_SLOTS, SLOT_DIM)
    return slots[:, OBS_DIM + 1:OBS_DIM + 1 + MOTION_DIM]


def slot_of(lag: int) -> int:
    back = 0
    for i in range(N_SLOTS - 1, -1, -1):
        back += SLOT_STEPS[i]
        if lag < back:
            return i
    raise ValueError(f"lag {lag} is beyond the {HISTORY_STEPS}-step window")

MIN_ARC_FRACTION = 0.25
MIN_POINTS = 5
MIN_RATIO = 0.2
CLIP = 3.0
_FIT_SCALE = 1000.0


def observe(points: np.ndarray, half_extent: tuple[float, float],
            marker: bool = False) -> np.ndarray:
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
