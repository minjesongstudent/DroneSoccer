"""The seven numbers: the only thing the policy sees.

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

FIELDS = ("u", "v", "major", "ratio", "sin_2t", "cos_2t", "visible")
OBS_DIM = len(FIELDS)
NOT_VISIBLE = np.zeros(OBS_DIM, dtype=np.float32)

# Same order and signs as djitellopy's send_rc_control(left_right, forward_backward,
# up_down, yaw), each in [-1, 1]. fly.py only has to scale, never reorder.
ACTION_FIELDS = ("right", "forward", "up", "yaw_cw")
ACTION_DIM = len(ACTION_FIELDS)

# The policy sees the last 3 observations and its own last 2 actions.
OBS_STACK = 3
ACTION_HISTORY = 2
POLICY_INPUT_DIM = OBS_STACK * OBS_DIM + ACTION_HISTORY * ACTION_DIM

MIN_POINTS = 5     # fitEllipse needs at least five
MIN_RATIO = 0.2    # flatter than this is a hoop seen near edge-on: treat as not found
CLIP = 3.0         # bound u, v, major when a partial arc puts the centre off-frame
_FIT_SCALE = 1000.0  # fitEllipse is tuned for pixel-scale input


def observe(points: np.ndarray, half_extent: tuple[float, float]) -> np.ndarray:
    """Seven numbers from hoop edge points.

    points: (N, 2) normalised image coordinates of the hoop's outer edge.
    half_extent: half the frame width and height in normalised coordinates,
        so that u and v are +-1 at the frame edges.
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
    ], dtype=np.float32)
    return obs
