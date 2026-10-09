"""See the hoop in a camera frame: the B3 detector.

    python -m src.detect photo.jpg [more.jpg ...]   draw the detection, print the eight numbers
    python -m src.detect video.mp4                  same, frame by frame (Esc to stop)

The hoop is found by colour: the blue noodle against yellow walls and black net.
Blue alone also finds door glass, straps and stickers, so a fit only counts as the
hoop if it looks like the noodle in this cage: a band of blue along the edge, an
open middle, and yellow wall, white pipe or black net just outside. Live, it must
also hold its place and size over consecutive frames (confirm=).
The front marker is found by contrast: a white disc with a black cross, at the
hoop's 6 o'clock. Colour would also find the white cage pipes, so the marker test
asks for a bright round blob with dark strokes inside, next to the fitted ellipse.

Both feed contract.observe, the same function the simulator calls.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
# Deterministic bounded CPU image processing; default native pool failed on the PC.
cv2.setNumThreads(1)
import numpy as np

from src import contract
from src.camera_calibration import CameraCalibration, resolve_calibration
from src.params import CALIB_PATH, Camera

WORK_WIDTH = 480          # DESIGN: detect at 480 px wide; the net speckles less
HOOP_HSV_LO = (95, 120, 60)    # UNMEASURED: blue noodle, from one photo. B3 tunes on cage footage
HOOP_HSV_HI = (125, 255, 255)
MIN_ARC_AREA = 40         # px^2 at WORK_WIDTH: smaller blobs are net speckle
RANSAC_TRIES = 400        # DESIGN: 5-point samples per frame
MIN_RING_PX = 20          # smaller ellipses are knots and blobs, not the ring
COVER_BINS = 36
RANSAC_TOL_PX = 2.0       # inlier band at WORK_WIDTH
EDGE_STEP_PX = 3          # look this far either side of an edge point

# Ring test. The noodle is 6.5 cm thick on a 22.5 cm outer radius, so it fills
# radii 0.71-1.0 of the outer edge, and the middle is open.
RING_SAMPLES = 72
RING_INSIDE = 0.88        # mid-noodle
RING_OUTSIDE = 1.12       # just outside the noodle
RING_HOLE = (0.2, 0.45)   # well inside the hole
YELLOW_HUE = (15, 40)     # UNMEASURED: the cage wall, from cage footage 2026-10-01
YELLOW_MIN_S = 80
BLACK_MAX_V = 70          # the net, and shadow
WHITE_MIN_V = 150         # UNMEASURED: from one photo; B3 tunes on cage footage
WHITE_MAX_S = 60
WARM_WHITE_MAX_S = 120   # Warm-lit disc in 2026-10-06 capture: S=103-113.
                        # Fallback still requires geometry, position and dark cross.
DARK_MAX_V = 90
# A fit is the hoop only above all three. From cage footage 2026-10-01: front views
# score band >= 0.78, hole 0, backdrop >= 0.74; door glass, straps and a blue
# sticker failed at least one.
MIN_BAND = 0.55
MAX_HOLE = 0.3
MIN_BACKDROP = 0.6
CONFIRM_FRAMES = 2        # live: consecutive agreeing fits before the hoop counts
AGREE_SHIFT = 0.25        # centre moves less than this * major between agreeing fits
AGREE_SIZE = 1.4          # and major changes by less than this factor
MARKER_MIN_FILL = 0.80    # blob area / its fitted ellipse's area: rejects ragged blobs
MARKER_MIN_RATIO = 0.4    # narrowest disc: cos 60 deg (marker_max_angle_deg) is 0.5
MARKER_DARK = (0.06, 0.6) # share of the disc that is dark: a cross, not blank, not a hole
MARKER_SIZE = (0.06, 0.45)   # disc diameter / hoop major axis
MARKER_REACH = 0.35       # disc centre within this * major of the hoop's lowest point


@dataclass
class Detection:
    obs: np.ndarray                    # the eight numbers (contract.FIELDS)
    edge_px: Optional[np.ndarray]      # outer-edge points used for the fit, full-frame pixels
    ellipse_px: Optional[tuple]        # cv2 ellipse in full-frame pixels, for drawing
    marker_px: Optional[tuple]         # (x, y, radius) in full-frame pixels
    rejected: str = ""                 # why a fit was not taken as the hoop, for drawing


def load_intrinsics(width: int, height: int, calibration_path: str | Path | None = None
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Validated calibration for this frame size, or labeled team defaults."""
    return resolve_calibration(calibration_path).intrinsics(width, height)


class HoopDetector:
    """confirm: consecutive frames whose fits must agree before the hoop counts.
    1 for still photos; CONFIRM_FRAMES on live video, where junk flickers."""

    def __init__(self, width: int = Camera.width, height: int = Camera.height, confirm: int = 1,
                 calibration_path: str | Path | None = None, *,
                 calibration: CameraCalibration | None = None):
        if calibration is not None and calibration_path is not None:
            raise ValueError("Supply calibration or calibration_path, not both")
        self.calibration = calibration if calibration is not None else resolve_calibration(calibration_path)
        self.calibration_status = self.calibration.status()
        self.size = (width, height)
        self.confirm = confirm
        self._streak, self._last_fit = 0, None
        self.K, self.dist = self.calibration.intrinsics(width, height)
        self.scale = WORK_WIDTH / width
        # Normalised half extent of the frame, so u, v are +-1 at its edges (as in sim).
        self.half_extent = (width / (2 * self.K[0, 0]), height / (2 * self.K[1, 1]))
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._rng = np.random.default_rng(0)
        self._edge_hint = None

    def __call__(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr.shape[1::-1] != self.size:
            self.__init__(*frame_bgr.shape[1::-1], confirm=self.confirm, calibration=self.calibration)
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        mask = self._mask(small)
        edge = self._outer_edge(mask, small)
        if edge is None:
            self._streak = 0
            self._edge_hint = None
            return Detection(contract.NOT_VISIBLE.copy(), None, None, None)
        ellipse = cv2.fitEllipse(edge.astype(np.float32))
        (cx, cy), (w, h), ang = ellipse
        s = 1 / self.scale
        ellipse_px = ((cx * s, cy * s), (w * s, h * s), ang)
        edge_px = edge / self.scale

        why = self._reject(mask, small, ellipse)
        if why:
            self._streak = 0
            self._edge_hint = None
            return Detection(contract.NOT_VISIBLE.copy(), edge_px, ellipse_px, None, why)

        marker = self._marker(small, ellipse)
        norm = cv2.undistortPoints(edge_px.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist).reshape(-1, 2)
        obs = contract.observe(norm, self.half_extent, marker=marker is not None)
        if not obs[6]:
            self._streak = 0
            self._edge_hint = None
            return Detection(obs, edge_px, ellipse_px, None, "too flat")
        self._edge_hint = ellipse
        if self._streak < self.confirm:
            return Detection(contract.NOT_VISIBLE.copy(), edge_px, ellipse_px, None,
                             f"confirming {self._streak}/{self.confirm}")
        return Detection(obs, edge_px, ellipse_px,
                         None if marker is None else tuple(v * s for v in marker))

    def _reject(self, mask, small, ellipse) -> str:
        """Why this fit is not the hoop, or "" if it is. Also counts agreeing frames."""
        band, hole, backdrop = self.last_scores = ring_scores(mask, small, ellipse)
        if band < MIN_BAND:
            return f"no ring {band:.2f}"
        if hole > MAX_HOLE:
            return f"solid {hole:.2f}"
        if backdrop < MIN_BACKDROP:
            return f"backdrop {backdrop:.2f}"
        (cx, cy), (w, h), _ = ellipse
        major = max(w, h)
        last = self._last_fit
        agrees = (last is not None
                  and np.hypot(cx - last[0], cy - last[1]) < AGREE_SHIFT * major
                  and 1 / AGREE_SIZE < major / last[2] < AGREE_SIZE)
        self._streak = self._streak + 1 if agrees else 1
        self._last_fit = (cx, cy, major)
        return ""

    def _mask(self, small: np.ndarray) -> np.ndarray:
        """Noodle-blue pixels, at WORK_WIDTH scale."""
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), HOOP_HSV_LO, HOOP_HSV_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)

    def _outer_edge(self, mask: np.ndarray, small: np.ndarray) -> Optional[np.ndarray]:
        """Points on the noodle's outer edge, at WORK_WIDTH scale."""

        # Every arc above the area floor, not the largest: the net, the pole and the
        # marker cut the ring into pieces that all lie on the same ellipse.
        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        arcs = [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) >= MIN_ARC_AREA]
        if not arcs:
            return None
        pts = np.concatenate(arcs).astype(np.float64)
        if len(pts) < contract.MIN_POINTS:
            return None

        # Use the previous accepted ellipse only to select fresh outer-edge pixels.
        # Refit and rerun every ring test on this image; never emit a cached detection.
        # If the current image does not support it, use the ordinary global search.
        if self._edge_hint is not None:
            near = pts[_inliers(pts, self._edge_hint, 4 * RANSAC_TOL_PX)]
            if len(near) >= contract.MIN_POINTS:
                tracked = _refine(near, self._edge_hint)
                if tracked is not None:
                    old_c, old_size, _ = self._edge_hint
                    new_c, new_size, _ = tracked
                    shift = np.linalg.norm(np.asarray(new_c)-old_c)
                    ratio = max(new_size)/max(max(old_size), 1e-6)
                    band, hole, backdrop = ring_scores(mask, small, tracked)
                    keep = near[_inliers(near, tracked)]
                    if (shift < AGREE_SHIFT*max(old_size)
                            and 1/AGREE_SIZE < ratio < AGREE_SIZE
                            and band >= MIN_BAND and hole <= MAX_HOLE and backdrop >= MIN_BACKDROP
                            and _coverage(keep, tracked) >= contract.MIN_ARC_FRACTION*COVER_BINS
                            and len(keep) >= contract.MIN_POINTS):
                        return keep

        # The straps are blue too and join the ring into one blob, so RANSAC finds
        # the ellipse whose points reach furthest round it: roughly the ring, though
        # it may mix the noodle's outer and inner edges. Its centre is enough to tell
        # them apart. A point on the outer edge has noodle just inside it and
        # background just outside. Fit only those, as the simulator projects the
        # outer edge.
        seed = _ransac(pts, self._rng)
        if seed is None:
            return None
        (cx, cy), _, _ = seed
        d = pts - (cx, cy)
        d /= np.maximum(np.hypot(d[:, 0], d[:, 1]), 1e-6)[:, None]
        hgt, wid = mask.shape
        def at(q):
            q = np.rint(q).astype(int)
            ok = (q[:, 0] >= 0) & (q[:, 0] < wid) & (q[:, 1] >= 0) & (q[:, 1] < hgt)
            v = np.zeros(len(q), bool)
            v[ok] = mask[q[ok, 1], q[ok, 0]] > 0
            return v, ok
        out_v, out_ok = at(pts + EDGE_STEP_PX * d)
        in_v, _ = at(pts - EDGE_STEP_PX * d)
        # Off the frame edge counts as neither: the clipped boundary is not the hoop's.
        outer = pts[out_ok & ~out_v & in_v]
        if len(outer) < contract.MIN_POINTS:
            return None
        ring = _ransac(outer, self._rng)
        if ring is None:
            return None
        # As sim.observe: too little of the edge in frame and the hoop is not seen.
        if _coverage(outer, ring) < contract.MIN_ARC_FRACTION * COVER_BINS:
            return None
        keep = outer[_inliers(outer, ring)]
        return keep if len(keep) >= contract.MIN_POINTS else None

    def _marker(self, small: np.ndarray, ellipse) -> Optional[tuple]:
        primary = self._marker_with_saturation(small, ellipse, WHITE_MAX_S)
        return primary if primary is not None else self._marker_with_saturation(
            small, ellipse, WARM_WHITE_MAX_S)

    def _marker_with_saturation(self, small: np.ndarray, ellipse, max_s: int) -> Optional[tuple]:
        """(x, y, radius) of a white disc with dark strokes by the hoop's bottom, or None."""
        (cx, cy), (w, h), ang = ellipse
        major = max(w, h)
        # The ellipse's lowest point in the image: 6 o'clock on an upright hoop.
        t = np.linspace(0, 2 * np.pi, 72)
        a, b, th = w / 2, h / 2, np.deg2rad(ang)
        ys = cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
        xs = cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
        bottom = np.array([xs[ys.argmax()], ys.max()])

        # White: bright and unsaturated. The yellow wall is as bright but saturated.
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        white = cv2.inRange(hsv, (0, 0, WHITE_MIN_V), (180, max_s, 255))
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_MAX_V))
        # Close over the cross so the disc is one blob, then open with a kernel
        # wider than the pole it sits on, which leaves the disc and drops the pole.
        disc = cv2.morphologyEx(white, cv2.MORPH_CLOSE, _kernel(major * 0.1))
        disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, _kernel(major * MARKER_SIZE[0]))
        contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in contours:
            if len(c) < 5:
                continue
            # Seen off-axis the disc is an ellipse: up to 60 degrees off, half as wide.
            (mx, my), (ew, eh), _ = cv2.fitEllipse(c)
            rad = max(ew, eh) / 2
            d = np.hypot(mx - bottom[0], my - bottom[1])
            if (not MARKER_SIZE[0] <= 2 * rad / major <= MARKER_SIZE[1]
                    or min(ew, eh) / max(ew, eh, 1e-6) < MARKER_MIN_RATIO
                    or cv2.contourArea(c) / (np.pi * ew * eh / 4 + 1e-6) < MARKER_MIN_FILL  # pipe, strip
                    or d > MARKER_REACH * major):
                continue
            # The cross: dark strokes inside the disc. Blank paper or a pipe end has none.
            inside = np.zeros_like(white)
            cv2.drawContours(inside, [c], -1, 255, -1)
            inside = cv2.erode(inside, _kernel(rad * 0.3))
            share = cv2.countNonZero(dark & inside) / max(cv2.countNonZero(inside), 1)
            if MARKER_DARK[0] <= share <= MARKER_DARK[1] and (best is None or d < best[0]):
                best = (d, (mx, my, rad))
        return None if best is None else best[1]


def _kernel(size: float) -> np.ndarray:
    k = max(3, int(round(size)) | 1)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def _ransac(pts: np.ndarray, rng: np.random.Generator):
    """The ellipse whose inliers cover the most of its circumference, refined."""
    best, best_score = None, 0
    for _ in range(RANSAC_TRIES):
        try:
            e = cv2.fitEllipse(pts[rng.choice(len(pts), 5, replace=False)].astype(np.float32))
        except cv2.error:
            continue
        score = _coverage(pts, e)
        if score > best_score:
            best, best_score = e, score
    return None if best is None else _refine(pts, best) or best


def _refine(pts: np.ndarray, ellipse):
    """Refit to the points near the ellipse, tightening the band each pass."""
    for tol in (4.0, 2.0, 1.0):
        near = pts[_inliers(pts, ellipse, tol * RANSAC_TOL_PX)]
        if len(near) < contract.MIN_POINTS:
            return None
        ellipse = cv2.fitEllipse(near.astype(np.float32))
    return ellipse


def _coverage(pts: np.ndarray, ellipse) -> int:
    """Occupied ellipse-parameter sectors, invariant to fitted axis squashing.

    Image polar angles overcount arcs near the minor-axis tips and undercount
    those near the major-axis tips. Undo the fitted rotation and semiaxis scale
    before binning. This is not a perspective-invariant physical-hoop measure.
    """
    inl = _inliers(pts, ellipse)
    if not inl.any():
        return 0
    (cx, cy), (w, h), angle = ellipse
    theta = np.deg2rad(angle)
    dx, dy = pts[inl, 0] - cx, pts[inl, 1] - cy
    x = (dx * np.cos(theta) + dy * np.sin(theta)) / (w / 2)
    y = (-dx * np.sin(theta) + dy * np.cos(theta)) / (h / 2)
    a = np.arctan2(y, x)
    sectors = np.floor((a + np.pi) / (2 * np.pi) * COVER_BINS).astype(int) % COVER_BINS
    return len(np.unique(sectors))


def _inliers(pts: np.ndarray, ellipse, tol_px: float = None) -> np.ndarray:
    (_, _), (w, h), _ = ellipse
    if not (np.isfinite([w, h]).all() and min(w, h) > MIN_RING_PX and min(w, h) / max(w, h) >= contract.MIN_RATIO):
        return np.zeros(len(pts), bool)
    # |r - 1| times the mean radius is close to the distance in pixels.
    return np.abs(_radius(pts, ellipse) - 1.0) * (w + h) / 4 <= (tol_px or RANSAC_TOL_PX)


def ring_scores(mask: np.ndarray, small: np.ndarray, ellipse) -> tuple[float, float, float]:
    """How ring-like the blue is around a fitted outer edge: (band, hole, backdrop).

    band: share of the edge (in frame) with noodle just inside and none just outside.
    hole: share of the middle that is blue. A hoop has a hole; a blob does not.
    backdrop: share of just outside the edge that is the cage behind the hoop:
        yellow wall, white pipe or black net.
    """
    (cx, cy), (w, h), ang = ellipse
    t = np.linspace(0, 2 * np.pi, RING_SAMPLES, endpoint=False)
    th = np.deg2rad(ang)
    ex, ey = np.cos(t) * w / 2, np.sin(t) * h / 2
    dx, dy = ex * np.cos(th) - ey * np.sin(th), ex * np.sin(th) + ey * np.cos(th)
    hgt, wid = mask.shape

    def at(r):
        x, y = np.rint(cx + r * dx).astype(int), np.rint(cy + r * dy).astype(int)
        ok = (x >= 0) & (x < wid) & (y >= 0) & (y < hgt)
        return np.clip(x, 0, wid - 1), np.clip(y, 0, hgt - 1), ok

    def blue(r):
        x, y, ok = at(r)
        return (mask[y, x] > 0) & ok, ok

    inner, ok_in = blue(RING_INSIDE)
    outer, ok_out = blue(RING_OUTSIDE)
    ok = ok_in & ok_out
    band = float((inner & ~outer)[ok].mean()) if ok.any() else 0.0
    mids = [blue(r) for r in RING_HOLE]
    seen = np.concatenate([m[1] for m in mids])
    hole = float(np.concatenate([m[0] for m in mids])[seen].mean()) if seen.any() else 1.0
    x, y, ok = at(RING_OUTSIDE)
    hsv = cv2.cvtColor(small[y, x].reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(int)
    hue, sat, val = hsv[:, 0], hsv[:, 1], hsv[:, 2]
    yellow = (hue >= YELLOW_HUE[0]) & (hue <= YELLOW_HUE[1]) & (sat >= YELLOW_MIN_S)
    white = (sat <= WHITE_MAX_S) & (val >= WHITE_MIN_V)
    black = val <= BLACK_MAX_V
    backdrop = float((yellow | white | black)[ok].mean()) if ok.any() else 0.0
    return band, hole, backdrop


def _radius(pts: np.ndarray, ellipse) -> np.ndarray:
    """Each point's distance from the ellipse centre, as a fraction of the ellipse's
    radius in that direction: 1 on the ellipse, above 1 outside it."""
    (cx, cy), (w, h), ang = ellipse
    th = np.deg2rad(ang)
    dx, dy = pts[:, 0] - cx, pts[:, 1] - cy
    x = dx * np.cos(th) + dy * np.sin(th)
    y = -dx * np.sin(th) + dy * np.cos(th)
    return np.sqrt((x / max(w / 2, 1e-6)) ** 2 + (y / max(h / 2, 1e-6)) ** 2)


def draw(frame: np.ndarray, det: Detection) -> np.ndarray:
    out = frame.copy()
    t = max(1, frame.shape[1] // 400)
    if det.edge_px is not None:
        for x, y in det.edge_px[:: max(1, len(det.edge_px) // 300)]:
            cv2.circle(out, (int(x), int(y)), t, (0, 255, 255), -1)
    if det.ellipse_px is not None:
        if det.rejected:
            # Red: a blue shape the detector looked at and did not take as the hoop.
            cv2.ellipse(out, det.ellipse_px, (0, 0, 255), t)
            (x, y), _, _ = det.ellipse_px
            cv2.putText(out, det.rejected, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.4 * t, (0, 0, 255), t)
        else:
            cv2.ellipse(out, det.ellipse_px, (0, 255, 0), 2 * t)
    if det.marker_px is not None:
        x, y, r = det.marker_px
        cv2.circle(out, (int(x), int(y)), int(r), (255, 0, 255), 2 * t)
    text = "  ".join(f"{n}={v:+.2f}" for n, v in zip(contract.FIELDS, det.obs))
    cv2.putText(out, text, (10, 12 * t + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (0, 0, 255), t)
    return out


def main(argv: list[str]) -> None:
    if not argv:
        print(__doc__)
        return
    for path in argv:
        image = cv2.imread(path)
        if image is not None:
            det = HoopDetector()(image)
            print(path, dict(zip(contract.FIELDS, np.round(det.obs, 3).tolist())))
            out = Path(path).with_name(Path(path).stem + "_detect.png")
            cv2.imwrite(str(out), draw(image, det))
            print("  wrote", out)
            continue
        cap = cv2.VideoCapture(path)
        detector = HoopDetector(confirm=CONFIRM_FRAMES)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            cv2.imshow("detect", draw(frame, detector(frame)))
            if cv2.waitKey(1) == 27:
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv[1:])
