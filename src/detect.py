from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src import contract
from src.params import CALIB_PATH, Camera

WORK_WIDTH = 480
HOOP_HSV_LO = (95, 120, 60)
HOOP_HSV_HI = (125, 255, 255)
MIN_ARC_AREA = 40
RANSAC_TRIES = 400
MIN_RING_PX = 20
COVER_BINS = 36
RANSAC_TOL_PX = 2.0
EDGE_STEP_PX = 3

RING_SAMPLES = 72
RING_INSIDE = 0.88
RING_OUTSIDE = 1.12
RING_HOLE = (0.2, 0.45)
YELLOW_HUE = (15, 40)
YELLOW_MIN_S = 80
BLACK_MAX_V = 70
WHITE_MIN_V = 150
WHITE_MAX_S = 60
MARKER_WHITE_MAX_S = (60, 100)
DARK_MAX_V = 90
MIN_BAND = 0.55
MAX_HOLE = 0.3
MIN_BACKDROP = 0.6
CONFIRM_FRAMES = 2
AGREE_SHIFT = 0.25
AGREE_SIZE = 1.4
MARKER_MIN_FILL = 0.80
MARKER_MIN_RATIO = 0.4
MARKER_DARK = (0.06, 0.6)
MARKER_SIZE = (0.06, 0.45)
MARKER_REACH = 0.35


@dataclass
class Detection:
    obs: np.ndarray
    edge_px: Optional[np.ndarray]
    ellipse_px: Optional[tuple]
    marker_px: Optional[tuple]
    rejected: str = ""


def load_intrinsics(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    if CALIB_PATH.exists():
        data = np.load(CALIB_PATH)
        K, dist = data["K"].astype(np.float64), data["dist"].astype(np.float64)
        cw, ch = (int(v) for v in data["size"])
    else:
        cam = Camera()
        K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
        dist, cw, ch = np.zeros(5), cam.width, cam.height
    K = K.copy()
    K[0] *= width / cw
    K[1] *= height / ch
    return K, dist


class HoopDetector:

    def __init__(self, width: int = Camera.width, height: int = Camera.height, confirm: int = 1):
        self.size = (width, height)
        self.confirm = confirm
        self._streak, self._last_fit = 0, None
        self.K, self.dist = load_intrinsics(width, height)
        self.scale = WORK_WIDTH / width
        self.half_extent = (width / (2 * self.K[0, 0]), height / (2 * self.K[1, 1]))
        self._open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self._close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        self._rng = np.random.default_rng(0)

    def __call__(self, frame_bgr: np.ndarray) -> Detection:
        if frame_bgr.shape[1::-1] != self.size:
            self.__init__(*frame_bgr.shape[1::-1], confirm=self.confirm)
        small = cv2.resize(frame_bgr, None, fx=self.scale, fy=self.scale, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (5, 5), 0)

        mask = self._mask(small)
        edge = self._outer_edge(mask)
        if edge is None:
            self._streak = 0
            return Detection(contract.NOT_VISIBLE.copy(), None, None, None)
        ellipse = cv2.fitEllipse(edge.astype(np.float32))
        (cx, cy), (w, h), ang = ellipse
        s = 1 / self.scale
        ellipse_px = ((cx * s, cy * s), (w * s, h * s), ang)
        edge_px = edge / self.scale

        why = self._reject(mask, small, ellipse)
        if why:
            self._streak = 0
            return Detection(contract.NOT_VISIBLE.copy(), edge_px, ellipse_px, None, why)

        marker = self._marker(small, ellipse)
        norm = cv2.undistortPoints(edge_px.reshape(-1, 1, 2).astype(np.float64), self.K, self.dist).reshape(-1, 2)
        obs = contract.observe(norm, self.half_extent, marker=marker is not None)
        if not obs[6]:
            self._streak = 0
            return Detection(obs, edge_px, ellipse_px, None, "too flat")
        if self._streak < self.confirm:
            return Detection(contract.NOT_VISIBLE.copy(), edge_px, ellipse_px, None,
                             f"confirming {self._streak}/{self.confirm}")
        return Detection(obs, edge_px, ellipse_px,
                         None if marker is None else tuple(v * s for v in marker))

    def _reject(self, mask, small, ellipse) -> str:
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
        mask = cv2.inRange(cv2.cvtColor(small, cv2.COLOR_BGR2HSV), HOOP_HSV_LO, HOOP_HSV_HI)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._open)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._close)

    def _outer_edge(self, mask: np.ndarray) -> Optional[np.ndarray]:

        contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        arcs = [c.reshape(-1, 2) for c in contours if cv2.contourArea(c) >= MIN_ARC_AREA]
        if not arcs:
            return None
        pts = np.concatenate(arcs).astype(np.float64)
        if len(pts) < contract.MIN_POINTS:
            return None

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
        outer = pts[out_ok & ~out_v & in_v]
        if len(outer) < contract.MIN_POINTS:
            return None
        ring = _ransac(outer, self._rng)
        if ring is None:
            return None
        if _coverage(outer, ring) < contract.MIN_ARC_FRACTION * COVER_BINS:
            return None
        keep = outer[_inliers(outer, ring)]
        return keep if len(keep) >= contract.MIN_POINTS else None

    def _marker(self, small: np.ndarray, ellipse) -> Optional[tuple]:
        (cx, cy), (w, h), ang = ellipse
        major = max(w, h)
        t = np.linspace(0, 2 * np.pi, 72)
        a, b, th = w / 2, h / 2, np.deg2rad(ang)
        ys = cy + a * np.cos(t) * np.sin(th) + b * np.sin(t) * np.cos(th)
        xs = cx + a * np.cos(t) * np.cos(th) - b * np.sin(t) * np.sin(th)
        bottom = np.array([xs[ys.argmax()], ys.max()])

        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        dark = cv2.inRange(hsv, (0, 0, 0), (180, 255, DARK_MAX_V))
        for max_s in MARKER_WHITE_MAX_S:
            found = self._disc(hsv, dark, major, bottom, max_s)
            if found is not None:
                return found
        return None

    @staticmethod
    def _disc(hsv, dark, major, bottom, max_s) -> Optional[tuple]:
        white = cv2.inRange(hsv, (0, 0, WHITE_MIN_V), (180, max_s, 255))
        disc = cv2.morphologyEx(white, cv2.MORPH_CLOSE, _kernel(major * 0.1))
        disc = cv2.morphologyEx(disc, cv2.MORPH_OPEN, _kernel(major * MARKER_SIZE[0]))
        contours, _ = cv2.findContours(disc, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = None
        for c in contours:
            if len(c) < 5:
                continue
            (mx, my), (ew, eh), _ = cv2.fitEllipse(c)
            rad = max(ew, eh) / 2
            d = np.hypot(mx - bottom[0], my - bottom[1])
            if (not MARKER_SIZE[0] <= 2 * rad / major <= MARKER_SIZE[1]
                    or min(ew, eh) / max(ew, eh, 1e-6) < MARKER_MIN_RATIO
                    or cv2.contourArea(c) / (np.pi * ew * eh / 4 + 1e-6) < MARKER_MIN_FILL
                    or d > MARKER_REACH * major):
                continue
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
    for tol in (4.0, 2.0, 1.0):
        near = pts[_inliers(pts, ellipse, tol * RANSAC_TOL_PX)]
        if len(near) < contract.MIN_POINTS:
            return None
        ellipse = cv2.fitEllipse(near.astype(np.float32))
    return ellipse


def _coverage(pts: np.ndarray, ellipse) -> int:
    inl = _inliers(pts, ellipse)
    if not inl.any():
        return 0
    (cx, cy), _, _ = ellipse
    a = np.arctan2(pts[inl, 1] - cy, pts[inl, 0] - cx)
    return len(np.unique(((a + np.pi) / (2 * np.pi) * COVER_BINS).astype(int)))


def _inliers(pts: np.ndarray, ellipse, tol_px: float = None) -> np.ndarray:
    (_, _), (w, h), _ = ellipse
    if not (np.isfinite([w, h]).all() and min(w, h) > MIN_RING_PX and min(w, h) / max(w, h) >= contract.MIN_RATIO):
        return np.zeros(len(pts), bool)
    return np.abs(_radius(pts, ellipse) - 1.0) * (w + h) / 4 <= (tol_px or RANSAC_TOL_PX)


def ring_scores(mask: np.ndarray, small: np.ndarray, ellipse) -> tuple[float, float, float]:
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
