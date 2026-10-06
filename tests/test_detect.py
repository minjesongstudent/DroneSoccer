import cv2
import numpy as np

from src.detect import HoopDetector

W, H = 960, 720


def scene(cross=True, disc=(235, 235, 235)):
    """Blue ring on a yellow wall, white disc at 6 o'clock with a black cross."""
    img = np.full((H, W, 3), (40, 200, 220), np.uint8)       # yellow (BGR)
    cv2.line(img, (480, 0), (480, H), (235, 235, 235), 10)   # the white pole
    cv2.circle(img, (480, 330), 170, (200, 120, 30), 45)     # blue noodle
    cv2.circle(img, (480, 500), 40, disc, -1)
    if cross:
        cv2.line(img, (480, 470), (480, 530), (10, 10, 10), 8)
        cv2.line(img, (450, 500), (510, 500), (10, 10, 10), 8)
    return img


def test_sees_hoop_and_marker():
    obs = HoopDetector()(scene()).obs
    assert obs[6] == 1 and obs[7] == 1
    assert abs(obs[0]) < 0.05 and obs[3] > 0.9   # centred, round


def test_no_marker_without_cross_or_from_behind():
    det = HoopDetector()
    assert det(scene(cross=False)).obs[7] == 0
    assert det(scene(cross=False, disc=(10, 10, 10))).obs[7] == 0


def test_nothing_in_empty_frame():
    assert not HoopDetector()(np.full((H, W, 3), (40, 200, 220), np.uint8)).obs.any()


def test_marker_seen_off_axis():
    # The sim shows the marker up to 60 degrees off the hoop's axis, where the
    # disc is half as wide as it is tall.
    img = scene()
    squashed = cv2.resize(img, (W // 2, H))
    padded = np.full_like(img, (40, 200, 220))
    padded[:, W // 4: W // 4 + W // 2] = squashed
    obs = HoopDetector()(padded).obs
    assert obs[6] == 1 and obs[7] == 1


def test_tracking_requires_current_pixels_and_reconfirmation():
    det = HoopDetector(confirm=2)
    assert det(scene()).obs[6] == 0
    assert det(scene()).obs[6] == 1
    empty = np.full((H,W,3),(40,200,220),np.uint8)
    assert det(empty).obs[6] == 0
    assert det._edge_hint is None
    assert det(scene()).obs[6] == 0
    assert det(scene()).obs[6] == 1


def test_tracking_does_not_keep_front_marker_after_it_disappears():
    det = HoopDetector()
    assert det(scene()).obs[7] == 1
    assert det(scene(cross=False)).obs[7] == 0


def test_warm_lit_marker_still_requires_cross():
    # Actual recorded white marker under yellow lab lighting: HSV ~20,110,210.
    colour = tuple(int(v) for v in cv2.cvtColor(
        np.array([[[20,110,210]]],np.uint8),cv2.COLOR_HSV2BGR)[0,0])
    assert HoopDetector()(scene(disc=colour)).obs[7] == 1
    assert HoopDetector()(scene(cross=False,disc=colour)).obs[7] == 0
