"""Declared analytic/synthetic development checks; no recorded or frozen frames."""
import cv2
import numpy as np
import pytest

from src import contract
from src.camera_calibration import default_calibration
from src.detect import COVER_BINS, HoopDetector, _coverage


def ellipse_arc(ellipse, centre, span, count=401):
    (cx, cy), (w, h), rotation = ellipse
    phase = np.deg2rad(np.linspace(centre-span/2, centre+span/2, count))
    local = np.column_stack([w/2*np.cos(phase), h/2*np.sin(phase)])
    theta = np.deg2rad(rotation)
    axes = np.array([[np.cos(theta), -np.sin(theta)],
                     [np.sin(theta), np.cos(theta)]])
    return local @ axes.T + [cx, cy]


@pytest.mark.parametrize('ratio', [1., .5, .25])
@pytest.mark.parametrize('rotation', [0., 27., 90., 147.])
@pytest.mark.parametrize('centre', [3.17, 93.17, 183.17])
def test_parameter_arc_coverage_is_rotation_and_aspect_invariant(ratio, rotation, centre):
    ellipse = ((240., 180.), (240., 240.*ratio), rotation)
    # Off-boundary endpoints avoid a floating-point sector-edge tie. The long
    # arc exceeds the existing quarter-hoop gate; the short one must not pass
    # even at a minor-axis tip, where image polar angles used to overcount it.
    for span, expected in [(40., 5), (100., 11)]:
        bins = _coverage(ellipse_arc(ellipse, centre, span), ellipse)
        assert bins == expected
        assert (bins >= contract.MIN_ARC_FRACTION * COVER_BINS) == (span == 100.)


@pytest.mark.parametrize('rotation', [0., 27., 90.])
def test_full_ellipse_has_exactly_36_sectors_even_with_duplicate_seam(rotation):
    ellipse = ((0., 0.), (240., 60.), rotation)
    points = ellipse_arc(ellipse, 0., 360.)  # includes both -pi and +pi
    assert _coverage(points, ellipse) == COVER_BINS == 36


def test_identical_positive_and_negative_pi_do_not_create_a_37th_sector():
    ellipse = ((0., 0.), (240., 120.), 0.)
    points = np.array([[-120., 0.], [-120., -0.]])
    assert _coverage(points, ellipse) == 1
    wrapped = ellipse_arc(ellipse, 181.7, 40.)
    assert _coverage(wrapped, ellipse) == 5


def test_equivalent_swapped_fitted_axes_preserve_coverage():
    ellipse = ((240., 180.), (240., 100.), 27.)
    points = ellipse_arc(ellipse, 3.17, 100.)
    swapped = (ellipse[0], (100., 240.), 117.)
    assert _coverage(points, ellipse) == _coverage(points, swapped) == 11


def test_noninliers_cannot_supply_missing_angular_support():
    ellipse = ((240., 180.), (240., 100.), 27.)
    short = ellipse_arc(ellipse, 3.17, 40.)
    remote = ellipse_arc(ellipse, 0., 360.)
    remote = 2 * (remote - ellipse[0]) + ellipse[0]
    assert _coverage(remote, ellipse) == 0
    assert _coverage(np.vstack([short, remote]), ellipse) == _coverage(short, ellipse) == 5
    assert _coverage(np.empty((0, 2)), ellipse) == 0


def synthetic_scene(kind, centre, diameters, angle):
    """Pinned procedural fixtures also declared before editing in comparison evidence."""
    image = np.full((360, 480, 3), (40, 200, 220), np.uint8)
    if kind in ('ring', 'blob'):
        cv2.ellipse(image, (centre, diameters, angle), (200, 120, 30), -1)
        if kind == 'ring':
            inner = tuple(.71 * size for size in diameters)
            cv2.ellipse(image, (centre, inner, angle), (40, 200, 220), -1)
    elif kind == 'strip':
        cv2.rectangle(image, (110, 155), (370, 205), (200, 120, 30), -1)
    return image


@pytest.mark.parametrize('centre,diameters,angle', [
    ((240, 180), (220, 220), 0),       # full round
    ((240, 180), (280, 90), 0),        # full oblique
    ((240, 180), (260, 110), 35),      # full rotated
    ((40, 180), (240, 240), 0),        # clipped at left edge
    ((240, 15), (280, 100), 0),        # clipped at top edge, oblique
])
def test_existing_synthetic_full_and_cropped_positives_keep_confirmation(
        centre, diameters, angle):
    image = synthetic_scene('ring', centre, diameters, angle)
    detector = HoopDetector(480, 360, confirm=2, calibration=default_calibration())
    assert not detector(image).obs.any()
    result = detector(image)
    assert result.obs[6] == 1 and result.obs[7] == 0
    assert np.isfinite(result.obs).all()
    assert not result.rejected


@pytest.mark.parametrize('kind,centre,diameters,angle', [
    ('blob', (240, 180), (220, 140), 20),
    ('strip', (240, 180), (260, 100), 0),
    ('empty', (240, 180), (220, 220), 0),
])
def test_blue_blob_strip_and_empty_negatives_stay_rejected(kind, centre, diameters, angle):
    image = synthetic_scene(kind, centre, diameters, angle)
    detector = HoopDetector(480, 360, confirm=2, calibration=default_calibration())
    assert not detector(image).obs.any()
    assert not detector(image).obs.any()
