"""Offline calibration loading tests. Synthetic matrices exist only in temporary files."""
import hashlib
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.camera_calibration import (default_calibration, load_calibration,
                                    resolve_calibration, validate_calibration_bytes)
from src.detect import HoopDetector


def synthetic_payload(**updates):
    fields = dict(K=np.array([[810., 0, 470], [0, 820, 355], [0, 0, 1]]),
                  dist=np.array([.01, -.02, .001, .002, .003]),
                  size=np.array([960, 720]), rms=.25)
    fields.update(updates)
    out = BytesIO()
    np.savez(out, **fields)
    return out.getvalue()


class CalibrationTests(unittest.TestCase):
    def test_explicit_defaults_do_not_discover_a_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'synthetic.npz'
            path.write_bytes(synthetic_payload())
            with patch('src.camera_calibration.CALIB_PATH', path):
                self.assertEqual(default_calibration().status()['status'], 'defaults')
                self.assertEqual(resolve_calibration().status()['status'], 'calibration_file')
            with patch('src.camera_calibration.CALIB_PATH', Path(td) / 'absent.npz'):
                self.assertEqual(resolve_calibration().status()['status'], 'defaults')
            with self.assertRaises(FileNotFoundError):
                resolve_calibration(Path(td) / 'missing-explicit.npz')

    def test_validation_provenance_and_scaled_intrinsics(self):
        payload = synthetic_payload()
        calibration = validate_calibration_bytes(payload, 'synthetic-test.npz')
        status = calibration.status()
        self.assertEqual(status['sha256'], hashlib.sha256(payload).hexdigest())
        self.assertEqual(status['size'], [960, 720])
        self.assertEqual(status['rms_px'], .25)
        self.assertFalse(status['physical_camera_match_verified'])
        K, dist = calibration.intrinsics(480, 360)
        np.testing.assert_allclose(K, [[405, 0, 235], [0, 410, 177.5], [0, 0, 1]])
        np.testing.assert_array_equal(dist, calibration.dist)
        K[0, 0] = 1
        dist[0] = 1
        self.assertEqual(calibration.K[0, 0], 810)
        self.assertEqual(calibration.dist[0], .01)
        with self.assertRaisesRegex(ValueError, 'aspect ratio'):
            calibration.intrinsics(1280, 720)
        with self.assertRaisesRegex(ValueError, 'positive'):
            calibration.intrinsics(0, 720)

    def test_detector_retains_selected_calibration_after_resize(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'synthetic.npz'
            path.write_bytes(synthetic_payload())
            detector = HoopDetector(calibration_path=path)
            path.unlink()  # A frame-size change must not reload or lose the selection.
            detector(np.zeros((360, 480, 3), np.uint8))
            self.assertEqual(detector.K[0, 0], 405)
            self.assertEqual(detector.calibration_status['status'], 'calibration_file')
            with self.assertRaisesRegex(ValueError, 'aspect ratio'):
                detector(np.zeros((720, 1280, 3), np.uint8))

    def test_invalid_file_never_falls_back(self):
        invalid = [dict(K=np.eye(2)), dict(K=np.full((3, 3), np.nan)),
                   dict(K=np.diag([-1., 1., 1.])),
                   dict(K=np.array([[810., 0, 470], [0, 820, 355], [0, 0, 2]])),
                   dict(K=np.array([[810., 2, 470], [0, 820, 355], [0, 0, 1]])),
                   dict(K=np.array([[810., 0, 970], [0, 820, 355], [0, 0, 1]])),
                   dict(K=np.eye(3, dtype=complex)), dict(size=[960, 0]),
                   dict(size=[960, 720.5]), dict(size=[960, 720, 1]),
                   dict(size=['960', '720']), dict(dist=np.zeros(3)),
                   dict(dist=np.zeros((2, 2))), dict(dist=[0, 0, 0, np.inf]),
                   dict(rms=np.nan), dict(rms=-.1), dict(rms=1.), dict(rms=[.1, .2])]
        for fields in invalid:
            with self.subTest(fields=fields), self.assertRaises(ValueError):
                validate_calibration_bytes(synthetic_payload(**fields), 'synthetic-invalid.npz')
        for payload in (b'not an archive', synthetic_payload()[:80]):
            with self.subTest(payload=payload[:20]), self.assertRaises(ValueError):
                validate_calibration_bytes(payload, 'broken.npz')
        out = BytesIO()
        np.savez(out, K=np.eye(3), dist=np.zeros(5), size=[960, 720])
        with self.assertRaisesRegex(ValueError, 'missing fields: rms'):
            validate_calibration_bytes(out.getvalue(), 'no-rms.npz')

    def test_supported_opencv_distortion_vectors(self):
        for count in (4, 5, 8, 12, 14):
            for shape in ((count,), (1, count), (count, 1)):
                with self.subTest(shape=shape):
                    cal = validate_calibration_bytes(
                        synthetic_payload(dist=np.zeros(shape)), 'synthetic-test.npz')
                    self.assertEqual(cal.dist.shape, (count,))


if __name__ == '__main__':
    unittest.main()
