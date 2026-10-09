"""Validate camera intrinsics produced by ``python -m src.labcheck calib``.

File validation establishes a usable OpenCV calibration and records its bytes;
it cannot establish which physical camera, lens, or crop produced those bytes.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from src.params import CALIB_PATH, Camera


@dataclass(frozen=True)
class CameraCalibration:
    K: np.ndarray
    dist: np.ndarray
    size: tuple[int, int]
    source_file: str | None = None
    sha256: str | None = None
    rms_px: float | None = None

    def status(self) -> dict:
        loaded = self.source_file is not None
        return dict(
            status="calibration_file" if loaded else "defaults",
            label="Loaded camera calibration (camera match unverified)" if loaded
            else "Team default intrinsics (uncalibrated)",
            source_file=self.source_file, sha256=self.sha256,
            size=list(self.size), rms_px=self.rms_px,
            physical_camera_match_verified=False,
            validation="structure_and_reported_rms" if loaded else None,
        )

    def intrinsics(self, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
        if width <= 0 or height <= 0:
            raise ValueError("Frame dimensions must be positive")
        cw, ch = self.size
        # Scaling alone cannot account for a different crop or stretched stream.
        if self.source_file is not None and width * ch != height * cw:
            raise ValueError(
                f"Calibration aspect ratio {cw}x{ch} does not match frame {width}x{height}; "
                "use calibration for this camera stream/crop"
            )
        K = self.K.copy()
        K[0] *= width / cw
        K[1] *= height / ch
        return K, self.dist.copy()


def default_calibration() -> CameraCalibration:
    """Use the team's declared defaults without discovering a file on disk."""
    cam = Camera()
    K = np.array([[cam.fx, 0, cam.cx], [0, cam.fy, cam.cy], [0, 0, 1]], np.float64)
    return CameraCalibration(K, np.zeros(5), (cam.width, cam.height))


def validate_calibration_bytes(payload: bytes, source_file: str) -> CameraCalibration:
    """Validate exact bytes, including the existing lab utility's RMS < 1 px gate."""
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as data:
            missing = {"K", "dist", "size", "rms"}.difference(data.files)
            if missing:
                raise ValueError("missing fields: " + ", ".join(sorted(missing)))
            values = {name: np.asarray(data[name]) for name in ("K", "dist", "size", "rms")}
            if any(value.dtype.kind not in "iuf" for value in values.values()):
                raise ValueError("calibration fields must contain real numeric values")
            K, dist, size, rms = (values[name].astype(np.float64)
                                  for name in ("K", "dist", "size", "rms"))
        if K.shape != (3, 3) or not np.isfinite(K).all():
            raise ValueError("K must be a finite 3x3 camera matrix")
        if K[0, 0] <= 0 or K[1, 1] <= 0:
            raise ValueError("K must have positive focal lengths")
        if not np.allclose(K[2], [0, 0, 1], rtol=0, atol=1e-9):
            raise ValueError("K must have homogeneous bottom row [0, 0, 1]")
        if not np.allclose([K[0, 1], K[1, 0]], [0, 0], rtol=0, atol=1e-9):
            raise ValueError("K must use the OpenCV zero-skew camera model")
        if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
            raise ValueError("size must contain positive finite width and height")
        if not np.equal(size, np.floor(size)).all():
            raise ValueError("size dimensions must be integers")
        width, height = (int(v) for v in size)
        if not (0 <= K[0, 2] < width and 0 <= K[1, 2] < height):
            raise ValueError("K principal point must be inside the calibration image")
        if (dist.ndim not in (1, 2) or (dist.ndim == 2 and 1 not in dist.shape)
                or dist.size not in (4, 5, 8, 12, 14) or not np.isfinite(dist).all()):
            raise ValueError("dist must contain 4, 5, 8, 12, or 14 finite OpenCV coefficients")
        if rms.size != 1 or not np.isfinite(rms).all() or not 0 <= rms.item() < 1:
            raise ValueError("reported RMS reprojection error must be finite and below 1 pixel")
    except (ValueError, TypeError, OSError, KeyError, AttributeError, EOFError, BadZipFile) as exc:
        raise ValueError(f"Invalid camera calibration {source_file}: {exc}") from exc
    K.setflags(write=False)
    dist = dist.reshape(-1).copy()
    dist.setflags(write=False)
    return CameraCalibration(K, dist, (width, height), source_file,
                             hashlib.sha256(payload).hexdigest(), float(rms.item()))


def load_calibration(path: str | Path) -> CameraCalibration:
    path = Path(path).expanduser().resolve()
    return validate_calibration_bytes(path.read_bytes(), str(path))


def resolve_calibration(path: str | Path | None = None) -> CameraCalibration:
    """An explicit bad path fails; an absent conventional file uses labeled defaults."""
    if path is not None:
        return load_calibration(path)
    return load_calibration(CALIB_PATH) if CALIB_PATH.exists() else default_calibration()
