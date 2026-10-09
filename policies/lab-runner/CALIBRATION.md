# Camera intrinsics and calibration

The package manifest and preview metadata identify the actual intrinsics source.
`defaults` means the team's 960 x 720 camera defaults and zero lens distortion;
it does not mean the camera was calibrated. `calibration_file` means a supplied
file passed structural checks and its reported RMS reprojection error is below
1 pixel, the existing lab calibration utility's acceptance threshold. This
cannot verify that the file belongs to the current physical camera or crop.

No measured camera calibration file was available when this integration was
implemented. A package built without `--calibration` contains no `calib.npz` and
explicitly uses defaults. A stray file is not selected automatically by the
preview: selection comes from the manifest or the command-line option.

## Obtain measured intrinsics using the existing repository utility

From the full DroneSoccer repository, with its requirements installed, use
`python -m src.labcheck board` to display the chessboard and check its square
size with a ruler. The repository's `python -m src.labcheck calib` command
collects camera views with the drone held, motors off. That hardware step has
not been run as part of this software fix. Record the camera identity, stream
resolution, square measurement, capture date and calibration session folder.

To recompute from already captured images without contacting hardware:

```sh
python -m src.labcheck calib --from-dir runs/lab/YOUR_CALIBRATION_SESSION --square-mm YOUR_MEASURED_SQUARE_SIZE
```

Use the actual measured square size, not the placeholder above. The existing
utility needs at least eight usable chessboard views and recommends fifteen
or more; inspect its coverage report. It saves `calib.npz` in the session, and
only writes the root calibration and camera parameters when RMS is below
1 pixel. Preserve the source images and report. The preview package contains
the validation and loading code; the full repository provides this acquisition
utility.

## Build a package with an existing calibration

From the repository, choose a new output folder:

```sh
python tools/build_mac_lab_preview.py --out ../Mac-Lab-Recorder-Calibrated --calibration /path/to/measured/calib.npz
```

This validates and copies the exact file bytes, records the SHA-256, original
filename, image dimensions and reported RMS in `preview-manifest.json`, and
includes the file in `PACKAGE-SHA256.json`. Invalid or missing explicit files
stop the build; they never silently fall back to defaults. Existing package
folders or ZIPs are preserved: choose a fresh `--out` for every build.

## Select calibration on the Mac

The normal recording and replay launchers use the calibration declared in the
package manifest. To override it, open Terminal in the extracted package:

```sh
.preview-venv/bin/python live_preview.py --seconds 120 --calibration /path/to/measured/calib.npz
```

For an existing recording, add `--video /path/to/camera.h264`. Replay does not
contact the drone. The selected file's SHA-256 and source are recorded in each
session's `preview_metadata.json`; the browser shows whether defaults or a
calibration file are in use. A bundled file whose bytes do not match its
manifest stops preview startup. An explicit override is validated and logged
with its actual hash.

The loader requires numeric finite `K` (3 x 3 OpenCV zero-skew intrinsics),
`dist` (4, 5, 8, 12 or 14 coefficients), `size` (positive integer width, height),
and `rms` (finite, nonnegative, below 1 pixel). It scales intrinsics for an
unchanged aspect ratio; a different aspect ratio stops analysis because a crop
or stretched image needs matching calibration. Same aspect ratio alone does
not establish an identical crop or camera. Calibration does not validate the
detector, policy, exposure timing or physical flight behavior.
