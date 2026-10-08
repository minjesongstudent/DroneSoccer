# Frozen camera detector validation, 7 October 2026

The existing hand-carried, motors-off recording now has a reproducible blind
annotation package and a sequential detector scorer. No detector thresholds were
changed for this review. The original recording and identifiable review images
remain in the local `lab-data` folder, outside this repository.

## What was frozen

Source session: `20261006-175222-714047`. Source SHA256:
`648a149c92d7ee3c2bc8959c50f9b5c30c82e7c0f9f09091ff7cd738bf3ce3b6`.

The selection is deterministic: decoded frame indices `17 + 75*k`, for 48
samples through frame 3542. Selection did not use detector predictions. All
3,583 frames were decoded sequentially, rather than seeking through the damaged
H.264 elementary stream. Source hash, raw PNG hash, decoded-pixel hash, frame
index and frame size are stored in the frozen selection.

Codex visually reviewed all 48 images using contact sheets without detector
overlays, and inspected frames 1592, 1817, 2792 and 2867 again at native 960x720
resolution. These are **AI visual annotations, not human-adjudicated truth**.
The label file identifies that provenance for every reviewed frame.

There are 14 visible-ring samples, all clipped at a frame boundary, and 33 clear
negative samples. Frame 2642 remains unknown because of substantial compression
damage. Nine samples show an identifiable front cross. Frame 1817 has only a
white disc sliver at the bottom edge, so its marker label remains unknown.
Unknown labels are excluded separately from each target's denominator.

This recording already influenced detector tuning and prior review. It cannot
become an independently unseen test capture. The frozen labels can be reserved
against future changes; the baseline numbers are **retrospective sample
measurements**. No independent confidence interval, general false-positive-rate
claim, or full-video accuracy claim is supported by this small correlated sample.

## Measured baseline

Every decoded frame was replayed with `confirm=2`; only reviewed, definite
sample labels were scored. The detector used the team's default intrinsics,
explicitly reported as uncalibrated. Detector SHA256:
`62d32d5c315d07828eca8238a31dac2d62d8645c7615831f0f5a7359a45ed06a`.

| Target | True positives | Misses | False positives | True negatives | Unknown excluded | Sample recall |
|---|---:|---:|---:|---:|---:|---:|
| Ring presence (all clipped views) | 7 | 7 | 0 | 33 | 1 | 7/14 = 50.0% |
| Identifiable front cross | 4 | 5 | 0 | 37 | 2 | 4/9 = 44.4% |

Observed ring false positives were 0/33 reviewed negatives, and marker false
positives 0/37. These are sample counts, not evidence of zero false positives
in deployment. Full-ring recall is undefined because this sample contains no
fully visible ring. Ring-presence labels include even small clipped arcs, so
their recall does not measure ellipse-fit accuracy or whether a pose can be
reliably estimated.

Ring misses occurred at frames 1217, 1592, 1817, 2192, 2342, 2942 and 3017.
Marker misses occurred at 1217, 1742, 2792, 2867 and 3017. At 1742, 2792 and
2867 the ring was reported but the visually identifiable front cross was missed;
the other two marker misses accompanied missed ring outputs. These examples
are documented for diagnosis; no thresholds were tuned on the frozen labels.

Eleven scorer tests passed, covering unknown/unreviewed-label exclusion,
undefined denominators, malformed labels, missing predictions, temporal history
through unsampled frames, image tampering, and preservation of existing evidence.
The first replay rejected a concurrent dependency edit; a second replay completed
with stable source hashes. Original decoder warnings are preserved in local logs.

## Diagnosis of the seven ring misses

All seven raw images were inspected at native resolution. None shows the obvious
large compression blocks seen in the separately excluded frame 2642; subtle
compression effects cannot be ruled out. The recorded rejection text describes
the detector's proposed ellipse, not the physical hoop. A `solid` rejection does
not mean the real hoop lacks a hole. No candidate ellipse or geometric truth was
saved in this baseline, so these observations do not establish localization error.

| Frame | Recorded rejection | Image evidence and implication |
|---|---|---|
| 1217 | `solid 0.90` | Large lower U-shaped ring arc and open yellow middle remain visible; top and sides are cropped. The proposed fit sampled blue in its supposed hole. Candidate fitting or mask support needs diagnosis; this cannot simply be attributed to an absent ring. |
| 1592 | `solid 0.35` | Strongly oblique ring with top clipped, but a narrow opening and both sides remain visible. Could be fit contamination or limited oblique geometry; neither usable ellipse accuracy nor an unavoidable miss is established. |
| 1817 | `no ring 0.45` | Substantial left and lower arc; top/right are outside the image. The fitted band failed its blue-inside/background-outside test. Crop-sensitive fitting or color support is a plausible cause, not a proven software defect. |
| 2192 | `solid 1.00` | Only a short arc in the upper-right corner remains. Useful whole-ring geometry is weakly constrained; a rejected local blue-blob fit is unsurprising. Visible circumference has not been measured against the 25% coverage requirement. |
| 2342 | empty (`no_candidate`) | Very near ring material appears in the lower corners, with its useful outer boundary largely outside the image. Presence is clear, but the outer-edge observation required by the policy is poorly supported. The empty reason does not identify which edge-selection stage failed. |
| 2942 | `no ring 0.42` | A sizeable arc occupies the right edge, with the center and much of the ring off-screen. Band support failed. The image alone cannot distinguish an inaccurate proposed ellipse from a mask/lighting problem. |
| 3017 | `backdrop 0.04` | Left/lower ring arc, yellow wall, and part of the front cross are visible; top/right are clipped. Only 4% of the proposed ellipse's exterior samples passed the configured backdrop colors. A wrong fit or color mismatch needs checking; the image is not evidence that the hoop is absent. |

**No detector performance fix is established by this review.** In particular,
it does not justify lowering the hole, band, backdrop, minimum-ratio, or coverage
thresholds, accepting stored geometry, or relabelling these misses as successes.
The detector remains incomplete; the new work establishes repeatable evidence.

The directly justified software improvement is richer diagnostics: record the
candidate ellipse, fresh edge/inlier count, angular coverage, valid in-frame
sample counts, band/hole/backdrop scores, and the exact no-candidate stage. These
can be added without changing detection outputs, then exercised on a separately
declared development segment and synthetic crop/obliquity scenes. Development
frames must exclude this frozen selection and its nearby temporal context;
interleaving neighboring video frames would not produce an independent split.

If those development traces show a wrong high-coverage candidate suppressing a
valid ring candidate, a concrete implementation to test is retaining multiple
ellipse candidates and applying the existing ring/hole/backdrop checks before
choosing among them. The current global search selects one coverage winner
before those checks. This is a hypothesis for development, not an implemented
or validated remedy. A candidate change must preserve the existing false-positive
checks and pass the frozen evaluation once its development is complete. If too
little outer edge exists, software cannot manufacture its geometry: retain an
unavailable observation and handle that condition explicitly in the controller.

## Reproduction

From repository root, in a Python environment with the project dependencies:

```powershell
python tools/validate_lab_detector.py prepare ../lab-data/20261006-175222-714047/camera.h264 --out ../lab-data/20261006-175222-714047/a-new-validation-package
```

Open that package's `review.html`, review its raw frames, and fill `labels.json`
using its embedded label policy. `null` means uncertain or unlabelled; it never
means negative. `reviewed` is set only after image inspection. Preserve the
unmodified `selection.json`. Do not tune detector parameters from the frozen
evaluation frames or their temporal neighborhood.

To reproduce the current frozen set's score:

```powershell
python tools/validate_lab_detector.py evaluate ../lab-data/20261006-175222-714047/camera.h264 --labels ../lab-data/20261006-175222-714047/validation-frozen-20261007/labels.json --out ../lab-data/20261006-175222-714047/validation-frozen-20261007/a-new-report.json
python -m pytest tests/test_lab_detector_validation.py -q
```

The scorer runs every frame in order with the deployed `confirm=2`, including
unsampled frames that affect temporal confirmation and tracking. It scores only
explicit reviewed labels. Source files, calibration, labels and selection are
hashed; changing them during evaluation rejects the report. Video, selected
image and decoded-pixel mismatches also fail closed. Existing packages/reports
are never overwritten. Undefined precision/recall is emitted as `null`.

Local package:
`../lab-data/20261006-175222-714047/validation-frozen-20261007/`.
It contains raw PNGs, eight contact sheets, a blind HTML review page, immutable
selection, reviewed labels, a baseline report, and the exact evaluated Python
sources under `source-snapshot/`. Numeric evidence is mirrored
under `review-20261007-validation/`; images are not mirrored.

## Remaining coverage

Human adjudication, a genuinely independent recording, fully visible ring
examples in the frozen sample, and exhaustive continuous-segment labels remain absent. Geometry,
ellipse localization, camera pose, calibrated exposure timing, and flight
clearance are not measured here. Existing footage remains useful; another
capture is not required just to label more of this recording.
