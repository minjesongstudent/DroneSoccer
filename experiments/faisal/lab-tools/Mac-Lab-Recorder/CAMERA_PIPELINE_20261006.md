# Camera pipeline and complete recording audit

The operator confirmed the recording was hand-carried with motors off. This work
uses the team's existing params.py measurements and variable ring-height setup.
No additional measurements were requested or invented. The simulator's mission
acceptance and physical flight validation are still incomplete.

## Changes

- A previous accepted ellipse selects fresh edge pixels on the next image, which
  are refitted and checked against the same ring band, hole, backdrop and coverage
  tests. It never publishes a stored ellipse when current evidence is missing.
  Global search remains the fallback. No camera observations are synthesized.
- Warm lighting made the white marker's measured sample saturation 103-113,
  beyond the original ceiling of 60. Marker detection first keeps the original
  test, then tries a saturation ceiling of 120 only if that test fails. Shape,
  size, location and dark-cross tests remain required. A direct global threshold
  change broke the off-axis test and was rejected in favour of this fallback.
- CameraHistory builds exactly the current group's 109-input contract from camera
  features and issued actions. Tests compare it with the simulator's public history.
- The Mac recorder now has an optional loopback browser camera window with the
  detector overlay and compatible SAC model suggestions. This is an observation
  preview, not a flight controller. Its only drone commands remain command,
  streamon and streamoff. Suggestions are never inserted as issued actions.
- Raw recording stays in the receive loop; bounded preview decoding and inference
  run separately. Overrun counts, corrupt-packet decoder errors, freshness and
  policy history resets are visible/logged. Stale predictions are suppressed.
- Saved recordings can be replayed locally without any drone connection.

## Whole-recording comparison

Video SHA256: 648a149c92d7ee3c2bc8959c50f9b5c30c82e7c0f9f09091ff7cd738bf3ce3b6.
Every one of the 3,583 decoded frames was processed in order with confirm=2.

| Output count | Original group detector | Tracking only | Final |
|---|---:|---:|---:|
| Ring-positive frames | 379 | 574 | 574 |
| Marker-positive frames | 151 | 213 | 329 |
| Separate ring-positive runs | 37 | 20 | 20 |
| Inference p95, ms on test PC | 108.38 | 88.92 | 93.36 |
| Inference p99, ms on test PC | 122.43 | 108.04 | 117.20 |

Twenty-four evenly sampled newly accepted ring frames and twelve newly accepted
marker frames were visually inspected; the ring/marker was present in those
images. Both versions reported zero rings in the first 900 floor-view frames.
In the later segment, frames 2160-3079, ring positives increased 171 to 209 and
marker positives 74 to 122. Those later frames had already been viewed during
initial review, so this is not an independent unseen test set.

These are output counts and continuity observations, not accuracy, recall or
false-positive-rate estimates. There are no exhaustive ground-truth annotations.
Close cropped rings, color changes, oblique views and compression damage remain
limitations. p99 still exceeds one 100 ms interval on this PC; Mac timing requires
the local preview's timing logs. No 10 Hz real-time guarantee is claimed.

## End-to-end verification

Accelerated replay of the actual 28,110 saved UDP payloads through the preview
decoder produced 3,582 frames and 263 prediction records with zero preview queue
drops. Eighteen decoder-error events were retained (the original stream has
damage/missing startup headers). All predictions had 109 finite input values
and zero issued-action history. The accelerated replay is not latency validation.

67 simulator/training/detector/baseline/history tests and nine recorder/browser
integration tests passed. Browser tests exercised the HTTP page, JPEG endpoint,
model inference, saved logs, stale data suppression and offline shutdown. The
simulator emitted its existing two unbounded-observation warnings. No drone was
operated by these tests. The generated package passed the same nine tests and
archive hash/CRC checks on Windows; macOS runtime is checked by its setup launcher.

## Reproduction and delivery

From repository root:

    python tools/audit_lab_video.py /path/to/camera.h264 --out /path/to/audit
    python tools/build_mac_lab_preview.py
    python tools/check_camera_recording.py --package ../Mac-Lab-Recorder-Live --session /path/to/session --out /path/to/packet-check

Numeric logs and before/after summaries are in review-20261006. The original
detector is recoverable from commit 083b927. Original recording and identifiable
review images remain local. The package builder includes explicit software files
only, never sessions, virtual environments or recordings.

Mac delivery: Mac-Lab-Recorder-Live.zip in the shared project folder. Run
1-Setup.command once with internet, then 2-Record.command on drone Wi-Fi.
4-Replay-Latest.command opens existing recordings without drone Wi-Fi. The local
window works offline; internet chat still requires a separate internet link.

The checkpoint is lags60-scratch, selected for its exact 109-input compatibility.
It has poor published mission results; this preview does not promote it or make
it flight-ready. Stronger older checkpoints have different observation contracts.
Completion of autonomous repeated passes still requires a compatible policy to
pass strict simulation validation and supervised hardware validation. This
camera integration completes a software gap, not those acceptance gates.
