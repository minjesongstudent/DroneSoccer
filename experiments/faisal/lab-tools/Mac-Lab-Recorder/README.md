# Mac lab recorder — Tello EDU

For the double-click launchers, start with **START-HERE.txt** and
**1-Setup.command**. Use **2-Record.command** at the lab and
**3-Export-Latest.command** after adding measurements/photos. The commands below
are also available as a Terminal fallback.

This separate package collects ground-test data for the home-PC project.
It does not run training, a flight policy or a manual flight controller.
The only outbound commands allowed are `command`, `streamon`, `streamoff`.
Use ONLY with motors stopped. Do not run alongside another drone controller.

## Before leaving internet access

1. Install Python 3.11 or 3.12 for macOS from https://www.python.org/downloads/macos/ if needed.
2. Unzip this package into Downloads. Open Terminal and run:

```sh
cd ~/Downloads/Mac-Lab-Recorder
python3 --version
python3 -m unittest -v test_recorder.py
```

Recording uses only Python's standard library, so no pip install or GPU is needed.
These tests do not contact a drone. Tested on Windows with Python; actual macOS
and drone capture still require the lab check.

Optional: install VLC before leaving, to open the saved `camera.h264` video:
https://www.videolan.org/vlc/download-macosx.html

## At the lab

1. Power on the Tello EDU with motors OFF. Close other drone apps/controllers.
2. Join its direct Tello Wi-Fi from the Mac. Losing normal internet is expected.
3. Run from this package's folder:

```sh
python3 recorder.py --seconds 120
```

4. If macOS requests Local Network access for Terminal/Python, allow it. If
   connection times out, grant access in System Settings > Privacy & Security >
   Local Network, and retry. Do not disable the firewall globally.
5. Watch packet counters: telemetry packets AND video bytes should increase.
   Counters confirm reception, not that the video decodes correctly.
6. Keeping motors stopped, hold the drone level and move slowly toward, fully
   through, and around the outside of the hanging ring. Turn to face it again.
   Repeat from another angle if useful. Stop with Ctrl+C or wait two minutes.
7. Open `camera.h264` in VLC and check it before leaving. No live preview is provided.
   Keep powered-on ground sessions short and stop if the drone becomes hot.
8. Edit `measurements.md` in the new `sessions/<timestamp>` folder, add photos,
   and export the completed folder (replace the example timestamp):

```sh
python3 recorder.py --export sessions/20260928-120000-123456
```

## Send back to the home PC

Reconnect to normal internet. Upload the latest **session export ZIP** to your
own OneDrive or transfer it by USB later. On the home PC, download/copy it into
the project's `lab-data` folder. Send the ZIP or its location in this chat.
No remote-desktop connection or matching Wi-Fi is required for file transfer.
The package itself is initially saved on the home PC; you must download or
transfer `Mac-Lab-Recorder.zip` to the Mac before these steps can run there.

## What is saved / limits

- `telemetry.jsonl`: original received state plus parsed fields, preserving unknown fields.
- `camera.h264`: original video payloads; interrupted/lost UDP data may damage video.
- `video_packets.jsonl`: host receipt times, byte offsets and sizes (not frame timestamps).
- `commands.jsonl`: SDK/stream requests and received acknowledgements.
- `session.json`: computer/runtime, timing, counts and failures.
- `measurements.md`, your photos, and SHA256.json in each export ZIP.

Missing sensors remain missing; no observations are synthesized. Hand-carried
telemetry is not flight calibration. Host timestamps cannot establish camera
exposure time, precise camera/telemetry synchronization or one-way latency.
The recorder does not measure command-to-motion delay. Do not infer reliable
flight height/velocity from a ground recording alone.

Protocol reference: official Ryze Tello SDK 2.0 (commands 8889, state 8890,
video 11111): https://dl-cdn.ryzerobotics.com/downloads/Tello/Tello%20SDK%202.0%20User%20Guide.pdf
