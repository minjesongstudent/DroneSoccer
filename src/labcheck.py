"""Lab measurements on the real Tello: the values params.py still tags UNMEASURED.

    python -m src.labcheck check            no flight: link, battery, frame size, rates, live view
    python -m src.labcheck board            no drone: draw the chessboard so you can measure it
    python -m src.labcheck calib
                                            no flight: Tello held at this screen, writes calib.npz
    python -m src.labcheck tilt
                                            no flight: drone level, laptop 1 m ahead, camera pitch
    python -m src.labcheck tape             no drone: type in the tape measurements
    python -m src.labcheck steps --axis fb  flies: rc steps on one axis (fb, lr, ud or yaw)
    python -m src.labcheck hover            flies: hands-off hover, then an optional descent ladder
    python -m src.labcheck summary runs/lab/<folder>
                                            re-print the numbers from an earlier run

Each run writes a folder under runs/lab/: state.csv (every telemetry packet),
commands.csv (every rc sent), frames.csv (every video frame and its image shift),
events.txt, summary.txt, and video.mp4 with --record.

Keys in the video window (click it first so it has focus):
    Space    emergency: motors off at once, the drone drops
    Esc, L   land now
Ctrl-C in the terminal also lands.

calib, tilt and tape write what they measure into src/params.py, tagged MEASURED
with today's date.
"""
from __future__ import annotations

import argparse
import csv
import functools
import json
import re
import logging
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
from djitellopy import Tello

from src.params import CALIB_PATH, ROOT

PARAMS_PATH = ROOT / "src" / "params.py"

Tello.LOGGER.setLevel(logging.WARNING)

OUT = ROOT / "runs" / "lab"
DT = 0.1                      # control tick, matches params.Drone.dt
FIELDS = ("mid", "x", "y", "z", "pitch", "roll", "yaw", "vgx", "vgy", "vgz",
          "templ", "temph", "tof", "h", "bat", "baro", "time", "agx", "agy", "agz")
AXES = {"lr": 0, "fb": 1, "ud": 2, "yaw": 3}
STEP_DEFAULTS = {             # amplitudes (rc percent) and seconds per step
    "fb": ((15, 30, 45), 1.0),
    "lr": ((15, 30, 45), 1.0),
    "ud": ((30, 60), 1.0),
    "yaw": ((25, 50, 100), 2.0),
}
BOARD_CORNERS = (9, 6)        # inner corners; the board is 10 x 7 squares
KEY_SPACE, KEY_ESC = 32, 27
# The Tello answers 'error' to takeoff once it is about this hot. Sitting powered
# on with video streaming and no airflow gets it there in a few minutes.
TOO_HOT_C = 88


def default_host() -> str:
    """The drone IP from settings.json, as the front end reads it."""
    try:
        host = json.loads((ROOT / "settings.json").read_text())["drone_ip"].strip()
        return host or Tello.TELLO_IP
    except (OSError, ValueError, KeyError, AttributeError):
        return Tello.TELLO_IP


# ---------------------------------------------------------------------------
# Session: one drone, its logs, its video
# ---------------------------------------------------------------------------

@dataclass
class Hold:
    """Send one rc for some seconds. rc=None sends nothing (on the ground)."""
    label: str
    seconds: float
    rc: Optional[tuple] = (0, 0, 0, 0)


@dataclass
class Do:
    """A blocking SDK call (takeoff, land, move_down), run off the UI thread."""
    label: str
    fn: Callable[[], None]


class Session:
    def __init__(self, kind: str, host: str, record: bool = False, video: bool = True):
        self.dir = OUT / f"{datetime.now():%Y%m%d-%H%M%S}-{kind}"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.perf_counter()
        self.phase = "connect"
        self.cmd = (0, 0, 0, 0)
        self.stop = threading.Event()
        self.emergency_sent = False
        self.reader = None
        self.writer = None
        self.record = record
        self._last_frame = None
        self._last_small = None
        self.frame_bgr = None
        self._placeholder_at = -1.0

        self._events = open(self.dir / "events.txt", "w")
        self._state_file = open(self.dir / "state.csv", "w", newline="")
        self._state = csv.writer(self._state_file)
        self._state.writerow(["t", "phase", "cmd_lr", "cmd_fb", "cmd_ud", "cmd_yaw", *FIELDS])
        self._cmd_file = open(self.dir / "commands.csv", "w", newline="")
        self._cmds = csv.writer(self._cmd_file)
        self._cmds.writerow(["t", "phase", "lr", "fb", "ud", "yaw"])
        self._frame_file = open(self.dir / "frames.csv", "w", newline="")
        self._frames = csv.writer(self._frame_file)
        self._frames.writerow(["t", "phase", "width", "height", "shift_x", "shift_y"])

        self.event(f"connecting to {host}")
        self.drone = Tello(host)
        self.drone.connect()
        threading.Thread(target=self._log_state, daemon=True).start()
        self.event(f"connected, battery {self.state().get('bat')}%, temperature {self.state().get('temph')} C")
        try:
            self.drone.enable_mission_pads()
            self.drone.set_mission_pad_detection_direction(0)  # downward only, 20 Hz
            self.event("mission pads on, downward")
        except Exception as ex:
            self.event(f"mission pads not enabled: {ex}")
        if video:
            self._start_video()

    # -- time, logs ---------------------------------------------------------

    def now(self) -> float:
        return time.perf_counter() - self.t0

    def event(self, text: str) -> None:
        line = f"{self.now():8.3f}  {text}"
        print(line, flush=True)
        self._events.write(line + "\n")
        self._events.flush()

    def set_phase(self, label: str) -> None:
        self.phase = label
        self.event(f"phase: {label}")

    def state(self) -> dict:
        return self.drone.get_current_state() or {}

    def _log_state(self) -> None:
        """One row per telemetry packet. djitellopy swaps in a new dict per packet."""
        last = None
        while not self.stop.is_set():
            state = self.drone.get_current_state()
            if state and state is not last:
                last = state
                self._state.writerow([f"{self.now():.4f}", self.phase, *self.cmd,
                                      *(state.get(k, "") for k in FIELDS)])
            time.sleep(0.002)

    def send_rc(self, rc: tuple) -> None:
        self.cmd = tuple(int(v) for v in rc)
        self.drone.send_rc_control(*self.cmd)
        self._cmds.writerow([f"{self.now():.4f}", self.phase, *self.cmd])

    # -- video --------------------------------------------------------------

    def _start_video(self) -> None:
        """get_frame_read first, streamon while it waits, as the front end does."""
        base = self.drone.get_udp_video_address()
        self.drone.get_udp_video_address = lambda: f"{base}?overrun_nonfatal=1&fifo_size=5000"
        Tello.FRAME_GRAB_TIMEOUT = 12
        box: dict = {}

        def open_reader() -> None:
            try:
                box["reader"] = self.drone.get_frame_read()
            except Exception as ex:
                box["error"] = ex

        opener = threading.Thread(target=open_reader, daemon=True)
        opener.start()
        time.sleep(0.5)
        self.drone.streamon()
        opener.join(15)
        self.reader = box.get("reader")
        if self.reader is None:
            self.event(f"! video did not start: {box.get('error', 'timed out')}")
        else:
            self.event("video reader open")

    def poll_frame(self) -> bool:
        """Take the newest frame if there is one. Logs its arrival and image shift."""
        if self.reader is None:
            return False
        frame = self.reader.frame
        if frame is None or frame is self._last_frame or frame.shape[:2] == (300, 400):
            return False  # (300, 400) is the library's black placeholder
        self._last_frame = frame
        self.frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        # Whole-image shift between frames. A yaw or sideways step shows up here
        # once it reaches the camera, which is the delay the policy actually sees.
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (160, 120)).astype(np.float32)
        shift = (0.0, 0.0)
        if self._last_small is not None:
            shift, _ = cv2.phaseCorrelate(self._last_small, small)
        self._last_small = small
        scale = width / 160
        self._frames.writerow([f"{self.now():.4f}", self.phase, width, height,
                               f"{shift[0] * scale:.2f}", f"{shift[1] * scale:.2f}"])
        if self.record:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.dir / "video.mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            self.writer.write(self.frame_bgr)
        return True

    def show(self, lines: list, warn: str = "") -> int:
        """Draw the newest frame with a status block. Returns the key pressed, or -1.

        Until video arrives a black placeholder stands in, so the window, and with
        it the Space and Esc keys, exists from the start.
        """
        fresh = self.poll_frame()
        if not fresh and self.frame_bgr is None and self.now() - self._placeholder_at > 0.1:
            self._placeholder_at = self.now()
            fresh = True
        if fresh:
            view = self.frame_bgr.copy() if self.frame_bgr is not None else np.zeros((360, 480, 3), np.uint8)
            for i, text in enumerate(lines):
                cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(view, text, (12, 28 + 26 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (255, 255, 255), 1, cv2.LINE_AA)
            if warn:
                cv2.putText(view, warn, (12, view.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 0, 255), 2, cv2.LINE_AA)
            if self.record:
                cv2.circle(view, (view.shape[1] - 24, 24), 9, (0, 0, 255), -1)
            cv2.imshow("tello", view)
        key = cv2.waitKey(1)
        return -1 if key == -1 else key & 0xFF

    def status_lines(self) -> list:
        s = self.state()
        pad = f"pad {s.get('mid')}  x {s.get('x')}  y {s.get('y')}  z {s.get('z')}" \
            if s.get("mid", -1) > 0 else "pad not seen"
        return [f"{self.phase}",
                f"bat {s.get('bat')}%  {s.get('temph')} C  tof {s.get('tof')} cm  h {s.get('h')} cm  yaw {s.get('yaw')}",
                f"vg {s.get('vgx')} {s.get('vgy')} {s.get('vgz')}  {pad}",
                "Space EMERGENCY   Esc/L land"]

    # -- emergency, shutdown ---------------------------------------------------

    def emergency(self) -> None:
        self.drone.emergency()
        self.emergency_sent = True
        self.event("! EMERGENCY sent")

    def close(self) -> None:
        if self.drone.is_flying and not self.emergency_sent:
            try:
                self.drone.send_rc_control(0, 0, 0, 0)
                self.drone.land()
                self.event("landed on exit")
            except Exception as ex:
                self.event(f"! land on exit failed: {ex}")
        time.sleep(0.3)
        self.stop.set()
        if self.writer is not None:
            self.writer.release()
        try:
            self.drone.end()
        except Exception:
            pass
        cv2.destroyAllWindows()
        for f in (self._state_file, self._cmd_file, self._frame_file, self._events):
            f.close()


# ---------------------------------------------------------------------------
# Flying a schedule
# ---------------------------------------------------------------------------

def fly(session: Session, items: list, fence_cm: float, max_height_cm: float) -> None:
    """Run Hold and Do items on a 10 Hz deadline, then land. Space and Esc always work."""
    queue = list(items)
    until, rc = 0.0, None
    worker: Optional[threading.Thread] = None
    failure: list = []
    land_wanted = land_started = False
    next_tick = session.now()
    warn = ""
    # Distance from the pad, for the fence. The pad drops out of view before the
    # drone is far away, so between fixes add the displacement integrated from vg.
    # |fix| + |displacement| never underestimates. vg is taken as dm/s; if it were
    # cm/s this would overestimate tenfold, which errs towards landing early.
    fix, drift = 0.0, np.zeros(2)
    fixes: list = []          # recent pad distances; the median ignores a single glitched fix
    last_packet, last_t = None, session.now()

    def run(item: Do) -> threading.Thread:
        def target() -> None:
            try:
                item.fn()
            except Exception as ex:
                failure.append(f"{item.label}: {ex}")
        thread = threading.Thread(target=target, daemon=True)
        thread.start()
        return thread

    def land() -> None:
        session.drone.send_rc_control(0, 0, 0, 0)
        session.drone.land()

    try:
        while True:
            now = session.now()
            key = session.show(session.status_lines() + [f"left {max(0.0, until - now):4.1f} s"], warn)
            if key == KEY_SPACE:
                session.emergency()
                return
            if key in (KEY_ESC, ord("l"), ord("L")) and not land_wanted:
                session.event("land key")
                land_wanted = True

            s = session.state()
            if s is not last_packet:
                if s.get("mid", -1) > 0:
                    fixes = (fixes + [float(np.hypot(s.get("x", 0), s.get("y", 0)))])[-5:]
                    fix, drift = float(np.median(fixes)), np.zeros(2)
                elif session.drone.is_flying:
                    drift += 10.0 * np.array([s.get("vgx", 0), s.get("vgy", 0)], float) * (now - last_t)
                last_packet, last_t = s, now
            reach = fix + float(np.hypot(*drift))
            if session.drone.is_flying and not land_wanted:
                if s.get("tof", 0) > max_height_cm:
                    warn = f"too high: tof {s.get('tof')} cm"
                elif reach > fence_cm:
                    seen = "pad" if not drift.any() else "estimated"
                    warn = f"fence: {reach:.0f} cm from pad ({seen})"
                elif 0 < s.get("bat", 100) < 15:
                    warn = f"battery {s.get('bat')}%"
                else:
                    warn = ""
                if warn:
                    session.event(f"! {warn}, landing")
                    land_wanted = True

            if worker is not None:
                if worker.is_alive():
                    continue
                worker = None
                if failure:
                    session.event(f"! {failure.pop()}")
                    land_wanted = True
                    if not session.drone.is_flying:
                        return
                if land_started:
                    return

            if land_wanted and not land_started:
                session.set_phase("land")
                land_started = True
                worker = run(Do("land", land))
                continue

            if now >= until:
                if not queue:
                    land_wanted = True
                    continue
                item = queue.pop(0)
                session.set_phase(item.label)
                if isinstance(item, Do):
                    rc = None
                    worker = run(item)
                    continue
                until, rc = now + item.seconds, item.rc

            if rc is not None and now >= next_tick:
                session.send_rc(rc)
                next_tick += DT
                if now - next_tick > DT:  # fell behind: restart the deadline, don't burst
                    next_tick = now + DT
    except KeyboardInterrupt:
        session.event("Ctrl-C")


def countdown_and_takeoff(session: Session, min_battery: int) -> list:
    if session.reader is None:
        session.event("! no video, so no key window for Space and Esc. Not flying.")
        return []
    temp = session.state().get("temph", 0)
    if temp >= TOO_HOT_C:
        session.event(f"! drone is at {temp} C and will refuse takeoff. Switch it off for 5-10 min"
                      " (a fan helps), then run again without leaving it idle")
        return []
    bat = session.state().get("bat", 0)
    if bat < min_battery:
        session.event(f"! battery {bat}% is under {min_battery}%. Swap it and run again.")
        return []
    def takeoff() -> None:
        try:
            session.drone.takeoff()
        except Exception:
            session.drone.is_flying = False
            raise RuntimeError(f"refused at {session.state().get('temph')} C. If it is hot, cool it;"
                               " after a crash or motor cut, power-cycle it") from None

    def airborne() -> None:
        # 'ok' to takeoff does not mean it flew: a drone that tips on liftoff cuts
        # its motors and still sits on the floor with h = 0.
        # A hop before tipping can read high for a packet, so it must hold for 1 s.
        deadline = time.perf_counter() + 4.0
        up_since = None
        while time.perf_counter() < deadline:
            s = session.state()
            if s.get("h", 0) >= 20 or 30 <= s.get("tof", 0) < 1000:
                up_since = up_since or time.perf_counter()
                if time.perf_counter() - up_since >= 1.0:
                    return
            else:
                up_since = None
            time.sleep(0.05)
        s = session.state()
        session.drone.is_flying = False
        raise RuntimeError(f"not airborne after takeoff (h {s.get('h')} cm, tof {s.get('tof')} cm)."
                           " It probably tipped and cut its motors: check props and guards, power-cycle")

    session.event("taking off in 5 s. Click the video window: Space = emergency, Esc = land")
    return [Hold("countdown", 5.0, None),
            Do("takeoff", takeoff),
            Do("airborne check", airborne),
            Hold("settle", 3.0)]


def recentre_item(session: Session) -> Do:
    """Fly back over the pad. The drone drifts tens of cm during rests, and
    without this the steps walk it into the net."""
    def recentre() -> None:
        s = session.state()
        mid = s.get("mid", -1)
        if mid <= 0:
            session.event("recentre: pad not seen, skipped")
            return
        if np.hypot(s.get("x", 0), s.get("y", 0)) < 10:
            return
        try:
            session.drone.go_xyz_speed_mid(0, 0, int(np.clip(s.get("z", 80), 60, 120)), 30, mid)
        except Exception as ex:
            session.event(f"recentre failed: {ex}")
    return Do("recentre", recentre)


def step_items(session: Session, axis: str, amps: tuple, seconds: float, rest: float, repeat: int) -> list:
    """Each step starts from over the pad, then rests so the drone stops before the next."""
    items = []
    n = 0
    for _ in range(repeat):
        for amp in amps:
            for sign in (1, -1):
                n += 1
                rc = [0, 0, 0, 0]
                rc[AXES[axis]] = sign * amp
                items += [recentre_item(session), Hold("steady", 1.0)]
                items.append(Hold(f"step {axis} {sign * amp:+d} #{n}", seconds, tuple(rc)))
                items.append(Hold("rest", rest))
    items.append(recentre_item(session))  # land on the pad, ready for the next run
    return items


def ladder_items(session: Session, levels: int, seconds: float) -> list:
    def down() -> None:
        tof = session.state().get("tof", 0)
        if tof > 50:
            session.drone.move_down(20)
        else:
            session.event(f"tof {tof} cm: not going lower")
    items = []
    for i in range(levels):
        items += [Do(f"down #{i + 1}", down), Hold(f"low #{i + 1}", seconds)]
    return items


# ---------------------------------------------------------------------------
# Chessboard calibration
# ---------------------------------------------------------------------------

def screen() -> Optional[tuple]:
    """Main display as (width in points, height in points, mm per point), macOS only."""
    try:
        import ctypes
        import ctypes.util

        class Size(ctypes.Structure):
            _fields_ = [("width", ctypes.c_double), ("height", ctypes.c_double)]

        cg = ctypes.cdll.LoadLibrary(ctypes.util.find_library("CoreGraphics"))
        cg.CGMainDisplayID.restype = ctypes.c_uint32
        cg.CGDisplayScreenSize.restype = Size
        cg.CGDisplayScreenSize.argtypes = [ctypes.c_uint32]
        for fn in (cg.CGDisplayPixelsWide, cg.CGDisplayPixelsHigh):
            fn.restype = ctypes.c_size_t
            fn.argtypes = [ctypes.c_uint32]
        display = cg.CGMainDisplayID()
        width_mm = cg.CGDisplayScreenSize(display).width
        width, height = cg.CGDisplayPixelsWide(display), cg.CGDisplayPixelsHigh(display)
        return (width, height, width_mm / width) if width_mm > 0 and width > 0 else None
    except Exception:
        return None


SCREEN = screen()
# Where board saves the ruler-checked square size. On a MacBook with a notch the
# fullscreen board is drawn about 4% smaller than the screen size predicts, and
# OpenCV cannot tell, so the ruler has the last word.
SQUARE_PATH = OUT / "square_mm.json"


def board_layout() -> tuple:
    """(canvas width, canvas height, square side, grid left, grid top), in screen points.

    The canvas is the whole screen, shown fullscreen, so one image pixel is one
    point; the grid is as large as fits with a square of margin for the text.
    """
    cols, rows = BOARD_CORNERS[0] + 1, BOARD_CORNERS[1] + 1
    width, height = (SCREEN[0], SCREEN[1]) if SCREEN else ((cols + 2) * 80, (rows + 2) * 80)
    sq = int(min(width / (cols + 2), height / (rows + 2)))
    return width, height, sq, (width - cols * sq) // 2, (height - rows * sq) // 2


@functools.lru_cache(maxsize=1)
def board_image() -> np.ndarray:
    cols, rows = BOARD_CORNERS[0] + 1, BOARD_CORNERS[1] + 1
    width, height, sq, left, top = board_layout()
    img = np.full((height, width), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[top + r * sq:top + (r + 1) * sq, left + c * sq:left + (c + 1) * sq] = 0
    # Grey outline round the grid, so the outer white squares have an edge to
    # measure to. The detector only uses inner corners, so it does not see this.
    cv2.rectangle(img, (left, top), (left + cols * sq - 1, top + rows * sq - 1), 128, 2)
    return img


def screen_square_mm() -> Optional[float]:
    """Board square size predicted from the screen's size. A few percent too big
    on a MacBook with a notch; board asks for a ruler check and saves the truth."""
    return board_layout()[2] * SCREEN[2] if SCREEN else None


def saved_square_mm() -> Optional[float]:
    """The square size board saved, if it was measured on this screen."""
    try:
        saved = json.loads(SQUARE_PATH.read_text())
        return float(saved["square_mm"]) if saved.get("screen") == list(board_layout()[:2]) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def square_mm_arg(args) -> float:
    """--square-mm if given, else what board saved, else the screen's prediction."""
    if args.square_mm:
        return args.square_mm
    saved = saved_square_mm()
    if saved:
        print(f"Using square size {saved:.2f} mm, ruler-checked with board (--square-mm overrides)")
        return saved
    estimate = screen_square_mm()
    if estimate is None:
        sys.exit("Could not read the screen size. Run board first, or pass --square-mm.")
    print(f"! Using square size {estimate:.2f} mm predicted from the screen, not checked.\n"
          f"  Run board first to check it with a ruler.")
    return estimate


def open_board_window() -> None:
    cv2.namedWindow("board", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("board", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def show_board(status: str = "", flash: bool = False, hint: str = "") -> None:
    img = cv2.cvtColor(board_image(), cv2.COLOR_GRAY2BGR)
    width, height, sq, left, top = board_layout()
    scale = sq / 110
    if flash:
        cv2.rectangle(img, (0, 0), (width - 1, height - 1), (0, 200, 0), max(10, sq // 3))
    if status:
        cv2.putText(img, status, (left, top - sq // 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (0, 0, 200), 2, cv2.LINE_AA)
    if hint:  # under the grid, large enough to read with the laptop held out
        cv2.putText(img, hint, (left, height - top + sq // 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.3,
                    (200, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("board", img)


# Views calibration wants, as pose_signature entries {index: value}: the board
# in each of nine places in the frame, near and far, and tipped four ways.
# Index 0, 1: centre x, y as a fraction of the frame; 2: width as a fraction;
# 3: top wider than bottom; 4: left taller than right.
CALIB_TARGETS = [{0: x, 1: y, 2: 0.35} for y in (0.25, 0.5, 0.75) for x in (0.25, 0.5, 0.75)] + [
    {0: 0.5, 1: 0.5, 2: 0.2}, {0: 0.5, 1: 0.5, 2: 0.55},
    {0: 0.5, 1: 0.5, 3: 0.15}, {0: 0.5, 1: 0.5, 3: -0.15},
    {0: 0.5, 1: 0.5, 4: 0.15}, {0: 0.5, 1: 0.5, 4: -0.15},
]
TARGET_TOL = {0: 0.12, 1: 0.12, 2: 0.08, 3: 0.07, 4: 0.07}


def meets(sig: np.ndarray, target: dict) -> bool:
    return all(abs(sig[i] - v) < TARGET_TOL[i] for i, v in target.items())


def next_target(taken: list) -> Optional[dict]:
    for target in CALIB_TARGETS:
        if not any(meets(sig, target) for sig in taken):
            return target
    return None


def direction(sig: Optional[np.ndarray], target: Optional[dict]) -> str:
    """One instruction, for someone holding the laptop in front of the drone.

    Left and right are the drone's: the board sits further left in its picture
    when the laptop moves to the drone's left, which is your right as you face it.
    """
    if target is None:
        return "All views covered. Any extra angle helps, or press C"
    if sig is None:
        return "Board not seen: point the screen at the drone, 40-80 cm away"
    for i, v in target.items():  # fix position first, then size, then tilt
        d = v - sig[i]
        if abs(d) < TARGET_TOL[i]:
            continue
        if i == 0:
            return "Move laptop to the drone's " + ("RIGHT (your left)" if d > 0 else "LEFT (your right)")
        if i == 1:
            return "Move laptop " + ("DOWN" if d > 0 else "UP")
        if i == 2:
            return "Move laptop " + ("CLOSER to the drone" if d > 0 else "FURTHER from the drone")
        if i == 3:
            return "Tip the screen: " + ("TOP edge toward the drone" if d > 0 else "BOTTOM edge toward the drone")
        if i == 4:
            return "Turn the screen: " + ("drone's-LEFT edge toward it" if d > 0 else "drone's-RIGHT edge toward it")
    return "Hold still..."


def find_board(gray: np.ndarray, fast: bool):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, BOARD_CORNERS, flags=flags)
    return corners.reshape(-1, 2) if found else None


def pose_signature(corners: np.ndarray, width: int, height: int) -> np.ndarray:
    """Where the board sits, how big, and how tilted: enough to tell two views apart."""
    cx, cy = corners.mean(axis=0)
    w_top = np.linalg.norm(corners[BOARD_CORNERS[0] - 1] - corners[0])
    w_bottom = np.linalg.norm(corners[-1] - corners[-BOARD_CORNERS[0]])
    h_left = np.linalg.norm(corners[-BOARD_CORNERS[0]] - corners[0])
    h_right = np.linalg.norm(corners[-1] - corners[BOARD_CORNERS[0] - 1])
    size = (w_top + w_bottom) / (2 * width)
    return np.array([cx / width, cy / height, size, w_top / w_bottom - 1, h_left / h_right - 1])


def calibrate(folder: Path, square_mm: float) -> None:
    images = sorted(folder.glob("calib_*.png"))
    objp = np.zeros((BOARD_CORNERS[0] * BOARD_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_CORNERS[0], 0:BOARD_CORNERS[1]].T.reshape(-1, 2) * square_mm / 1000
    obj_points, img_points, size = [], [], None
    for path in images:
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_board(gray, fast=False)
        if corners is None:
            print(f"  {path.name}: board not found again, skipped")
            continue
        corners = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        obj_points.append(objp)
        img_points.append(corners)
    if len(img_points) < 8:
        print(f"Only {len(img_points)} usable views. Need at least 8, ideally 20.")
        return
    rms, K, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
    print(f"\nCalibrated from {len(img_points)} views at {size[0]}x{size[1]}")
    print(f"  RMS reprojection error {rms:.3f} px  (under 0.5 good, under 1.0 usable)")
    print(f"  fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}   "
          f"(guessed 920, 920, 480, 360)")
    print(f"  dist {np.round(dist.ravel(), 4)}")
    hfov = np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))
    print(f"  horizontal field of view {hfov:.1f} deg")
    np.savez(folder / "calib.npz", K=K, dist=dist, size=np.array(size), rms=rms)
    if rms < 1.0:
        np.savez(CALIB_PATH, K=K, dist=dist, size=np.array(size), rms=rms)
        print(f"  wrote {CALIB_PATH.relative_to(ROOT)} (the simulator now loads it)")
        note = f"chessboard, {len(img_points)} views, rms {rms:.2f} px"
        for name, value in (("fx", K[0, 0]), ("fy", K[1, 1]), ("cx", K[0, 2]), ("cy", K[1, 2])):
            set_param(name, f"{value:.1f}", note)
    else:
        print(f"  RMS too high: kept in {folder.relative_to(ROOT)} only, not written to calib.npz")


def run_calib(args) -> None:
    if args.from_dir:
        calibrate(Path(args.from_dir), square_mm_arg(args))
        return
    open_board_window()
    square_mm = square_mm_arg(args)
    session = Session("calib", args.host)
    taken: list = []
    previous = None
    flash_until = 0.0
    hint = "Waiting for video..."
    try:
        while len(taken) < args.frames:
            status = f"{len(taken)}/{args.frames} captured. Hold still at a new angle.  Esc stop, C calibrate now"
            show_board(status, flash=session.now() < flash_until, hint=hint)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                return
            if key in (ord("c"), ord("C")) and len(taken) >= 8:
                break
            if not session.poll_frame():
                continue
            frame = session.frame_bgr
            height, width = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_board(gray, fast=True)
            sig = None if corners is None else pose_signature(corners, width, height)
            hint = direction(sig, next_target(taken))
            if corners is not None:
                still = previous is not None and np.abs(corners - previous).mean() < 1.5
                new = all(np.abs(sig - other).max() > 0.08 for other in taken)
                if still and new:
                    cv2.imwrite(str(session.dir / f"calib_{len(taken):02d}.png"), frame)
                    taken.append(sig)
                    flash_until = session.now() + 0.4
                    session.event(f"captured view {len(taken)}")
            previous = corners
    finally:
        session.close()
    calibrate(session.dir, square_mm)


# ---------------------------------------------------------------------------
# Summary: the numbers for params.py
# ---------------------------------------------------------------------------

def load(path: Path) -> dict:
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    out = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        try:
            out[key] = np.array([float(v) if v != "" else np.nan for v in values])
        except ValueError:
            out[key] = np.array(values)
    return out


DELAYS = np.arange(0.0, 0.81, 0.01)
TAUS = np.geomspace(0.03, 1.2, 50)


def _response(s: np.ndarray, tau: np.ndarray, integrated: bool) -> np.ndarray:
    """Unit step response of a first-order lag (velocity), or its integral (position)."""
    s = np.maximum(s, 0.0)
    decay = np.exp(-s / tau)
    return s - tau * (1 - decay) if integrated else 1 - decay


def fit_step(t, y, on, off, integrated=False, delay=None, tau=None) -> dict:
    """Fit y = base + gain * (r(t - on - delay) - r(t - off - delay)).

    One step on at `on`, off at `off`; r is _response. Grid search over delay and
    tau (or hold them fixed), closed-form least squares for base and gain. The
    gain is the steady value per unit: raw velocity, or cm/s and deg/s when
    fitting a position or angle with integrated=True.
    """
    ok = np.isfinite(y)
    t, y = t[ok], y[ok]
    nan = {"delay": np.nan, "tau": np.nan, "gain": np.nan, "rms": np.nan}
    if len(t) < 6:
        return nan
    taus = (np.array([tau]) if tau is not None else TAUS)[:, None]
    yc = y - y.mean()
    best = None
    for d in ([delay] if delay is not None else DELAYS):
        x = _response(t - on - d, taus, integrated) - _response(t - off - d, taus, integrated)
        xc = x - x.mean(axis=1, keepdims=True)
        var = (xc ** 2).sum(axis=1)
        cov = (xc * yc).sum(axis=1)
        gain = np.where(var > 1e-12, cov / np.maximum(var, 1e-12), 0.0)
        err = (yc ** 2).sum() - gain * cov
        i = int(np.argmin(err))
        if best is None or err[i] < best[0]:
            best = (err[i], d, float(taus[i, 0]), float(gain[i]))
    err, d, k, g = best
    return {"delay": d, "tau": k, "gain": g, "rms": float(np.sqrt(max(err, 0) / len(t)))}


def deglitch(v: np.ndarray, ok: np.ndarray, limit: float = 15.0) -> np.ndarray:
    """Mark pad samples more than `limit` cm from the median of their 5 neighbours."""
    out = ok.copy()
    idx = np.flatnonzero(ok)
    for j, i in enumerate(idx):
        near = v[idx[max(0, j - 2):j + 3]]
        if abs(v[i] - np.median(near)) > limit:
            out[i] = False
    return out


def summarise(folder: Path) -> str:
    out = []
    say = out.append
    state, cmds, frames = (load(folder / n) for n in ("state.csv", "commands.csv", "frames.csv"))
    say(f"Summary of {folder.name}")

    if cmds:
        gaps = np.diff(cmds["t"])
        gaps = gaps[gaps < 1.0]  # skip pauses for takeoff and moves
        if len(gaps):
            say(f"  rc loop      {1 / gaps.mean():5.1f} Hz mean, interval p95 {np.percentile(gaps, 95) * 1000:.0f} ms,"
                f" worst {gaps.max() * 1000:.0f} ms   (target 10 Hz)")
    if state:
        gaps = np.diff(state["t"])
        say(f"  telemetry    {1 / np.median(gaps):5.1f} Hz (median), worst gap {gaps.max() * 1000:.0f} ms")
    if frames:
        gaps = np.diff(frames["t"])
        say(f"  video        {int(frames['width'][0])}x{int(frames['height'][0])},"
            f" {1 / np.median(gaps):4.1f} fps (median), worst gap {gaps.max() * 1000:.0f} ms")
    if not state:
        return "\n".join(out)

    t, phase = state["t"], state["phase"]
    airborne = ~np.isin(phase, ["connect", "countdown", "takeoff", "land"])
    if airborne.any():
        seen = (state["mid"][airborne] > 0).mean()
        say(f"  pad seen     {seen * 100:4.0f}% of airborne packets")

    for name in ("settle", "hover"):
        m = phase == name
        if m.sum() > 5:
            note = "   (takeoff_height guessed 80 cm)" if name == "settle" else ""
            say(f"  {name:12s} tof {np.median(state['tof'][m]):.0f} cm,"
                f" h {np.median(state['h'][m]):.0f} cm{note}")
    m = (phase == "hover") & (state["mid"] > 0)
    if m.sum() > 5:
        x, y = state["x"][m], state["y"][m]
        span = float(t[m][-1] - t[m][0])
        say(f"  hover drift  x {x.min():.0f}..{x.max():.0f} cm, y {y.min():.0f}..{y.max():.0f} cm,"
            f" end minus start ({x[-1] - x[0]:+.0f}, {y[-1] - y[0]:+.0f}) cm over {span:.0f} s")
    lows = sorted({p for p in phase if p.startswith("low #")})
    for p in lows:
        m = phase == p
        steady = m & (state["mid"] > 0)
        wobble = f", pad x/y std {state['x'][steady].std():.1f}/{state['y'][steady].std():.1f} cm" \
            if steady.sum() > 5 else ", pad not seen"
        say(f"  {p:12s} tof {np.median(state['tof'][m]):.0f} cm, vg std "
            f"{state['vgx'][m].std():.2f}/{state['vgy'][m].std():.2f}{wobble}")

    steps = []
    for label in dict.fromkeys(p for p in phase if p.startswith("step ")):
        _, axis, amp, _ = label.split()
        sent = cmds["t"][cmds["phase"] == label] if cmds else t[phase == label]
        if len(sent) < 2:
            continue
        on, off = float(sent[0]), float(sent[-1]) + DT
        w = (t >= on - 0.5) & (t <= off + 1.5)
        row = {"label": label, "axis": axis, "amp": int(amp), "speed": np.nan, "video_delay": np.nan}
        if axis == "yaw":
            yaw = np.degrees(np.unwrap(np.radians(state["yaw"])))
            fit = fit_step(t[w], yaw[w], on, off, integrated=True)
            row.update(fit, comp="yaw", unit="deg/s")
            if frames:
                # A turn slides the whole image sideways; per-frame shift tracks yaw
                # rate. Its delay is command to camera, which is what the policy sees.
                ft = frames["t"]
                fw = (ft >= on - 0.5) & (ft <= off + 1.5)
                row["video_delay"] = fit_step(ft[fw], frames["shift_x"][fw], on, off)["delay"]
        else:
            comps = {"fb": ("vgx", "vgy"), "lr": ("vgy", "vgx"), "ud": ("vgz",)}[axis]
            fits = {c: fit_step(t[w], state[c][w], on, off) for c in comps}
            comp = max(fits, key=lambda c: abs(np.nan_to_num(fits[c]["gain"])))
            row.update(fits[comp], comp=comp, unit="raw")
            d, k = row["delay"], row["tau"]
            keys = ("z",) if axis == "ud" else ("x", "y")
            pad = w & (state["mid"] > 0)
            for c in keys:
                pad = deglitch(state[c], pad)
            if pad.sum() >= 6 and abs(np.nan_to_num(row["gain"])) < 0.5:
                # vg is whole dm/s, so slow steps read 0 and the fit above is empty.
                # Fit delay and tau from pad position instead, on the axis that moved most.
                pf = [fit_step(t[pad], state[c][pad], on, off, True) for c in keys]
                best = max(pf, key=lambda f: abs(np.nan_to_num(f["gain"])))
                d, k = best["delay"], best["tau"]
                row.update(delay=d, tau=k, comp="pad")
            if pad.sum() >= 6 and np.isfinite(d):
                gains = [fit_step(t[pad], state[c][pad], on, off, True, d, k)["gain"] for c in keys]
                row["speed"] = float(np.hypot(*gains) if len(gains) == 2 else abs(gains[0])) / 100
            if axis == "ud":
                row["tof_speed"] = fit_step(t[w], state["tof"][w], on, off, True, d, k)["gain"] / 100
        steps.append(row)

    if steps:
        say("")
        say("  step                 delay    tau    gain (raw)       pad speed   video delay")
        for r in steps:
            extra = f"   tof {r['tof_speed']:+.2f} m/s" if "tof_speed" in r else ""
            say(f"  {r['label']:20s} {r['delay'] * 1000:4.0f} ms {r['tau']:5.2f} s  {r['comp']:4s} {r['gain']:+7.1f}"
                f"   {r['speed']:6.2f} m/s   {r['video_delay'] * 1000:5.0f} ms{extra}")
        say("")
        lin = [r for r in steps if r["axis"] != "yaw"
               and (r["comp"] == "pad" or abs(np.nan_to_num(r["gain"])) >= 0.5)]
        moving = [r for r in steps if r["axis"] == "yaw"] + lin
        if moving:
            say(f"  delay (telemetry) median {np.nanmedian([r['delay'] for r in moving]) * 1000:.0f} ms;"
                f" it includes the telemetry link's own lag")
        video = [r["video_delay"] for r in steps if np.isfinite(r["video_delay"])]
        if video:
            vd = float(np.median(video))
            say(f"  delay (video, yaw steps) median {vd * 1000:.0f} ms"
                f"  -> delay_steps {vd / DT:.1f} at 10 Hz (use this one: it is what the policy sees)")
        if lin:
            say(f"  -> tau {np.nanmedian([r['tau'] for r in lin]):.2f} s (guessed 0.3, randomised 0.15-0.5)")
        for axis in ("fb", "lr", "ud"):
            rows = [r for r in steps if r["axis"] == axis]
            if not rows:
                continue
            amps = np.array([abs(r["amp"]) for r in rows], float)
            speed = np.array([r["speed"] for r in rows])
            raw = np.array([abs(r["gain"]) for r in rows])
            ok = np.isfinite(speed) & (speed > 0.02)
            if ok.sum() >= 2 and np.ptp(amps[ok]) > 0:
                slope, icpt = np.polyfit(amps[ok], speed[ok], 1)
                say(f"  -> {axis}: {slope * 100:.2f} m/s per 100 rc (from the pad), so 0.4 m/s at rc"
                    f" {(0.4 - icpt) / slope:.0f}; raw {rows[0]['comp']} per m/s {np.median(raw[ok] / speed[ok]):.1f}")
            else:
                say(f"  -> {axis}: pad too patchy for a speed; raw {rows[0]['comp']} by rc "
                    f"{dict(zip(amps.astype(int).tolist(), np.round(raw, 1).tolist()))}")
            if axis == "ud" and max(abs(np.nan_to_num(r.get("tof_speed", 0.0))) for r in rows) < 0.05:
                say("  !! ud: tof barely moved, so the rc up/down channel looks ignored. The policy uses it.")
        yaw = [r for r in steps if r["axis"] == "yaw" and np.isfinite(r["gain"])]
        if yaw:
            say("  -> yaw: " + ", ".join(f"rc {r['amp']:+d} = {r['gain']:+.0f} deg/s" for r in yaw)
                + f"; tau {np.median([r['tau'] for r in yaw]):.2f} s")
    return "\n".join(out)


def finish(session: Session) -> None:
    text = summarise(session.dir)
    (session.dir / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nSaved to {session.dir.relative_to(ROOT)}")


# ---------------------------------------------------------------------------
# Writing results into params.py
# ---------------------------------------------------------------------------

def set_param(name: str, value: str, note: str) -> None:
    """Replace `name: type = value  # (UN)MEASURED ...` in params.py and drop its
    continuation comment lines, which describe how to measure it."""
    text = PARAMS_PATH.read_text()
    pattern = re.compile(rf"^(\s+{name}: \w+ = )[^#\n]+#\s*(?:UN)?MEASURED[^\n]*(?:\n\s{{20,}}#[^\n]*)*",
                         re.MULTILINE)
    line = f"{datetime.now():%Y-%m-%d}: {note}"
    text, n = pattern.subn(lambda m: f"{m.group(1)}{value}  # MEASURED {line}", text, count=1)
    if n:
        PARAMS_PATH.write_text(text)
        print(f"  params.py: {name} = {value}")
    else:
        print(f"  ! params.py: no {name} line tagged UNMEASURED or MEASURED, set it by hand to {value}")


# ---------------------------------------------------------------------------
# Camera tilt
# ---------------------------------------------------------------------------

def ask_cm(question: str) -> float:
    while True:
        try:
            return float(input(question).strip())
        except ValueError:
            print("  a number, please")


def run_tilt(args) -> None:
    """Pitch of the camera below horizontal, from where the board centre appears.

    The board's position comes from solvePnP with the calibrated camera, so the
    distance need not be measured; only how far the board centre sits above the
    lens (board_cm - lens_cm), which turns into an elevation angle we subtract.
    """
    if not CALIB_PATH.exists():
        print("No calib.npz yet. Run calib first: tilt needs the intrinsics.")
        return
    data = np.load(CALIB_PATH)
    K, dist = data["K"], data["dist"]
    lens_cm = args.lens_cm if args.lens_cm is not None else ask_cm(
        "Camera lens height above the floor (cm): ")
    board_cm = args.board_cm if args.board_cm is not None else ask_cm(
        "Middle of the laptop screen, height above the floor (cm): ")
    open_board_window()
    objp = np.zeros((BOARD_CORNERS[0] * BOARD_CORNERS[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_CORNERS[0], 0:BOARD_CORNERS[1]].T.reshape(-1, 2)
    objp[:, :2] -= objp[:, :2].mean(axis=0)  # origin at the board centre
    objp *= square_mm_arg(args) / 1000
    dh = (board_cm - lens_cm) / 100

    session = Session("tilt", args.host)
    pitches, distances = [], []
    imu = None
    hint = "Waiting for video..."
    try:
        while len(pitches) < args.frames:
            show_board(f"{len(pitches)}/{args.frames} frames. Do not touch the drone. Esc stop", hint=hint)
            if cv2.waitKey(1) & 0xFF == KEY_ESC:
                return
            if not session.poll_frame():
                continue
            frame = session.frame_bgr
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            corners = find_board(gray, fast=True)
            hint = "Measuring, keep still" if corners is not None else \
                "Board not seen: move laptop closer or in line with the camera"
            if corners is not None:
                corners = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                                           (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
                ok, _, tvec = cv2.solvePnP(objp, corners, K, dist)
                if ok:
                    x, y, z = tvec.ravel()  # camera frame: x right, y down, z forward
                    above_axis = np.degrees(np.arctan2(-y, np.hypot(x, z)))
                    horizontal = np.sqrt(max(x * x + y * y + z * z - dh * dh, 1e-6))
                    elevation = np.degrees(np.arctan2(dh, horizontal))
                    pitches.append(above_axis - elevation)
                    distances.append(horizontal)
        imu = session.state().get("pitch")
    finally:
        session.close()

    pitch, spread = float(np.median(pitches)), float(np.std(pitches))
    print(f"\nCamera pitch {pitch:.2f} deg below horizontal (spread {spread:.2f}) "
          f"from {len(pitches)} frames at {np.median(distances):.2f} m")
    print(f"  drone IMU pitch while sitting: {imu} deg")
    if imu is not None and abs(imu) >= 2:
        print("  ! the drone is not sitting level, so this includes its own tilt. Level it and rerun.")
    elif spread > 0.5:
        print("  ! spread over 0.5 deg: something moved. Not written; rerun.")
    else:
        set_param("pitch_deg", f"{pitch:.1f}", f"labcheck tilt at {np.median(distances):.1f} m")


# ---------------------------------------------------------------------------
# Tape measurements
# ---------------------------------------------------------------------------

TAPE = (  # name in params.py, question, factor to the stored value (None: stored as typed)
    ("thickness", "Noodle diameter (cm)", 0.01),
    ("hanger_half_width", "Bar on top of the hoop, full length (cm)", 0.005),
    ("offset_forward", "Camera lens ahead of the drone's centre, horizontally (cm)", 0.01),
    ("marker_max_angle_deg", "Widest angle off the hoop's front at which the marker still shows (deg)", None),
)


def run_tape(args) -> None:
    print("Type each measurement, or press Enter to skip it.")
    for name, question, factor in TAPE:
        answer = input(f"{question}: ").strip()
        if not answer:
            continue
        try:
            value = float(answer)
        except ValueError:
            print(f"  '{answer}' is not a number, skipped")
            continue
        if factor is None:
            set_param(name, f"{value:.1f}", "by eye")
        else:
            half = " / 2" if name == "hanger_half_width" else ""
            set_param(name, f"{value * factor:.4g}", f"tape, {answer} cm{half}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_check(args) -> None:
    session = Session("check", args.host, record=args.record)
    try:
        for name, query in (("sdk", "query_sdk_version"), ("serial", "query_serial_number"),
                            ("wifi snr", "query_wifi_signal_noise_ratio")):
            try:
                session.event(f"{name}: {getattr(session.drone, query)()}")
            except Exception as ex:
                session.event(f"{name}: failed ({ex})")
        session.set_phase("view")
        session.event("live view. Hold the drone over the pad to test pad range. Esc to stop")
        while True:
            key = session.show(session.status_lines()[:3] + ["Esc to stop"])
            if key in (KEY_ESC, ord("q")):
                break
    except KeyboardInterrupt:
        pass
    finally:
        session.close()
    finish(session)


def run_board(args) -> None:
    total = BOARD_CORNERS[0] + 1
    estimate = screen_square_mm()
    expected = f"should be about {estimate * total:.0f} mm" if estimate else "measure it"
    print(f"Measure the top row of {total} squares along the grey outline with a ruler.\n"
          f"Press Esc on the board, then type the length here.")
    open_board_window()
    while True:
        show_board(f"Top row: {total} squares, {expected}. Measure, then Esc.")
        if cv2.waitKey(30) & 0xFF == KEY_ESC:
            break
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    while True:
        answer = input(f"Top row of {total} squares, in mm: ").strip()
        try:
            row = float(answer)
            break
        except ValueError:
            print("  a number, please")
    square = row / total
    SQUARE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SQUARE_PATH.write_text(json.dumps({"square_mm": square, "row_mm": row,
                                       "screen": list(board_layout()[:2]),
                                       "date": f"{datetime.now():%Y-%m-%d}"}) + "\n")
    print(f"Saved: one square {square:.2f} mm. calib and tilt use this without --square-mm.")


def run_steps(args) -> None:
    amps, seconds = STEP_DEFAULTS[args.axis]
    amps = tuple(args.amps) if args.amps else amps
    seconds = args.seconds or seconds
    session = Session(f"steps-{args.axis}", args.host, record=args.record)
    try:
        start = countdown_and_takeoff(session, args.min_battery)
        if start:
            fly(session, start + step_items(session, args.axis, amps, seconds, args.rest, args.repeat),
                args.fence_cm, args.max_height_cm)
    finally:
        session.close()
    finish(session)


def run_hover(args) -> None:
    session = Session("hover", args.host, record=args.record)
    try:
        start = countdown_and_takeoff(session, args.min_battery)
        if start:
            items = start + [Hold("hover", args.seconds)]
            if args.ladder:
                items += ladder_items(session, args.ladder, 8.0)
            fly(session, items, args.fence_cm, args.max_height_cm)
    finally:
        session.close()
    finish(session)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def drone_args(p, flies: bool):
        p.add_argument("--host", default=default_host(), help="drone IP (default from settings.json)")
        p.add_argument("--record", action="store_true", help="also save video.mp4")
        if flies:
            p.add_argument("--fence-cm", type=float, default=60,
                           help="land if the pad reads further than this from the drone (default 60)")
            p.add_argument("--max-height-cm", type=float, default=170, help="land above this tof (default 170)")
            p.add_argument("--min-battery", type=int, default=30, help="refuse to take off below this")

    p = sub.add_parser("check", help="no flight: link, battery, frame size, rates, live view")
    drone_args(p, flies=False)
    p.set_defaults(fn=run_check)

    p = sub.add_parser("board", help="no drone: show the chessboard to measure")
    p.set_defaults(fn=run_board)

    p = sub.add_parser("calib", help="no flight: point the Tello at the on-screen board")
    p.add_argument("--host", default=default_host())
    p.add_argument("--square-mm", type=float, help="square size on this screen (default: from the screen size)")
    p.add_argument("--frames", type=int, default=20, help="views to capture (default 20)")
    p.add_argument("--from-dir", help="recompute from a saved calib run folder, no drone")
    p.set_defaults(fn=run_calib)

    p = sub.add_parser("tilt", help="no flight: drone level, board on the laptop ahead, camera pitch")
    p.add_argument("--host", default=default_host())
    p.add_argument("--square-mm", type=float, help="square size on this screen (default: from the screen size)")
    p.add_argument("--lens-cm", type=float, help="camera lens height above the floor (asked for if left out)")
    p.add_argument("--board-cm", type=float, help="board centre height above the floor (asked for if left out)")
    p.add_argument("--frames", type=int, default=60, help="frames to average (default 60)")
    p.set_defaults(fn=run_tilt)

    p = sub.add_parser("tape", help="no drone: type in the tape measurements")
    p.set_defaults(fn=run_tape)

    p = sub.add_parser("steps", help="flies: rc steps on one axis")
    drone_args(p, flies=True)
    p.add_argument("--axis", choices=tuple(AXES), required=True)
    p.add_argument("--amps", type=int, nargs="+", help="rc amplitudes, default per axis")
    p.add_argument("--seconds", type=float, help="seconds per step (default 1, yaw 2)")
    p.add_argument("--rest", type=float, default=2.5, help="hover between steps (default 2.5 s)")
    p.add_argument("--repeat", type=int, default=1, help="run the whole set this many times")
    p.set_defaults(fn=run_steps)

    p = sub.add_parser("hover", help="flies: hands-off hover, optional descent ladder")
    drone_args(p, flies=True)
    p.add_argument("--seconds", type=float, default=30, help="hover time (default 30)")
    p.add_argument("--ladder", type=int, default=0, metavar="N",
                   help="then step down 20 cm N times, 8 s each; only moves while tof > 50 cm")
    p.set_defaults(fn=run_hover)

    p = sub.add_parser("summary", help="re-print the numbers from a run folder")
    p.add_argument("folder")
    p.set_defaults(fn=lambda a: print(summarise(Path(a.folder))))

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
