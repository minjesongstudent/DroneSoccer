from __future__ import annotations

import argparse
import collections
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

from src import baseline, contract
from src.detect import CONFIRM_FRAMES, HoopDetector, draw
from src.params import CALIB_PATH, ROOT, Cage, Camera, Detector, Drone, Hoop

PARAMS_PATH = ROOT / "src" / "params.py"

Tello.LOGGER.setLevel(logging.WARNING)

OUT = ROOT / "runs" / "lab"
DT = 0.1
FIELDS = ("mid", "x", "y", "z", "pitch", "roll", "yaw", "vgx", "vgy", "vgz",
          "templ", "temph", "tof", "h", "bat", "baro", "time", "agx", "agy", "agz")
AXES = {"lr": 0, "fb": 1, "ud": 2, "yaw": 3}
STEP_DEFAULTS = {
    "fb": ((15, 30, 45), 1.0),
    "lr": ((15, 30, 45), 1.0),
    "ud": ((30, 60), 1.0),
    "yaw": ((25, 50, 100), 2.0),
}
BOARD_CORNERS = (9, 6)
KEY_SPACE, KEY_ESC = 32, 27
TOO_HOT_C = 88


def default_host() -> str:
    try:
        host = json.loads((ROOT / "settings.json").read_text())["drone_ip"].strip()
        return host or Tello.TELLO_IP
    except (OSError, ValueError, KeyError, AttributeError):
        return Tello.TELLO_IP



@dataclass
class Hold:
    label: str
    seconds: float
    rc: Optional[tuple] = (0, 0, 0, 0)


@dataclass
class Do:
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
        self.shift = (0.0, 0.0)
        self.moved = np.zeros(2)
        self.frame_bgr = None
        self._placeholder_at = -1.0
        self.decorate: Optional[Callable[[np.ndarray], np.ndarray]] = None

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
            self.drone.set_mission_pad_detection_direction(0)
            self.event("mission pads on, downward")
        except Exception as ex:
            self.event(f"mission pads not enabled: {ex}")
        if video:
            self._start_video()


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


    def _start_video(self) -> None:
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
        if self.reader is None:
            return False
        frame = self.reader.frame
        if frame is None or frame is self._last_frame or frame.shape[:2] == (300, 400):
            return False
        self._last_frame = frame
        self.frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = frame.shape[:2]
        small = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), (160, 120)).astype(np.float32)
        shift = (0.0, 0.0)
        if self._last_small is not None:
            shift, _ = cv2.phaseCorrelate(self._last_small, small)
        self._last_small = small
        scale = width / 160
        self.shift = (shift[0] * scale, shift[1] * scale)
        self.moved = self.moved + np.array(self.shift)
        self._frames.writerow([f"{self.now():.4f}", self.phase, width, height,
                               f"{shift[0] * scale:.2f}", f"{shift[1] * scale:.2f}"])
        if self.record:
            if self.writer is None:
                self.writer = cv2.VideoWriter(str(self.dir / "video.mp4"),
                                              cv2.VideoWriter_fourcc(*"mp4v"), 30, (width, height))
            self.writer.write(self.frame_bgr)
        return True

    def show(self, lines: list, warn: str = "") -> int:
        fresh = self.poll_frame()
        if not fresh and self.frame_bgr is None and self.now() - self._placeholder_at > 0.1:
            self._placeholder_at = self.now()
            fresh = True
        if fresh:
            view = self.frame_bgr.copy() if self.frame_bgr is not None else np.zeros((360, 480, 3), np.uint8)
            if self.decorate is not None and self.frame_bgr is not None:
                view = self.decorate(view)
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



def fly(session: Session, items: list, fence_cm: float, max_height_cm: float) -> None:
    queue = list(items)
    until, rc = 0.0, None
    worker: Optional[threading.Thread] = None
    failure: list = []
    land_wanted = land_started = False
    next_tick = session.now()
    warn = ""
    fix, drift = 0.0, np.zeros(2)
    fixes: list = []
    last_packet, last_t = None, session.now()
    too_high = 0

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
                tof = s.get("tof", 0)
                too_high = too_high + 1 if 10 < tof < 1000 and tof > max_height_cm else 0
                last_packet, last_t = s, now
            reach = fix + float(np.hypot(*drift))
            if session.drone.is_flying and not land_wanted:
                if too_high >= 2:
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
                sent = rc() if callable(rc) else rc
                if sent is None:
                    until, rc = now, None
                    continue
                session.send_rc(sent)
                next_tick += DT
                if now - next_tick > DT:
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
    def recentre() -> None:
        mid = pad_seen(session, 1.5)
        s = session.state()
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


PAD_SEARCH_STEPS = 2
PAD_LOOK_CM = 100


def pad_seen(session: Session, wait: float = 1.0) -> int:
    deadline = time.perf_counter() + wait
    while time.perf_counter() < deadline:
        mid = session.state().get("mid", -1)
        if mid > 0:
            return mid
        time.sleep(0.05)
    return 0


def to_pad_items(session: Session, may_back_up: Callable[[], bool] = lambda: False) -> list:
    def to_pad() -> None:
        mid = pad_seen(session, 1.5)
        for _ in range(PAD_SEARCH_STEPS if may_back_up() else 0):
            if mid:
                break
            session.drone.move_back(25)
            mid = pad_seen(session)
        if not mid:
            session.event("! pad not found: landing where it is")
            return
        session.drone.go_xyz_speed_mid(0, 0, 60, 30, mid)
        session.event(f"over pad {mid}")
    return [to_height_item(session, PAD_LOOK_CM), Do("to pad", to_pad), Hold("over pad", 1.0)]


def step_items(session: Session, axis: str, amps: tuple, seconds: float, rest: float, repeat: int) -> list:
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
    items.append(recentre_item(session))
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



def screen() -> Optional[tuple]:
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
SQUARE_PATH = OUT / "square_mm.json"


def board_layout() -> tuple:
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
    cv2.rectangle(img, (left, top), (left + cols * sq - 1, top + rows * sq - 1), 128, 2)
    return img


def screen_square_mm() -> Optional[float]:
    return board_layout()[2] * SCREEN[2] if SCREEN else None


def saved_square_mm() -> Optional[float]:
    try:
        saved = json.loads(SQUARE_PATH.read_text())
        return float(saved["square_mm"]) if saved.get("screen") == list(board_layout()[:2]) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def square_mm_arg(args) -> float:
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
    if hint:
        cv2.putText(img, hint, (left, height - top + sq // 2), cv2.FONT_HERSHEY_SIMPLEX, scale * 1.3,
                    (200, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("board", img)


CALIB_TARGETS = [{0: x, 1: y, 2: 0.35} for y in (0.25, 0.5, 0.75) for x in (0.25, 0.5, 0.75)] + [
    {0: 0.5, 1: 0.5, 2: 0.2}, {0: 0.5, 1: 0.5, 2: 0.55},
    {0: 0.5, 1: 0.5, 3: 0.12}, {0: 0.5, 1: 0.5, 3: -0.12},
    {0: 0.5, 1: 0.5, 4: 0.15}, {0: 0.5, 1: 0.5, 4: -0.15},
]
TARGET_TOL = {0: 0.12, 1: 0.12, 2: 0.08, 3: 0.07, 4: 0.07}
TARGET_NAMES = ["top left", "top middle", "top right", "middle left", "centre", "middle right",
                "bottom left", "bottom middle", "bottom right", "far", "near",
                "from above", "from below", "from the left", "from the right"]
HINT_MIN_SECONDS = 3.0
HINT_SETTLE_SECONDS = 1.0
STILL_SECONDS = 0.6


def meets(sig: np.ndarray, target: dict) -> bool:
    return all(abs(sig[i] - v) < TARGET_TOL[i] for i, v in target.items())


def missing_targets(taken: list) -> list:
    return [i for i, target in enumerate(CALIB_TARGETS) if not any(meets(sig, target) for sig in taken)]


def direction(sig: Optional[np.ndarray], target: Optional[dict]) -> str:
    if target is None:
        return "All views done. Any extra angle helps, or press C to finish"
    if sig is None:
        return "Board not seen: point the drone at the screen, 40-80 cm away"
    for i, v in target.items():
        d = v - sig[i]
        if abs(d) < TARGET_TOL[i]:
            continue
        if i == 0:
            return "Point the drone a little to its " + ("LEFT" if d > 0 else "RIGHT")
        if i == 1:
            return "Point the drone's nose a little " + ("UP" if d > 0 else "DOWN")
        if i == 2:
            return "Move the drone " + ("CLOSER to" if d > 0 else "FURTHER from") + " the screen"
        if i == 3:
            return ("Raise the drone ABOVE the screen, nose tipped down at it" if d > 0
                    else "Lower the drone BELOW the screen, nose tipped up at it")
        if i == 4:
            return ("Move the drone to its LEFT, then turn it back to face the screen" if d > 0
                    else "Move the drone to its RIGHT, then turn it back to face the screen")
    return "Hold still..."


class SlowHint:

    def __init__(self, first: str):
        self.shown, self.shown_at = first, -HINT_MIN_SECONDS
        self.pending, self.pending_at = first, 0.0

    def update(self, wanted: str, now: float) -> str:
        if wanted != self.pending:
            self.pending, self.pending_at = wanted, now
        if (wanted != self.shown and now - self.pending_at >= HINT_SETTLE_SECONDS
                and now - self.shown_at >= HINT_MIN_SECONDS):
            self.shown, self.shown_at = wanted, now
        return self.shown


def find_board(gray: np.ndarray, fast: bool):
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
    if fast:
        flags |= cv2.CALIB_CB_FAST_CHECK
    found, corners = cv2.findChessboardCorners(gray, BOARD_CORNERS, flags=flags)
    return corners.reshape(-1, 2) if found else None


def pose_signature(corners: np.ndarray, width: int, height: int) -> np.ndarray:
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
    names, img_points, sigs, size = [], [], [], None
    for path in images:
        gray = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2GRAY)
        size = gray.shape[::-1]
        corners = find_board(gray, fast=False)
        if corners is None:
            print(f"  {path.name}: board not found again, skipped")
            continue
        corners = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2).astype(np.float32), (5, 5), (-1, -1),
                                   (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001))
        names.append(path.name)
        img_points.append(corners)
        sigs.append(pose_signature(corners.reshape(-1, 2), *size))

    while True:
        if len(img_points) < 8:
            print(f"Only {len(img_points)} usable views. Need at least 8, ideally 15 or more.")
            return
        rms, K, dist, _, _, std, _, per_view = cv2.calibrateCameraExtended(
            [objp] * len(img_points), img_points, size, None, None)
        worst = int(np.argmax(per_view))
        if per_view[worst, 0] <= 1.0:
            break
        print(f"  dropped {names[worst]}: {per_view[worst, 0]:.2f} px, probably blurred")
        for items in (names, img_points, sigs):
            del items[worst]

    std = std.ravel()
    print(f"\nCalibrated from {len(img_points)} views at {size[0]}x{size[1]}")
    print(f"  RMS reprojection error {rms:.3f} px  (under 0.5 good, under 1.0 usable)")
    print(f"  fx {K[0, 0]:.1f} +- {std[0]:.1f}   fy {K[1, 1]:.1f} +- {std[1]:.1f}")
    print(f"  cx {K[0, 2]:.1f} +- {std[2]:.1f}   cy {K[1, 2]:.1f} +- {std[3]:.1f}")
    print(f"  dist {np.round(dist.ravel(), 4)}")
    print(f"  horizontal field of view {np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0]))):.1f} deg")
    missing = missing_targets(sigs)
    print("  coverage: all 15 views" if not missing else
          "  coverage: missing " + ", ".join(TARGET_NAMES[i] for i in missing))
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
    previous, still_since = None, None
    flash_until = 0.0
    hint = SlowHint("Point the drone at the screen")
    try:
        while True:
            missing = missing_targets(taken)
            if not missing and len(taken) >= args.frames:
                break
            target = CALIB_TARGETS[missing[0]] if missing else None
            step = f"view {len(taken) + 1}: {TARGET_NAMES[missing[0]]}" if missing else "extras"
            status = (f"{len(CALIB_TARGETS) - len(missing)}/{len(CALIB_TARGETS)} views. "
                      f"Now: {step}.  Esc stop, C finish")
            if session.now() < flash_until:
                status = "Got it! Next one..."
            show_board(status, flash=session.now() < flash_until, hint=hint.shown)
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_ESC:
                return
            if key in (ord("c"), ord("C")) and len(taken) >= 8:
                break
            if not session.poll_frame():
                continue
            frame = session.frame_bgr
            height, width = frame.shape[:2]
            corners = find_board(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), fast=True)
            sig = None if corners is None else pose_signature(corners, width, height)
            hint.update(direction(sig, target), session.now())
            if corners is None:
                previous, still_since = None, None
                continue
            moving = previous is None or np.abs(corners - previous).mean() > 1.5
            previous = corners
            if moving:
                still_since = None
                continue
            still_since = still_since or session.now()
            fills = meets(sig, target) if target is not None else \
                all(np.abs(sig - other).max() > 0.08 for other in taken)
            if fills and session.now() - still_since >= STILL_SECONDS and session.now() > flash_until:
                cv2.imwrite(str(session.dir / f"calib_{len(taken):02d}.png"), frame)
                taken.append(sig)
                flash_until = session.now() + 1.5
                session.event(f"captured view {len(taken)} ({step})")
    finally:
        session.close()
    calibrate(session.dir, square_mm)



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
    s = np.maximum(s, 0.0)
    decay = np.exp(-s / tau)
    return s - tau * (1 - decay) if integrated else 1 - decay


def fit_step(t, y, on, off, integrated=False, delay=None, tau=None) -> dict:
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
        gaps = gaps[gaps < 1.0]
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
    if (folder / "detect.csv").exists():
        out += summarise_detect(load(folder / "detect.csv"))
    try:
        from src import sysid
        out += sysid.summary_lines(folder)
    except Exception as ex:
        out.append(f"  sysid        not fitted: {ex}")
    if (folder / "survey.csv").exists():
        out += summarise_survey(folder)
    out += summarise_holds(state, cmds)
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



RC_FULL = (50, 50, 50, 70)


class Watcher:

    def __init__(self, session: "Session"):
        self.session = session
        self.detector = HoopDetector(confirm=CONFIRM_FRAMES)
        self.det = None
        self.obs = contract.NOT_VISIBLE.copy()
        self.rc = (0, 0, 0, 0)
        self.in_view = True
        self._frame = None
        self._file = open(session.dir / "detect.csv", "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["t", "phase", "in_view", "ms", *contract.FIELDS, "rc_lr", "rc_fb", "rc_ud", "rc_yaw"])
        session.decorate = self.draw

    def update(self) -> bool:
        frame = self.session.frame_bgr
        if frame is None or frame is self._frame:
            return False
        self._frame = frame
        t0 = time.perf_counter()
        self.det = self.detector(frame)
        ms = (time.perf_counter() - t0) * 1000
        self.obs = self.det.obs
        self._csv.writerow([f"{self.session.now():.4f}", self.session.phase, int(self.in_view), f"{ms:.1f}",
                            *(f"{v:.4f}" for v in self.obs), *self.rc])
        return True

    def draw(self, view: np.ndarray) -> np.ndarray:
        if self.det is None:
            return view
        view = draw(view, self.det)
        if not self.in_view:
            cv2.putText(view, "operator: hoop NOT in view (h)", (12, view.shape[0] - 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2, cv2.LINE_AA)
        return view

    def close(self) -> None:
        self._file.close()


def summarise_detect(d: dict) -> list:
    if not d:
        return []
    out = []
    say = out.append
    vis, marker, in_view = d["visible"] > 0, d["marker"] > 0, d["in_view"] > 0
    say(f"  detector     {len(vis)} frames, {np.median(d['ms']):.0f} ms median, p95 {np.percentile(d['ms'], 95):.0f} ms")
    if in_view.any():
        say(f"  detected     {vis[in_view].mean():6.1%} of frames the operator marked in view (target > 95%)")
    if (~in_view).any():
        say(f"  false hoop   {vis[~in_view].mean():6.1%} of frames marked not in view")
    if vis.any():
        say(f"  marker       {marker[vis].mean():6.1%} of frames with the hoop seen")
        both = vis[1:] & vis[:-1]
        if both.sum() >= 5:
            jit = {k: np.std(np.diff(d[k])[both]) / np.sqrt(2) for k in ("u", "v", "major", "ratio")}
            say("  jitter       " + "  ".join(f"{k} {v:.4f}" for k, v in jit.items())
                + "   (per frame; hold still for this to mean noise)")
        gaps, run = [], 0
        for seen in vis[np.argmax(vis):]:
            if seen:
                if run:
                    gaps.append(run)
                run = 0
            else:
                run += 1
        if gaps:
            say(f"  dropouts     {len(gaps)}, longest {max(gaps)} frames, median {int(np.median(gaps))}")
    return out


def run_hoop(args) -> None:
    session = Session("hoop", args.host, record=args.record)
    watcher = Watcher(session)
    try:
        session.set_phase("view")
        session.event("detector live. Point the drone at the hoop; h marks the hoop out of view and back. Esc to stop")
        while True:
            watcher.update()
            key = session.show(session.status_lines()[:1] + ["h: hoop in/out of view   Esc: stop"])
            if key in (KEY_ESC, ord("q")):
                break
            if key in (ord("h"), ord("H")):
                watcher.in_view = not watcher.in_view
                session.event(f"operator: hoop {'in' if watcher.in_view else 'out of'} view")
    except KeyboardInterrupt:
        pass
    finally:
        watcher.close()
        session.close()
    finish(session)



RATE_AMPS = {"ud": (30, 50, 70, 100), "yaw": (5, 10, 20, 40, 70)}
RATE_SECONDS = 3.0
RATE_TOP_CM, RATE_BOTTOM_CM = 165, 60
SETTLE_S = {"ud": 0.5, "yaw": 0.8}

NEAR, FAR = 0.65, 0.45
FULL_VIEW = 0.97
SLIDE_RC = {"left": (0, 0, 0, 25), "right": (0, 0, 0, -25), "top": (0, 0, -40, 0)}
SLIDE_REPEATS = 3
SLIDE_MAX_ERROR = 0.1
SLIDE_FLOOR_CM = 70
ORBIT_RC = 20
ORBIT_ANGLES = (20, 35, 50)
ORBIT_HOLD_S = 3.0
ORBIT_TOO_CLOSE = 0.6


def rim_fraction(ellipse, width: int, height: int, dx: float = 0.0, dy: float = 0.0):
    (cx, cy), (w, h), ang = ellipse
    t = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    th = np.deg2rad(ang)
    x = cx + dx + w / 2 * np.cos(t) * np.cos(th) - h / 2 * np.sin(t) * np.sin(th)
    y = cy + dy + w / 2 * np.cos(t) * np.sin(th) + h / 2 * np.sin(t) * np.cos(th)
    out = {"left": (x < 0).sum(), "right": (x >= width).sum(), "top": (y < 0).sum(), "bottom": (y >= height).sum()}
    inside = float(np.mean((x >= 0) & (x < width) & (y >= 0) & (y < height)))
    edge = max(out, key=out.get) if inside < 1.0 else ""
    return inside, edge


def fully_in_frame(u, v, major) -> np.ndarray:
    hx, hy = Camera().half_extent
    return (np.abs(u) + major / 2 / hx < 0.98) & (np.abs(v) + major / 2 / hy < 0.98)


def to_height_item(session: Session, cm: float) -> Do:
    def go() -> None:
        tof = session.state().get("tof", 0)
        if not 10 < tof < 1000:
            session.event(f"to {cm:.0f} cm: tof {tof} unreadable, skipped")
            return
        diff = cm - tof
        if abs(diff) >= 20:
            (session.drone.move_up if diff > 0 else session.drone.move_down)(int(min(abs(diff), 500)))
    return Do(f"to {cm:.0f} cm", go)


def climb_rc(session: Session, amp: int) -> Callable:
    stopped = [False]

    def rc():
        s = session.state()
        tof = s.get("tof", 0)
        up = -10.0 * s.get("vgz", 0)
        ahead = tof + 0.6 * up
        if not 10 < tof < 1000 or (amp > 0 and ahead >= RATE_TOP_CM) or (amp < 0 and ahead <= RATE_BOTTOM_CM):
            stopped[0] = True
        return (0, 0, 0, 0) if stopped[0] else (0, 0, amp, 0)
    return rc


def rate_items(session: Session, axis: str) -> list:
    items = []
    for amp in RATE_AMPS[axis]:
        if axis == "ud":
            items += [to_height_item(session, RATE_BOTTOM_CM + 10), recentre_item(session), Hold("steady", 1.5),
                      Hold(f"hold ud {amp:+d}", RATE_SECONDS, climb_rc(session, amp)), Hold("rest", 2.0),
                      Hold(f"hold ud {-amp:+d}", RATE_SECONDS, climb_rc(session, -amp)), Hold("rest", 2.0)]
        else:
            items.append(recentre_item(session))
            for sign in (1, -1):
                items += [Hold("steady", 1.0), Hold(f"hold yaw {sign * amp:+d}", RATE_SECONDS, (0, 0, 0, sign * amp)),
                          Hold("rest", 1.5)]
    return items


class HoopSurvey:

    def __init__(self, session: Session, behind: bool):
        self.session = session
        self.behind = behind
        self.detector = HoopDetector(confirm=1)
        slides = [s for edge in SLIDE_RC for n in range(1, SLIDE_REPEATS + 1) for s in (f"slide {edge} {n}", "centre far")]
        self.queue = (["find", "centre far"] + slides
                      + ["centre near", "centre far", "orbit left", "centre far", "orbit right", "centre far"])
        self.stage, self.started, self.good_since, self.phase = None, 0.0, None, ""
        self.last_frame, self.obs, self.det = None, contract.NOT_VISIBLE.copy(), None
        self.lost_since = None
        self.flying = False
        self.finished = False
        self.ratios = collections.deque(maxlen=15)
        self.anchor = None
        self.hold = ""
        self._row = None
        self._file = open(session.dir / "survey.csv", "w", newline="")
        self._csv = csv.writer(self._file)
        self._csv.writerow(["t", "stage", "phase", "hold", "detected", "marker", "u", "v", "major", "ratio",
                            "rim", "side", "tof", "rc_lr", "rc_fb", "rc_ud", "rc_yaw"])
        self._slides_file = open(session.dir / "slides.csv", "w", newline="")
        self._slides = csv.writer(self._slides_file)
        self._slides.writerow(["slide", "edge", "valid", "error", "why"])
        session.decorate = self.draw
        self._next()

    def _next(self):
        self.stage = self.queue.pop(0) if self.queue else None
        self.finished = self.stage is None
        self.started, self.good_since, self.phase, self.anchor, self.hold = self.session.now(), None, "", None, ""
        if self.stage:
            self.session.event(f"survey: {self.stage}")

    def close(self):
        self._file.close()
        self._slides_file.close()

    @property
    def face_on(self) -> float:
        return float(np.median(self.ratios)) if len(self.ratios) >= 5 else 0.95


    def predicted(self):
        if self.anchor is None:
            return None
        (cx, cy), size, ang = self.anchor["ellipse"]
        dx, dy = self.session.moved - self.anchor["moved"]
        if self.anchor["edge"] == "top":
            tof = self.session.state().get("tof", 0)
            if 10 < tof < 1000:
                dy = -Camera().fy * (self.anchor["tof"] - tof) / 100 / self.anchor["distance"]
            else:
                dy = self.anchor.get("last_dy", 0.0)
            self.anchor["last_dy"] = dy
        return ((cx + dx, cy + dy), size, ang)

    def draw(self, view):
        if self.det is not None:
            view = draw(view, self.det)
        p = self.predicted()
        if p is not None:
            cv2.ellipse(view, p, (0, 255, 255), 2)
        return view

    def _see(self) -> bool:
        s = self.session
        frame = s.frame_bgr
        if frame is None or frame is self.last_frame:
            return False
        self.last_frame = frame
        self.size = frame.shape[1::-1]
        self.det = self.detector(frame)
        self.obs = self.det.obs
        p = self.predicted()
        rim = rim_fraction(p, *self.size)[0] if p is not None else np.nan
        self.rim = rim
        self._row = [f"{s.now():.4f}", self.stage, self.phase, self.hold, int(self.obs[6] > 0), int(self.obs[7] > 0),
                     *(f"{v:.4f}" for v in self.obs[:4]), f"{rim:.3f}",
                     "back" if self.behind else "front", s.state().get("tof", "")]
        return True

    def full_view(self) -> bool:
        return bool(self.obs[contract.VISIBLE]) and rim_fraction(self.det.ellipse_px, *self.size)[0] >= FULL_VIEW


    @staticmethod
    def _v_target(major: float) -> float:
        half = major / 2 / Camera().half_extent[1]
        return max(baseline.V_LEVEL, -1 + 1.1 * half)

    def _centre(self, target: float, lr: int = 0) -> tuple:
        u, v, major = self.obs[0], self.obs[1], self.obs[2]
        yaw = int(np.clip(60 * u, -30, 30))
        ud = int(np.clip(-60 * (v - self._v_target(major)), -30, 30))
        fb = int(np.clip(60 * (target - major) / target, -20, 20))
        return (lr, fb, ud, yaw)

    def __call__(self):
        if not self.flying:
            self.flying, self.started = True, self.session.now()
        if not self._see():
            return self.session.cmd
        rc = self._step()
        if rc is None:
            self.close()
            return None
        self._csv.writerow(self._row + list(rc))
        return rc

    def _step(self):
        s, now = self.session, self.session.now()
        if self.stage is None:
            return None
        seen = bool(self.obs[contract.VISIBLE])
        age = now - self.started
        self.lost_since = None if seen else (self.lost_since or now)
        kind, _, arg = self.stage.partition(" ")

        if kind == "find":
            if self.full_view():
                self._next()
                return (0, 0, 0, 0)
            if age > 40:
                s.event("! survey: hoop not found in 40 s; is the drone facing it?")
                return None
            if seen:
                return self._centre(FAR)
            tof = s.state().get("tof", 0)
            return (0, 0, 35, 0) if tof < RATE_TOP_CM else (0, 0, 0, 20)

        if kind == "centre":
            if self.lost_since and now - self.lost_since > 4.0:
                s.event("survey: hoop lost, searching again")
                self.queue.insert(0, self.stage)
                self.stage, self.started = "find", now
                return (0, 0, 0, 0)
            if not seen:
                return (0, 0, 0, 0)
            target = NEAR if arg == "near" else FAR
            rc = self._centre(target)
            ok = abs(self.obs[0]) < 0.12 and abs(self.obs[1] - self._v_target(self.obs[2])) < 0.15 \
                and abs(self.obs[2] - target) / target < 0.15
            if ok and arg == "far":
                self.ratios.append(float(self.obs[3]))
            self.good_since = (self.good_since or now) if ok else None
            if (self.good_since and now - self.good_since > 1.5) or age > 15:
                self._next()
            return rc

        if kind == "slide":
            return self._slide(arg.split()[0], seen, now, age)
        if kind == "orbit":
            return self._orbit(arg, seen, now, age)
        return None

    def _slide(self, edge: str, seen: bool, now: float, age: float):
        rc = SLIDE_RC[edge]
        if self.phase == "":
            if self.full_view():
                self.anchor = {"ellipse": self.det.ellipse_px, "moved": self.session.moved.copy(), "edge": edge,
                               "tof": self.session.state().get("tof", 0),
                               "distance": Hoop().outer_height / max(float(self.obs[2]), 1e-3)}
                self.phase, self.started = "out", now
                return rc
            if age > 3.0:
                self._end_slide(edge, False, np.nan, "no full view to start from")
            return (0, 0, 0, 0)
        if self.phase == "out":
            tof = self.session.state().get("tof", 0)
            gone = (np.isfinite(self.rim) and self.rim < 0.15) or \
                   (self.lost_since and now - self.lost_since > 0.5 and np.isfinite(self.rim) and self.rim < 0.35)
            low = edge == "top" and 10 < tof < SLIDE_FLOOR_CM
            if gone or low or age > 8.0:
                self.phase, self.started = "back", now
                return (0, 0, 0, 0)
            return rc
        if self.full_view():
            (px, py), _, _ = self.predicted()
            (cx, cy), (w, h), _ = self.det.ellipse_px
            error = float(np.hypot(px - cx, py - cy) / max(w, h))
            self._end_slide(edge, error <= SLIDE_MAX_ERROR, error,
                            "" if error <= SLIDE_MAX_ERROR else "prediction drifted")
            return (0, 0, 0, 0)
        if age > 10.0:
            self._end_slide(edge, False, np.nan, "not found again")
            return (0, 0, 0, 0)
        return tuple(-x for x in rc)

    def _end_slide(self, edge, valid, error, why):
        self._slides.writerow([self.stage, edge, int(valid), f"{error:.3f}", why])
        self._slides_file.flush()
        self.session.event(f"survey: {self.stage} " + ("ok" if valid else f"dropped ({why})")
                           + (f", error {error:.2f}" if np.isfinite(error) else ""))
        self._next()

    def _orbit(self, side: str, seen: bool, now: float, age: float):
        lr = ORBIT_RC if side == "right" else -ORBIT_RC
        if self.phase == "":
            self.phase, self.targets = "move", list(ORBIT_ANGLES)
        if self.phase != "back" and (not seen and self.lost_since and now - self.lost_since > 1.0
                                     or seen and self.obs[2] > ORBIT_TOO_CLOSE or age > 30.0):
            self.session.event(f"survey: {self.stage} cut short at {self.hold or 'start'}")
            self.phase, self.started, self.hold = "back", now, ""
        if self.phase == "move":
            angle = self.targets[0]
            if seen and self.obs[3] <= self.face_on * np.cos(np.deg2rad(angle)):
                self.phase, self.hold, self.hold_from = "hold", str(angle), now
            return self._centre(FAR, lr) if seen else (0, 0, 0, 0)
        if self.phase == "hold":
            if now - self.hold_from >= ORBIT_HOLD_S:
                self.targets.pop(0)
                self.hold = ""
                self.phase = "move" if self.targets else "back"
                self.started = now if self.phase == "back" else self.started
            return self._centre(FAR) if seen else (0, 0, 0, 0)
        if seen and self.obs[3] >= self.face_on - 0.03 or now - self.started > 10.0:
            self._next()
            return (0, 0, 0, 0)
        return self._centre(FAR, -lr) if seen else (-lr, 0, 0, 0)


def survey_parts(args) -> list:
    return args.parts or (["rates-ud", "rates-yaw", "hoop"] if not args.behind else ["hoop"])


def run_survey(args) -> None:
    session = Session("survey", args.host, record=args.record)
    hoop = None
    try:
        start = countdown_and_takeoff(session, args.min_battery)
        if start:
            items = list(start)
            parts = survey_parts(args)
            if "rates-ud" in parts:
                items += rate_items(session, "ud")
            if "rates-yaw" in parts:
                items += rate_items(session, "yaw")
            if "hoop" in parts:
                hoop = HoopSurvey(session, args.behind)
                items += [Hold("hoop survey", 240.0, hoop)]
            items += to_pad_items(session, lambda: hoop is not None and hoop.finished)
            fly(session, items, args.fence_cm, args.max_height_cm)
    finally:
        if hoop is not None:
            hoop.close()
        session.close()
    finish(session)


AXIS_COL = {"ud": "ud", "yaw": "yaw"}


def summarise_holds(state: dict, cmds: dict) -> list:
    if not state or not cmds:
        return []
    rows = []
    for label in dict.fromkeys(p for p in cmds["phase"] if str(p).startswith("hold ")):
        _, axis, amp = label.split()
        amp = int(amp)
        m = (cmds["phase"] == label) & (cmds[AXIS_COL[axis]] != 0)
        if m.sum() < 3:
            continue
        a, b = float(cmds["t"][m][0]) + SETTLE_S[axis], float(cmds["t"][m][-1]) + 0.15
        w = (state["t"] >= a) & (state["t"] <= b)
        if axis == "ud":
            y, scale = state["tof"], 0.01
            w = deglitch(y, w & (y > 10) & (y < 1000), limit=12.0)
        else:
            y, scale = np.degrees(np.unwrap(np.radians(state["yaw"]))), 1.0
        rate = np.nan
        if w.sum() >= 4 and b - a >= 0.4:
            tt, yy = state["t"][w], y[w]
            i, j = np.triu_indices(len(tt), 1)
            ok = tt[j] - tt[i] > 0.15
            rate = float(np.median((yy[j] - yy[i])[ok] / (tt[j] - tt[i])[ok]) * scale)
        rows.append({"axis": axis, "amp": amp, "rate": rate, "steady_s": max(0.0, b - a)})
    if not rows:
        return []
    drone = Drone()
    out = ["", "  hold          steady   measured        simulator for this rc"]
    for r in rows:
        if r["axis"] == "ud":
            sim = r["amp"] / RC_FULL[2] * drone.max_speed
            out.append(f"  ud  {r['amp']:+4d}     {r['steady_s']:4.1f} s   {r['rate']:+6.2f} m/s      {sim:+.2f} m/s")
        else:
            sim = r["amp"] / RC_FULL[3] * drone.max_yaw_rate_deg
            out.append(f"  yaw {r['amp']:+4d}     {r['steady_s']:4.1f} s   {r['rate']:+6.1f} deg/s    {sim:+.1f} deg/s")
    def crossing(pts, value):
        pts = sorted(pts)
        for (a0, r0), (a1, r1) in zip(pts, pts[1:]):
            if r0 < value <= r1:
                return a0 + (value - r0) / (r1 - r0) * (a1 - a0)
        return None

    for sign, name in ((1, "up"), (-1, "down")):
        pts = [(abs(r["amp"]), abs(r["rate"])) for r in rows if r["axis"] == "ud" and np.sign(r["amp"]) == sign
               and np.isfinite(r["rate"]) and r["steady_s"] >= 0.4]
        rc = crossing(pts, drone.max_up_speed)
        if rc is not None:
            out.append(f"  -> ud {name}: {drone.max_up_speed} m/s (the simulator's full up/down) at rc {rc:.0f}. "
                       f"RC_FULL ud is {RC_FULL[2]}")
    pts = [(abs(r["amp"]), abs(r["rate"])) for r in rows if r["axis"] == "yaw" and np.isfinite(r["rate"])]
    big = [p for p in pts if p[0] >= 10]
    if len(big) >= 2:
        k, c = np.polyfit(*zip(*big), 1)
        at_full = [v for a, v in pts if a == RC_FULL[3]]
        small = ", ".join(f"rc {a} = {v:.1f} deg/s" for a, v in pts if a < 10)
        out.append(f"  -> yaw: deadband ~rc {max(0.0, -c / k):.0f} (yaw_deadband {max(0.0, -c / k) / RC_FULL[3]:.2f}); "
                   + (f"{np.mean(at_full):.0f} deg/s at RC_FULL yaw {RC_FULL[3]} (max_yaw_rate_deg "
                      f"{drone.max_yaw_rate_deg:.0f})" if at_full else f"{k:.2f} deg/s per rc")
                   + (f". Small: {small}" if small else ""))
    return out


def summarise_survey(folder: Path) -> list:
    d = load(folder / "survey.csv")
    if not d or "phase" not in d:
        return []
    out = [""]
    stage = np.array([str(x) for x in d["stage"]])
    phase = np.array([str(x) if str(x) != "nan" else "" for x in d["phase"]])
    det = d["detected"] > 0
    slides = load(folder / "slides.csv") if (folder / "slides.csv").exists() else {}
    crossings = {}
    if slides:
        names = np.array([str(x) for x in slides["slide"]])
        edges = np.array([str(x) for x in slides["edge"]])
        valid = slides["valid"] > 0
        whys = np.array([str(x) for x in slides["why"]])
        out.append("  slides kept (prediction checked when the hoop was found again):")
        for edge in SLIDE_RC:
            m = edges == edge
            dropped = ", ".join(sorted(set(whys[m & ~valid]))) or "-"
            out.append(f"    {edge:6s} {int((m & valid).sum())}/{int(m.sum())} kept   dropped: {dropped}")
        bins = [(0.15, 0.35), (0.35, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)]
        out.append("  rim share in frame -> detected, kept slides only")
        out.append("    edge    " + "  ".join(f"{lo:.2f}-" for lo, _ in bins) + "   half detected at")
        for edge in SLIDE_RC:
            keep = names[(edges == edge) & valid]
            rows = np.isin(stage, keep) & np.isin(phase, ("out", "back")) & np.isfinite(d["rim"])
            rates = []
            cells = []
            for lo, hi in bins:
                b = rows & (d["rim"] >= lo) & (d["rim"] < hi)
                rates.append(det[b].mean() if b.sum() >= 3 else np.nan)
                cells.append(f"{rates[-1]:5.0%}" if b.sum() >= 3 else "    -")
            mids = [(lo + hi) / 2 for lo, hi in bins]
            cross = next((mids[i - 1] + (0.5 - rates[i - 1]) / (rates[i] - rates[i - 1]) * (mids[i] - mids[i - 1])
                          for i in range(1, len(bins)) if np.isfinite(rates[i - 1]) and np.isfinite(rates[i])
                          and rates[i - 1] < 0.5 <= rates[i]), None)
            if cross is not None:
                crossings[edge] = cross
            out.append(f"    {edge:6s}  " + "  ".join(cells) + (f"   {cross:.2f}" if cross is not None else "   -"))
        if crossings:
            out.append(f"  -> params.Detector.min_arc_fraction: {np.mean(list(crossings.values())):.2f} "
                       f"(edges {', '.join(f'{k} {v:.2f}' for k, v in crossings.items())}; "
                       f"randomise over that spread). Now {Detector().min_arc_fraction}")
    hold = np.asarray(d["hold"], dtype=float)
    whole = det & fully_in_frame(d["u"], d["v"], d["major"])
    side = str(d["side"][0])
    out.append(f"  marker reported, whole hoop in frame, {side} side "
               + ("(should be ~100%)" if side == "front" else "(should be 0%)"))
    groups = [("1 m, face-on", (stage == "centre far") & whole), ("0.7 m, face-on", (stage == "centre near") & whole)]
    groups += [(f"1 m, {a} deg off", (hold == a) & whole) for a in ORBIT_ANGLES]
    for name, m in groups:
        out.append(f"    {name:16s} {d['marker'][m].mean():5.0%}  ({int(m.sum())} frames)" if m.sum() >= 5
                   else f"    {name:16s}     -  ({int(m.sum())} frames)")
    if side == "front":
        far, near = groups[0][1], groups[1][1]
        if far.sum() >= 5 and near.sum() >= 5:
            out.append(f"  -> params.Detector.marker_miss {1 - d['marker'][far].mean():.2f}, "
                       f"marker_miss_near {1 - d['marker'][near].mean():.2f}. "
                       f"Now {Detector().marker_miss}, {Detector().marker_miss_near}")
    return out


def run_markercheck(args) -> None:
    from src import detect as det_mod
    folder = Path(args.folder)
    side = args.side
    if side is None and (folder / "survey.csv").exists():
        side = str(load(folder / "survey.csv")["side"][0])
    side = side or "front"
    limits = [tuple(det_mod.MARKER_WHITE_MAX_S)] + [tuple(int(x) for x in v.split(","))
                                                    for v in args.white_max_s]
    limits = list(dict.fromkeys(limits))
    detector = HoopDetector(confirm=1)
    cap = cv2.VideoCapture(str(folder / "video.mp4"))
    counts = {v: {"near": [0, 0], "far": [0, 0]} for v in limits}
    i = 0
    original = det_mod.MARKER_WHITE_MAX_S
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % args.every:
                continue
            found = detector(frame)
            u, v, major = found.obs[0], found.obs[1], found.obs[2]
            if not found.obs[contract.VISIBLE] or not fully_in_frame(u, v, major):
                continue
            small = cv2.GaussianBlur(cv2.resize(frame, None, fx=detector.scale, fy=detector.scale,
                                                interpolation=cv2.INTER_AREA), (5, 5), 0)
            (cx, cy), (w, h), ang = found.ellipse_px
            s_ = detector.scale
            ellipse = ((cx * s_, cy * s_), (w * s_, h * s_), ang)
            near = "near" if major >= Detector().near_major else "far"
            for limit in limits:
                det_mod.MARKER_WHITE_MAX_S = limit
                counts[limit][near][0] += detector._marker(small, ellipse) is not None
                counts[limit][near][1] += 1
    finally:
        det_mod.MARKER_WHITE_MAX_S = original
    print(f"{folder.name}: {side} side, whole hoop in view, every {args.every} frames. Marker reported "
          + ("(should be ~100%):" if side == "front" else "(should be 0%):"))
    for limit in limits:
        label = {"near": "closer than 0.8 m", "far": "0.8 m and further"}
        cells = "   ".join(f"{label[k]}: {c[0] / c[1]:4.0%} of {c[1]}" if c[1] else f"{label[k]}: -"
                           for k, c in counts[limit].items())
        now = "  (now)" if limit == original else ""
        print(f"  saturation {','.join(map(str, limit)):>7s}{now}:  {cells}")


OLD_LAGS = (24, 20, 16, 12, 8, 4, 2, 1, 0)
OLD_ACTIONS = 3


def old_policy_input(frames, heights, motion, actions, unseen: bool) -> np.ndarray:
    frames, actions = list(frames), list(actions)
    return np.concatenate([*(frames[-1 - lag] for lag in OLD_LAGS),
                           *actions[-OLD_ACTIONS:]]).astype(np.float32)


def flag_policy_input(frames, heights, motion, actions, unseen: bool) -> np.ndarray:
    return np.append(old_policy_input(frames, heights, motion, actions, unseen),
                     1.0 if unseen else 0.0).astype(np.float32)


LAGS60 = (60, 50, 40, 32, 24, 16, 12, 8, 4, 2, 1, 0)


def lags60_policy_input(frames, heights, motion, actions, unseen: bool) -> np.ndarray:
    frames, actions = list(frames), list(actions)
    return np.concatenate([*(frames[-1 - lag] for lag in LAGS60), *actions[-OLD_ACTIONS:],
                           [1.0 if unseen else 0.0]]).astype(np.float32)


SLOTS_V1_DIM = contract.SLOTS_V1_DIM


def slots_v1_policy_input(frames, heights, motion, actions, unseen: bool) -> np.ndarray:
    return contract.slots_v1(contract.policy_input(frames, heights, motion, actions, unseen))


def motion_from_state(state: dict, start_yaw: float) -> np.ndarray:
    speed = float(np.hypot(state.get("vgx", 0), state.get("vgy", 0))) / 10
    return contract.motion_reading(speed, np.deg2rad(start_yaw - float(state.get("yaw", start_yaw))))


TOF_JUMP_CM = 15


def keep_tof(tof, kept, jumped):
    if not 10 < tof < 1000:
        return kept, jumped
    if kept is None or abs(tof - kept) <= TOF_JUMP_CM or (jumped is not None and abs(tof - jumped) <= TOF_JUMP_CM):
        return tof, None
    return kept, tof


def run_baseline(args) -> None:
    fly_policy(args, "baseline", baseline.Lapper(), contract.policy_input)


def run_policy(args) -> None:
    from src import actor
    if actor.is_ppo(args.model):
        if actor.is_privileged(args.model):
            sys.exit(f"{args.model} reads the sim's true state: a teacher for training, it cannot fly the drone")
        policy = actor.load_policy(args.model)
        if contract.for_model(policy.observation_size) is None:
            sys.exit(f"{args.model} takes {policy.observation_size} inputs; this code builds "
                     f"{contract.POLICY_INPUT_DIM} or {contract.SLOTS_V1_DIM}")
        fly_policy(args, "policy", policy, contract.policy_input, contract.not_yet_through)
        return
    from stable_baselines3 import SAC
    model = SAC.load(args.model, device="cpu")
    size = model.observation_space.shape[0]
    old = len(OLD_LAGS) * contract.OBS_DIM + OLD_ACTIONS * contract.ACTION_DIM
    lags60 = len(LAGS60) * contract.OBS_DIM + OLD_ACTIONS * contract.ACTION_DIM + 1
    inputs = {contract.POLICY_INPUT_DIM: contract.policy_input, SLOTS_V1_DIM: slots_v1_policy_input,
              old: old_policy_input, old + 1: flag_policy_input, lags60: lags60_policy_input}
    if size not in inputs:
        sys.exit(f"{args.model} takes {size} inputs; this code builds {' or '.join(map(str, inputs))}")
    rule = (seen_once if size == old + 1
            else latched_not_yet_through if args.flag == "latched" else contract.not_yet_through)
    fly_policy(args, "policy", lambda x: model.predict(x, deterministic=True)[0], inputs[size], rule)


def seen_once(unseen: bool, frames) -> bool:
    return bool(unseen) and not frames[-1][contract.VISIBLE]


def latched_not_yet_through(unseen: bool, frames) -> bool:
    before, now = frames[-2], frames[-1]
    major, u = contract.FIELDS.index("major"), contract.FIELDS.index("u")
    flew_at_it = (before[contract.VISIBLE] and not now[contract.VISIBLE]
                  and before[major] >= contract.PASS_MAJOR and abs(before[u]) < contract.PASS_U)
    return bool(unseen) and not flew_at_it


def fly_policy(args, kind: str, policy, policy_input, flag_rule=contract.not_yet_through) -> None:
    session = Session(kind, args.host, record=args.record)
    session.event(f"policy: {getattr(args, 'model', None) or kind}")
    watcher = Watcher(session)
    n = contract.HISTORY_STEPS
    frames = collections.deque([contract.NOT_VISIBLE.copy()] * n, maxlen=n)
    heights = collections.deque([Drone().takeoff_height] * n, maxlen=n)
    motion = collections.deque([contract.NO_MOTION.copy()] * n, maxlen=n)
    start_yaw = []
    actions = collections.deque([np.zeros(contract.ACTION_DIM, np.float32)] * n, maxlen=n)
    unseen = [True]
    tof = [None, None]

    def tick() -> tuple:
        watcher.update()
        frames.append(watcher.obs.copy())
        state = session.state()
        tof[:] = keep_tof(state.get("tof", 0), *tof)
        heights.append(Drone().takeoff_height if tof[0] is None else tof[0] / 100)
        if not start_yaw and "yaw" in state:
            start_yaw.append(float(state["yaw"]))
        motion.append(motion_from_state(state, start_yaw[0] if start_yaw else 0.0))
        unseen[0] = flag_rule(unseen[0], frames)
        x = policy_input(frames, heights, motion, actions, unseen[0])
        action = np.clip(policy(x), -1, 1)
        actions.append(action.astype(np.float32))
        watcher.rc = tuple(int(round(a * full)) for a, full in zip(action, RC_FULL))
        return watcher.rc

    try:
        start = countdown_and_takeoff(session, args.min_battery)
        if start:
            fly(session, start + [Hold(kind, args.seconds, tick)], args.fence_cm, args.max_height_cm)
    finally:
        watcher.close()
        session.close()
    finish(session)


def finish(session: Session) -> None:
    text = summarise(session.dir)
    (session.dir / "summary.txt").write_text(text + "\n")
    print("\n" + text)
    print(f"\nSaved to {session.dir.relative_to(ROOT)}")



def set_param(name: str, value: str, note: str) -> None:
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



def ask_cm(question: str) -> float:
    while True:
        try:
            return float(input(question).strip())
        except ValueError:
            print("  a number, please")


def run_tilt(args) -> None:
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
    objp[:, :2] -= objp[:, :2].mean(axis=0)
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
                    x, y, z = tvec.ravel()
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



TAPE = (
    ("thickness", "Noodle diameter (cm)", 0.01),
    ("outer_diameter", "Hoop width as hung, outside edge to outside edge (cm)", 0.01),
    ("outer_height", "Hoop height as hung, outside edge to outside edge (cm)", 0.01),
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
    answer = input("Gap from the top of the hoop to the cage roof (cm): ").strip()
    if answer:
        try:
            gap = float(answer)
        except ValueError:
            print(f"  '{answer}' is not a number, skipped")
            return
        from src.sim import HOOP_HEIGHT
        centre = Cage().height - gap / 100 - Hoop().outer_height / 2
        (OUT / "hoop.json").write_text(json.dumps({"gap_cm": gap, "centre_m": round(centre, 3),
                                                   "date": f"{datetime.now():%Y-%m-%d}"}) + "\n")
        inside = HOOP_HEIGHT[0] <= centre <= HOOP_HEIGHT[1]
        print(f"  hoop centre {centre:.2f} m above the floor: {'inside' if inside else 'OUTSIDE'} "
              f"sim.HOOP_HEIGHT {HOOP_HEIGHT}. Saved to {(OUT / 'hoop.json').relative_to(ROOT)}")



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

    def drone_args(p, flies: bool, fence_cm: float = 60, max_height_cm: float = 170):
        p.add_argument("--host", default=default_host(), help="drone IP (default from settings.json)")
        p.add_argument("--record", action="store_true", help="also save video.mp4")
        if flies:
            p.add_argument("--fence-cm", type=float, default=fence_cm,
                           help=f"land if the pad reads further than this from the drone (default {fence_cm:g})")
            p.add_argument("--max-height-cm", type=float, default=max_height_cm,
                           help=f"land above this tof (default {max_height_cm:g})")
            p.add_argument("--min-battery", type=int, default=30, help="refuse to take off below this")

    p = sub.add_parser("check", help="no flight: link, battery, frame size, rates, live view")
    drone_args(p, flies=False)
    p.set_defaults(fn=run_check)

    p = sub.add_parser("board", help="no drone: show the chessboard to measure")
    p.set_defaults(fn=run_board)

    p = sub.add_parser("calib", help="no flight: point the Tello at the on-screen board")
    p.add_argument("--host", default=default_host())
    p.add_argument("--square-mm", type=float, help="square size on this screen (default: from the screen size)")
    p.add_argument("--frames", type=int, default=18, help="views to capture, at least the 15 targets (default 18)")
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

    p = sub.add_parser("hoop", help="no flight: the hoop detector on the live video")
    drone_args(p, flies=False)
    p.set_defaults(fn=run_hoop)

    p = sub.add_parser("baseline", help="flies: the hand-written baseline on the real detector")
    drone_args(p, flies=True, fence_cm=200, max_height_cm=190)
    p.add_argument("--seconds", type=float, default=60, help="flight time before landing (default 60)")
    p.set_defaults(fn=run_baseline)

    p = sub.add_parser("policy", help="flies: a trained SAC model on the real detector")
    drone_args(p, flies=True, fence_cm=200, max_height_cm=190)
    p.add_argument("--model", required=True, help="a model.zip, e.g. trainedModels/nogamma-progress.zip")
    p.add_argument("--flag", choices=("rearm", "latched"), default="rearm",
                   help="not-yet-through rule the model was trained on: latched for 109-input models "
                        "from before 2026-10-09, e.g. teacher2-level1 (default rearm)")
    p.add_argument("--seconds", type=float, default=60, help="flight time before landing (default 60)")
    p.set_defaults(fn=run_policy)

    p = sub.add_parser("survey", help="flies, hands off: rates, then hoop edge and marker measurements")
    drone_args(p, flies=True, fence_cm=200, max_height_cm=185)
    p.add_argument("--parts", nargs="+", choices=("rates-ud", "rates-yaw", "hoop"),
                   help="only these parts (default all; --behind: hoop only)")
    p.add_argument("--behind", action="store_true",
                   help="the drone starts behind the hoop: the marker should never be reported")
    p.set_defaults(fn=run_survey)

    p = sub.add_parser("markercheck", help="no drone: the marker's white test re-run over a recorded video")
    p.add_argument("folder", help="a run folder with video.mp4, e.g. runs/lab/<survey>")
    p.add_argument("--white-max-s", nargs="+", default=["60", "100", "60,100"],
                   help="marker saturation limits to compare with detect.MARKER_WHITE_MAX_S, each tried "
                        "in turn: e.g. 60 100 60,100 (the default)")
    p.add_argument("--side", choices=("front", "back"), help="default: from survey.csv")
    p.add_argument("--every", type=int, default=3, help="use every Nth frame (default 3)")
    p.set_defaults(fn=run_markercheck)

    p = sub.add_parser("summary", help="re-print the numbers from a run folder")
    p.add_argument("folder")
    p.set_defaults(fn=lambda a: print(summarise(Path(a.folder))))

    args = parser.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main(sys.argv[1:])
