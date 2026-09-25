"""Tello EDU session for manual flight.

Flight commands and the camera both go through djitellopy (the Tello library):
connect, streamon, get_frame_read, send_rc_control, takeoff, and land.
Horizontal sticks and yaw go out as rc. Climb and descent use short up/down
moves because this airframe often ignores the up/down channel inside rc.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import socket
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import djitellopy.tello as tello_module
from djitellopy import Tello, TelloException
from djitellopy.tello import BackgroundFrameRead

# The library prints every rc line at INFO. The command window is the log.
Tello.LOGGER.setLevel(logging.WARNING)

# Matches the existing controller: one altitude step per W/S hold cycle.
ALTITUDE_STEP_CM = 30

# rc axes are a percent of full stick, matching the Tello SDK range.
MIN_STICK_SPEED = 15
MAX_STICK_SPEED = 70
DEFAULT_STICK_SPEED = 35

# djitellopy waits this long inside get_frame_read while the UDP socket is already
# open. streamon has to land in that window or the H.264 headers are missed.
LIBRARY_FRAME_WAIT_S = 12


def default_drone_host() -> str:
    """Read the drone IP from the shared settings file when it is present."""
    settings_path = Path(__file__).resolve().parent.parent / "settings.json"
    try:
        payload = json.loads(settings_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return Tello.TELLO_IP
    host = str(payload.get("drone_ip", "")).strip()
    return host or Tello.TELLO_IP


def format_duration(milliseconds: float | None) -> str:
    """Format a measured duration for the latency readouts."""
    if milliseconds is None:
        return "—"
    if milliseconds >= 1000:
        return f"{milliseconds / 1000:.1f} s"
    return f"{milliseconds:.0f} ms"


@dataclass(frozen=True)
class DroneSnapshot:
    """Immutable view of the link, copied onto the UI thread."""

    connected: bool
    connecting: bool
    flying: bool
    busy: bool
    video_ok: bool
    message: str
    battery: str
    height_cm: str
    tof_cm: str
    barometer: str
    pitch: str
    roll: str
    yaw: str
    speed_x: str
    speed_y: str
    speed_z: str
    temperature: str
    flight_time: str
    command_rtt: str
    command_rtt_ms: float | None
    telemetry_age: str
    telemetry_age_ms: float | None
    frame_age: str
    frame_age_ms: float | None
    frame_interval: str
    frame_interval_ms: float | None
    last_command: str
    stick: tuple[int, int, int, int]


class TelloSession:
    """Owns one Tello link, the stick loop, and the command log."""

    def __init__(self) -> None:
        self._io = threading.Lock()
        self._begin = threading.Lock()
        self._stop = threading.Event()
        self._drone: Tello | None = None
        self._reader = None
        self._last_rc_line = ""
        self._last_command = ""
        self._last_rtt_ms: float | None = None
        self._channels = (0, 0, 0, 0)
        self._sticks_live = False
        self._connected = False
        self._connecting = False
        self._busy = False
        self._video_ok = False
        self._video_stop = threading.Event()
        self._video_stop.set()
        self._announced_frame = False
        self._video_note = ""
        self._message = "Idle. Join the Tello Wi-Fi, then press Connect."
        self._frame_age_ms: float | None = None
        self._frame_interval_ms: float | None = None
        self._last_frame_at: float | None = None
        self._seen_frame = None
        self._vertical_pause_until = 0.0
        self._log: queue.Queue[str] = queue.Queue()
        self._flight_thread = threading.Thread(
            target=self._flight_loop,
            name="tello-sticks",
            daemon=True,
        )
        self._flight_thread.start()

    @property
    def message(self) -> str:
        return self._message

    def set_channels(self, left_right: int, forward: int, vertical: int, yaw: int) -> None:
        """Store the stick the flight loop will send. vertical is -1, 0, or 1."""
        self._channels = (
            _clamp(left_right, -100, 100),
            _clamp(forward, -100, 100),
            _clamp(vertical, -1, 1),
            _clamp(yaw, -100, 100),
        )

    def channels(self) -> tuple[int, int, int, int]:
        return self._channels

    def drain_log(self) -> list[str]:
        lines: list[str] = []
        while True:
            try:
                lines.append(self._log.get_nowait())
            except queue.Empty:
                return lines

    def note_video_frame(self, fresh: bool) -> None:
        """Record when a new camera frame was displayed, for the latency readout."""
        now = time.perf_counter()
        if fresh:
            if self._last_frame_at is not None:
                self._frame_interval_ms = (now - self._last_frame_at) * 1000.0
            self._last_frame_at = now
        if self._last_frame_at is None:
            self._frame_age_ms = None
        else:
            self._frame_age_ms = (now - self._last_frame_at) * 1000.0

    def copy_frame(self):
        """Return (is_new, rgb_copy) from djitellopy's BackgroundFrameRead."""
        frame = self._library_frame()
        if frame is None:
            return False, None
        fresh = frame is not self._seen_frame
        self._seen_frame = frame
        if not self._announced_frame:
            self._announced_frame = True
            self._message = "Connected. Camera is on. Press 1 to take off."
            self._emit("# video frame from get_frame_read")
        return fresh, frame.copy()

    def video_placeholder(self) -> str:
        """Text for the video surface until the library delivers a picture."""
        if not self._connected and not self._connecting:
            return "Enter the drone IP, then press Connect."
        return self._video_note or "Waiting for the Tello camera."

    def _library_frame(self):
        """Latest real frame from the library. The reader starts on a black placeholder."""
        reader = self._reader
        if reader is None:
            return None
        try:
            frame = reader.frame
        except Exception:
            return None
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        if frame.shape[0] == 300 and frame.shape[1] == 400 and int(frame.max()) == 0:
            return None
        return frame

    def connect(self, host: str) -> None:
        """Enter SDK mode and open the camera. Safe to call off the UI thread."""
        host = host.strip()
        if not host:
            raise ValueError("Enter the drone IP address.")
        with self._begin:
            if self._connected or self._connecting:
                return
            self._connecting = True
        self._message = f"Connecting to {host}."
        self._emit(f"# connecting to {host}")
        drone = Tello(host)
        self._install_command_log(drone)
        try:
            drone.connect()
            self._start_camera(drone)
            self._drone = drone
            self._connected = True
            self._emit("# link ready")
        except Exception:
            self._stop_camera()
            try:
                drone.end()
            except Exception as end_error:
                self._emit(f"! cleanup after a failed connect: {end_error}")
            self._message = "Connect failed. Check the Tello Wi-Fi and the IP address."
            raise
        finally:
            self._connecting = False

    def takeoff(self) -> None:
        drone = self._require_idle_drone()
        if drone is None:
            return
        self._message = "Taking off."
        try:
            with self._io:
                if self._stop.is_set():
                    return
                drone.takeoff()
            self._sticks_live = True
            self._message = "Airborne. Arrows move, W and S change height, A and D turn."
        except Exception as ex:
            self._emit(f"! takeoff failed: {ex}")
            self._message = "Takeoff failed."
            raise
        finally:
            self._busy = False

    def land(self) -> None:
        with self._begin:
            drone = self._drone
            if drone is None or not self._connected:
                self._message = "No drone is connected."
                return
            if not drone.is_flying or self._busy:
                return
            self._busy = True
            self._sticks_live = False
            self._channels = (0, 0, 0, 0)
        self._message = "Landing."
        try:
            with self._io:
                if self._stop.is_set():
                    return
                # Drop any leftover stick before the land command.
                drone.send_rc_control(0, 0, 0, 0)
                drone.land()
            self._message = "Landed."
        except Exception as ex:
            self._emit(f"! land failed: {ex}")
            self._message = "Land failed."
            raise
        finally:
            self._busy = False

    def emergency(self) -> None:
        """Cut the motors immediately. This is not a controlled landing."""
        drone = self._drone
        if drone is None or not self._connected:
            self._message = "No drone is connected."
            return
        self._sticks_live = False
        self._channels = (0, 0, 0, 0)
        self._message = "Emergency stop sent."
        try:
            with self._io:
                drone.emergency()
        except Exception as ex:
            self._emit(f"! emergency stop failed: {ex}")
            self._message = "Emergency stop failed."
            raise

    def hover(self) -> None:
        """Zero the stick. The flight loop sends rc 0 0 0 0 on the next tick."""
        self._channels = (0, 0, 0, 0)

    def disconnect(self, land_first: bool = True) -> None:
        self._sticks_live = False
        self._channels = (0, 0, 0, 0)
        drone = self._drone
        if drone is None:
            self._stop_camera()
            self._connected = False
            self._reset_frame_timing()
            self._message = "Disconnected."
            return

        if land_first and drone.is_flying and not self._stop.is_set():
            try:
                with self._io:
                    drone.send_rc_control(0, 0, 0, 0)
                    drone.land()
            except Exception as ex:
                self._emit(f"! land during disconnect failed: {ex}")

        self._stop_camera()
        try:
            with self._io:
                drone.end()
        except Exception as ex:
            self._emit(f"! disconnect failed: {ex}")
        self._drone = None
        self._connected = False
        self._reset_frame_timing()
        self._message = "Disconnected."
        self._emit("# disconnected")

    def shutdown(self) -> None:
        """Stop sticks, land if needed, and close the link."""
        self._sticks_live = False
        self._channels = (0, 0, 0, 0)
        self.disconnect(land_first=True)
        self._stop.set()
        self._flight_thread.join(timeout=2.0)

    def snapshot(self) -> DroneSnapshot:
        drone = self._drone
        state: dict = {}
        if drone is not None and self._connected:
            try:
                state = dict(drone.get_current_state())
            except Exception:
                state = {}

        telemetry_age_ms = _telemetry_age_ms(state)
        flying = bool(drone is not None and drone.is_flying)
        return DroneSnapshot(
            connected=self._connected,
            connecting=self._connecting,
            flying=flying,
            busy=self._busy,
            video_ok=self._video_ok,
            message=self._message,
            battery=_field(state, "bat", "%"),
            height_cm=_field(state, "h", " cm"),
            tof_cm=_field(state, "tof", " cm"),
            barometer=_field(state, "baro", " cm"),
            pitch=_field(state, "pitch", "°"),
            roll=_field(state, "roll", "°"),
            yaw=_field(state, "yaw", "°"),
            speed_x=_field(state, "vgx"),
            speed_y=_field(state, "vgy"),
            speed_z=_field(state, "vgz"),
            temperature=_temperature(state),
            flight_time=_field(state, "time", " s"),
            command_rtt=format_duration(self._last_rtt_ms),
            command_rtt_ms=self._last_rtt_ms,
            telemetry_age=format_duration(telemetry_age_ms),
            telemetry_age_ms=telemetry_age_ms,
            frame_age=format_duration(self._frame_age_ms),
            frame_age_ms=self._frame_age_ms,
            frame_interval=format_duration(self._frame_interval_ms),
            frame_interval_ms=self._frame_interval_ms,
            last_command=self._last_command or "—",
            stick=self._channels,
        )

    def _start_camera(self, drone: Tello) -> None:
        """Open the picture with Tello.get_frame_read, then Tello.streamon.

        get_frame_read binds the library video port and waits there for packets.
        streamon has to follow immediately. Extra SDK calls in between let that
        wait expire, and the H.264 headers are gone.
        """
        self._stop_camera()
        stop = threading.Event()
        self._video_stop = stop
        self._seen_frame = None
        self._announced_frame = False
        self._video_note = "Opening the Tello camera."
        Tello.FRAME_GRAB_TIMEOUT = LIBRARY_FRAME_WAIT_S

        reader = self._open_library_reader(drone, stop)
        if reader is None or stop.is_set():
            self._video_ok = False
            if not stop.is_set() and "camera port is busy" not in self._message:
                self._message = "Connected, but the Tello camera did not start. Manual sticks still fly after takeoff."
                self._video_note = self._message
            return

        self._reader = reader
        self._video_ok = True
        self._video_note = "Waiting for the Tello camera."
        self._message = self._video_note
        threading.Thread(
            target=self._watch_library_video,
            args=(drone, stop, reader),
            name="tello-video-watch",
            daemon=True,
        ).start()

    def _open_library_reader(self, drone: Tello, stop: threading.Event):
        """Call get_frame_read on a side thread so this thread can send streamon."""
        self._release_library_reader(drone, self._reader)
        self._reader = None
        # get_frame_read reuses a dead reader unless this is cleared.
        drone.background_frame_read = None
        box: dict = {}

        def open_reader() -> None:
            try:
                box["reader"] = drone.get_frame_read()
            except Exception as ex:
                box["error"] = ex

        port = int(drone.vs_udp_port)
        owner = _udp_listener(port)
        if owner is not None and owner[0] != os.getpid():
            self._message = (
                f"Camera port {port} is used by {owner[1]}. "
                "Close that program, then connect again."
            )
            self._video_note = self._message
            self._emit(f"! UDP {port} is already used by {owner[1]} (pid {owner[0]}). Close it and connect again.")
            return None
        # The previous decoder has to drop UDP 11111 before get_frame_read binds it.
        _wait_until_udp_closed(port, 2.0, stop)
        opener = threading.Thread(target=open_reader, name="tello-get-frame-read", daemon=True)
        opener.start()
        listening = _wait_for_udp(port, 3.0, stop)
        if stop.is_set():
            opener.join(timeout=1.0)
            return None
        if not listening:
            self._emit(f"# Tello video socket UDP {port} is not up yet.")
            self._video_note = f"Tello video port {port} did not open."
        try:
            with self._io:
                if stop.is_set() or self._drone not in (None, drone):
                    opener.join(timeout=1.0)
                    return None
                # Nothing else on the command socket here. streamon must meet the open wait.
                drone.streamon()
        except TelloException as ex:
            self._emit(f"! streamon failed: {ex}")
            self._video_note = f"streamon failed: {ex}"
            opener.join(timeout=1.0)
            return None

        opener.join(timeout=LIBRARY_FRAME_WAIT_S + 2)
        reader = box.get("reader")
        if reader is None:
            error = box.get("error", "timed out")
            if _address_in_use(error):
                self._message = "Connected, but the camera port is busy. Close the other controller and connect again."
            self._video_note = f"get_frame_read failed: {error}"
            self._emit(f"! get_frame_read failed: {error}")
            return None
        self._emit(f"# get_frame_read is listening on UDP {port}")
        return reader

    def _watch_library_video(self, drone: Tello, stop: threading.Event, reader) -> None:
        """Restart get_frame_read once if its decoder thread dies before a picture."""
        if stop.wait(5.0) or self._library_frame() is not None or self._reader is not reader:
            return
        error = getattr(reader, "_stream_error", None)
        alive = False
        worker = getattr(reader, "worker", None)
        if worker is not None:
            alive = worker.is_alive()
        self._emit(f"# no picture yet. decoder alive={alive} error={error}")
        if alive or stop.is_set() or self._drone is not drone:
            self._video_note = "Tello camera is connected, but no picture yet."
            self._message = self._video_note
            return
        self._video_note = "Restarting the Tello camera."
        self._emit("# decoder stopped. Opening get_frame_read again.")
        replacement = self._open_library_reader(drone, stop)
        if stop.is_set():
            return
        if replacement is None:
            self._video_ok = False
            self._message = self._video_note or "Connected, but the Tello camera did not start."
            return
        self._reader = replacement
        self._video_ok = True
        self._video_note = "Waiting for the Tello camera."
        self._message = self._video_note

    def _stop_camera(self) -> None:
        self._video_stop.set()
        reader = self._reader
        self._reader = None
        self._video_ok = False
        drone = self._drone
        self._release_library_reader(drone, reader)

    def _release_library_reader(self, drone: Tello | None, reader) -> None:
        if reader is None:
            return
        container = getattr(reader, "container", None)
        try:
            reader.stop()
        except Exception:
            pass
        if container is not None:
            try:
                container.close()
            except Exception:
                pass
        if drone is not None and drone.background_frame_read is reader:
            drone.background_frame_read = None

    def _install_command_log(self, drone: Tello) -> None:
        """Record every SDK command and the round trip of commands that reply."""
        original_with_return = drone.send_command_with_return
        original_without_return = drone.send_command_without_return
        session = self

        def send_command_with_return(command: str, timeout: int = Tello.RESPONSE_TIMEOUT) -> str:
            started = time.perf_counter()
            session._emit(f"> {command}")
            session._last_command = command
            response = original_with_return(command, timeout=timeout)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            session._last_rtt_ms = elapsed_ms
            session._emit(f"< {response}    {format_duration(elapsed_ms)}")
            return response

        def send_command_without_return(command: str) -> None:
            # rc is repeated many times a second. Keep a line only when it changes.
            if command.startswith("rc "):
                if command != session._last_rc_line:
                    session._last_rc_line = command
                    session._emit(f"> {command}")
            else:
                session._last_command = command
                session._emit(f"> {command}")
            original_without_return(command)

        drone.send_command_with_return = send_command_with_return  # type: ignore[method-assign]
        drone.send_command_without_return = send_command_without_return  # type: ignore[method-assign]

    def _require_idle_drone(self) -> Tello | None:
        with self._begin:
            drone = self._drone
            if drone is None or not self._connected:
                self._message = "Press Connect before takeoff."
                return None
            if self._busy or drone.is_flying:
                return None
            self._busy = True
            self._sticks_live = False
            return drone

    def _flight_loop(self) -> None:
        """Send sticks until shutdown. Altitude steps block, so they skip the short wait."""
        while not self._stop.is_set():
            vertical_step = False
            try:
                vertical_step = self._send_stick()
            except Exception as ex:
                self._emit(f"! control failed: {ex}")
                self._vertical_pause_until = time.perf_counter() + 1.0
            if not vertical_step:
                self._stop.wait(0.05)

    def _send_stick(self) -> bool:
        if self._stop.is_set() or not self._sticks_live:
            return False
        drone = self._drone
        if drone is None or not drone.is_flying:
            return False

        left_right, forward, vertical, yaw = self._channels
        with self._io:
            # Re-check after waiting for the socket so land/emergency is not overwritten.
            if self._stop.is_set() or not self._sticks_live or self._drone is not drone:
                return False
            if vertical != 0 and time.perf_counter() >= self._vertical_pause_until:
                drone.send_rc_control(0, 0, 0, 0)
                if vertical > 0:
                    drone.move_up(ALTITUDE_STEP_CM)
                else:
                    drone.move_down(ALTITUDE_STEP_CM)
                return True
            # The up/down rc channel stays 0. Height changes are the move above.
            drone.send_rc_control(left_right, forward, 0, yaw)
            return False

    def _reset_frame_timing(self) -> None:
        self._last_frame_at = None
        self._frame_age_ms = None
        self._frame_interval_ms = None
        self._seen_frame = None

    def _emit(self, text: str) -> None:
        stamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        self._log.put(f"{stamp}  {text}")


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def _field(state: dict, key: str, suffix: str = "") -> str:
    if key not in state:
        return "—"
    return f"{state[key]}{suffix}"


def _temperature(state: dict) -> str:
    low = state.get("templ")
    high = state.get("temph")
    if low is None and high is None:
        return "—"
    if low is None:
        return f"{high} °C"
    if high is None:
        return f"{low} °C"
    return f"{low}–{high} °C"


def _udp_port_open(port: int) -> bool:
    """True when some socket is already bound to this UDP port."""
    return _udp_listener(port) is not None


def _udp_listener(port: int) -> tuple[int, str] | None:
    """Return (pid, program name) for the socket bound to this UDP port."""
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        output = subprocess.check_output(
            ["netstat", "-ano", "-p", "udp"],
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=flags,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4 or parts[0].upper() != "UDP":
            continue
        if parts[1].rsplit(":", 1)[-1] != str(port):
            continue
        try:
            pid = int(parts[-1])
        except ValueError:
            return None
        return pid, _process_name(pid)
    return None


def _process_name(pid: int) -> str:
    """Best-effort executable name. Falls back to the process id."""
    try:
        import ctypes

        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        access = 0x0400 | 0x0010 | 0x1000
        handle = kernel.OpenProcess(access, False, pid)
        if not handle:
            return f"pid {pid}"
        try:
            buffer = ctypes.create_unicode_buffer(2048)
            length = psapi.GetProcessImageFileNameW(handle, buffer, 2048)
            if length:
                return buffer.value.rsplit("\\", 1)[-1]
        finally:
            kernel.CloseHandle(handle)
    except (OSError, AttributeError):
        return f"pid {pid}"
    return f"pid {pid}"


def _wait_for_udp(port: int, seconds: float, stop: threading.Event) -> bool:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if stop.is_set():
            return False
        if _udp_port_open(port):
            return True
        stop.wait(0.05)
    return _udp_port_open(port)


def _wait_until_udp_closed(port: int, seconds: float, stop: threading.Event) -> None:
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        if stop.is_set() or not _udp_port_open(port):
            return
        stop.wait(0.05)


def _address_in_use(error: object) -> bool:
    text = str(error).lower()
    return "10048" in text or "already in use" in text or "address already" in text


def _install_library_video_hooks() -> None:
    """Keep djitellopy's reader, and stop its decode thread from dying in silence.

    The published reader only catches ExitError. Any other PyAV error ends the
    thread and leaves the black placeholder in place, so the window never paints.
    The UDP flags below are the ones the library comments out; without them
    ffmpeg on Windows drops the Tello stream.
    """
    if getattr(BackgroundFrameRead, "_drone_soccer_hooks", False):
        return

    original_init = BackgroundFrameRead.__init__
    original_update = BackgroundFrameRead.update_frame

    def init(self, tello, address, with_queue=False, maxsize=32):
        if "overrun_nonfatal" not in str(address):
            address = f"{address}?overrun_nonfatal=1&fifo_size=5000"
        self._stream_error = None
        original_init(self, tello, address, with_queue, maxsize)

    def update_frame(self):
        try:
            original_update(self)
        except Exception as ex:
            self._stream_error = ex
            Tello.LOGGER.error("Tello video decode stopped: %s", ex)

    BackgroundFrameRead.__init__ = init  # type: ignore[method-assign]
    BackgroundFrameRead.update_frame = update_frame  # type: ignore[method-assign]
    BackgroundFrameRead._drone_soccer_hooks = True  # type: ignore[attr-defined]


_install_library_video_hooks()


def _install_state_clock() -> None:
    """Record a monotonic timestamp on every state packet.

    The published 2.5.0 listener replaces the state dict and does not keep
    when the datagram arrived, so telemetry age would otherwise stay blank.
    """
    if getattr(Tello.udp_state_receiver, "_stamps_arrival", False):
        return

    def udp_state_receiver() -> None:
        state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            state_socket.bind(("", Tello.STATE_UDP_PORT))
        except OSError as ex:
            Tello.LOGGER.error("State port %s is already in use: %s", Tello.STATE_UDP_PORT, ex)
            return

        while True:
            try:
                payload, address = state_socket.recvfrom(1024)
                host = address[0]
                if host not in tello_module.drones:
                    continue
                state = Tello.parse_state(payload.decode("ASCII"))
                state["received_at"] = time.perf_counter()
                tello_module.drones[host]["state"] = state
            except Exception as ex:
                Tello.LOGGER.error(ex)
                break

    udp_state_receiver._stamps_arrival = True  # type: ignore[attr-defined]
    Tello.udp_state_receiver = udp_state_receiver  # type: ignore[method-assign]


# Patch before the first Tello() starts the state thread.
_install_state_clock()


def _telemetry_age_ms(state: dict) -> float | None:
    """Age of the newest state packet. This is the live link-latency figure."""
    received = state.get("received_at")
    if isinstance(received, datetime):
        return max(0.0, (datetime.now() - received).total_seconds() * 1000.0)
    if isinstance(received, (int, float)):
        return max(0.0, (time.perf_counter() - float(received)) * 1000.0)
    return None
