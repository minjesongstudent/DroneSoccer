"""Manual flight window for a Ryze Tello / Tello EDU.

Run from this folder after the computer has joined the Tello Wi-Fi:

    pip install -r requirements.txt
    python main.py

The video window is this process's main window. Status and the command log
open beside it. Latency is the age of the telemetry packet and of the camera
frame, plus the round trip of the last command that expected a reply.
"""

from __future__ import annotations

import sys
import threading
import tkinter as tk
from tkinter import messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk

from tello_session import (
    ALTITUDE_STEP_CM,
    DEFAULT_STICK_SPEED,
    MAX_STICK_SPEED,
    MIN_STICK_SPEED,
    DroneSnapshot,
    TelloSession,
    default_drone_host,
)

# Keyboard matches the existing controller.
KEY_AXES = {
    "Up": ("fb", 1),
    "Down": ("fb", -1),
    "Left": ("lr", -1),
    "Right": ("lr", 1),
    "w": ("ud", 1),
    "s": ("ud", -1),
    "a": ("yaw", -1),
    "d": ("yaw", 1),
}

BG = "#121418"
PANEL = "#1C212B"
VIDEO_BG = "#0E1116"
TEXT = "#F2F4F8"
MUTED = "#9AA3B2"
BUTTON = "#2E3646"
BUTTON_ACTIVE = "#2F6F4E"
DANGER = "#8C2F39"
OK = "#1F6B4A"
FIELD = "#121418"
BORDER = "#4A5568"


def _enable_sharp_text() -> None:
    """Ask Windows to size Tk against the real display scale."""
    try:
        import ctypes

        ctypes.windll.shcore.SetProcessDpiAwareness(1)
    except (AttributeError, OSError):
        return


def _ui_font(size: int) -> ImageFont.ImageFont:
    windir = __import__("os").environ.get("WINDIR", r"C:\Windows")
    path = __import__("os").path.join(windir, "Fonts", "segoeui.ttf")
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def _latency_color(milliseconds: float | None) -> str:
    if milliseconds is None:
        return MUTED
    if milliseconds < 80:
        return "#7DCEA0"
    if milliseconds < 200:
        return "#F0C674"
    return "#FF8B8B"


class FlightApp:
    """Video, sticks, and the two extra windows."""

    def __init__(self) -> None:
        self.session = TelloSession()
        self.root = tk.Tk()
        self.root.title("Tello Manual Flight")
        self.root.configure(bg=BG)
        self.root.minsize(980, 700)

        self._held: set[str] = set()
        self._pad = {"lr": 0, "fb": 0, "ud": 0, "yaw": 0}
        self._space = False
        self._closing = False
        self._photo: ImageTk.PhotoImage | None = None
        self._overlay_font = _ui_font(16)
        self._pad_buttons: dict[tuple[str, int], tk.Button] = {}

        self._build_main()
        self._status_win = self._build_status_window()
        self._command_win = self._build_command_window()
        self._place_windows()

        self.root.bind("<KeyPress>", self._on_key_press)
        self.root.bind("<KeyRelease>", self._on_key_release)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.after(30, self._tick)
        self.root.focus_set()

    def run(self) -> None:
        self.root.mainloop()

    def _build_main(self) -> None:
        header = tk.Frame(self.root, bg=BG)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=16, pady=(14, 8))
        tk.Label(
            header,
            text="Tello EDU",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 20, "bold"),
        ).pack(side="left")
        self._battery = tk.Label(header, text="Battery —", bg=BG, fg=MUTED, font=("Segoe UI", 12))
        self._battery.pack(side="right")

        self._video_frame = tk.Frame(self.root, bg=VIDEO_BG, highlightthickness=0)
        self._video_frame.grid(row=1, column=0, sticky="nsew", padx=(16, 8), pady=(0, 16))
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(1, weight=1)

        self._video = tk.Label(
            self._video_frame,
            text="Enter the drone IP, then press Connect.",
            bg=VIDEO_BG,
            fg=MUTED,
            font=("Segoe UI", 14),
            justify="center",
        )
        self._video.pack(fill="both", expand=True)

        panel = tk.Frame(self.root, bg=PANEL)
        panel.grid(row=1, column=1, sticky="nsew", padx=(8, 16), pady=(0, 16))
        self.root.grid_columnconfigure(1, minsize=320)
        self._build_controls(panel)

    def _build_controls(self, panel: tk.Frame) -> None:
        inner = tk.Frame(panel, bg=PANEL)
        inner.pack(fill="both", expand=True, padx=14, pady=14)

        tk.Label(inner, text="Controller", bg=PANEL, fg=TEXT, font=("Segoe UI", 14, "bold")).pack(anchor="w")

        ip_row = tk.Frame(inner, bg=PANEL)
        ip_row.pack(fill="x", pady=(10, 4))
        tk.Label(ip_row, text="Drone IP", bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(side="left")
        self._host = tk.Entry(
            ip_row,
            bg=FIELD,
            fg=TEXT,
            insertbackground=TEXT,
            relief="flat",
            font=("Segoe UI", 11),
            width=16,
        )
        self._host.insert(0, default_drone_host())
        self._host.pack(side="right", ipady=4)

        actions = tk.Frame(inner, bg=PANEL)
        actions.pack(fill="x", pady=(4, 8))
        self._connect_btn = self._button(actions, "Connect", self._connect, OK)
        self._connect_btn.pack(side="left", padx=(0, 6))
        self._disconnect_btn = self._button(actions, "Disconnect", self._disconnect)
        self._disconnect_btn.pack(side="left")

        tk.Label(inner, text="Latency", bg=PANEL, fg=TEXT, font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(8, 2))
        self._latency_labels = {
            "telemetry": self._metric(inner, "Telemetry age"),
            "frame": self._metric(inner, "Video frame age"),
            "command": self._metric(inner, "Last command time"),
        }

        tk.Label(inner, text="Stick speed", bg=PANEL, fg=TEXT, font=("Segoe UI", 10)).pack(anchor="w", pady=(10, 0))
        self._speed = tk.Scale(
            inner,
            from_=MIN_STICK_SPEED,
            to=MAX_STICK_SPEED,
            orient="horizontal",
            resolution=5,
            bg=PANEL,
            fg=TEXT,
            troughcolor=FIELD,
            highlightthickness=0,
            activebackground=BUTTON_ACTIVE,
            font=("Segoe UI", 9),
            showvalue=True,
        )
        self._speed.set(DEFAULT_STICK_SPEED)
        self._speed.configure(takefocus=0)
        self._speed.bind("<ButtonRelease-1>", lambda _event: self.root.focus_set())
        self._speed.pack(fill="x")

        pad = tk.Frame(inner, bg=PANEL)
        pad.pack(pady=(8, 4))
        self._hold_button(pad, "Up  W", "ud", 1).grid(row=0, column=1, padx=3, pady=3)
        self._hold_button(pad, "Yaw left  A", "yaw", -1).grid(row=1, column=0, padx=3, pady=3)
        self._hold_button(pad, "Forward", "fb", 1).grid(row=1, column=1, padx=3, pady=3)
        self._hold_button(pad, "Yaw right  D", "yaw", 1).grid(row=1, column=2, padx=3, pady=3)
        self._hold_button(pad, "Left", "lr", -1).grid(row=2, column=0, padx=3, pady=3)
        self._hold_button(pad, "Back", "fb", -1).grid(row=2, column=1, padx=3, pady=3)
        self._hold_button(pad, "Right", "lr", 1).grid(row=2, column=2, padx=3, pady=3)
        self._hold_button(pad, "Down  S", "ud", -1).grid(row=3, column=1, padx=3, pady=3)

        flight = tk.Frame(inner, bg=PANEL)
        flight.pack(fill="x", pady=(8, 4))
        self._takeoff_btn = self._button(flight, "Takeoff  1", self._takeoff, OK)
        self._takeoff_btn.pack(side="left", padx=(0, 6))
        self._land_btn = self._button(flight, "Land  2", self._land)
        self._land_btn.pack(side="left", padx=(0, 6))
        self._hover_btn = self._button(flight, "Hover", self._hover)
        self._hover_btn.pack(side="left")

        self._emergency_btn = self._button(inner, "Emergency  motors off", self._emergency, DANGER)
        self._emergency_btn.pack(fill="x", pady=(4, 8))

        tk.Label(
            inner,
            text=(
                f"1 takes off. 2 lands. Arrows fly forward, back, left, and right. "
                f"W climbs and S descends {ALTITUDE_STEP_CM} cm at a time. "
                f"A and D turn. Space holds position. A climb pauses the stick until it finishes."
            ),
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=280,
            justify="left",
        ).pack(anchor="w", pady=(4, 8))

        windows = tk.Frame(inner, bg=PANEL)
        windows.pack(fill="x")
        self._button(windows, "Status", self._show_status).pack(side="left", padx=(0, 6))
        self._button(windows, "Commands", self._show_commands).pack(side="left")

        self._message = tk.Label(
            inner,
            text=self.session.message,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
            wraplength=280,
            justify="left",
        )
        self._message.pack(anchor="w", pady=(10, 0))

    def _build_status_window(self) -> tk.Toplevel:
        window = tk.Toplevel(self.root)
        window.title("Drone status")
        window.configure(bg=PANEL)
        window.minsize(340, 520)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)

        tk.Label(
            window,
            text="Drone status",
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(
            window,
            text="Telemetry from the state packets on UDP 8890.",
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=16, pady=(0, 8))

        body = tk.Frame(window, bg=PANEL)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        fields = [
            ("Link", "link"),
            ("Flight", "flight"),
            ("Battery", "battery"),
            ("Height", "height"),
            ("Downward ToF", "tof"),
            ("Barometer", "baro"),
            ("Pitch", "pitch"),
            ("Roll", "roll"),
            ("Yaw", "yaw"),
            ("Speed X", "vgx"),
            ("Speed Y", "vgy"),
            ("Speed Z", "vgz"),
            ("Temperature", "temp"),
            ("Flight time", "time"),
            ("Telemetry age", "telemetry"),
            ("Video frame age", "frame_age"),
            ("Frame interval", "frame_interval"),
            ("Last command time", "rtt"),
            ("Last command", "command"),
            ("Stick sent", "stick"),
        ]
        self._status_values: dict[str, tk.Label] = {}
        for row, (caption, key) in enumerate(fields):
            tk.Label(body, text=caption, bg=PANEL, fg=MUTED, font=("Segoe UI", 10)).grid(
                row=row, column=0, sticky="w", pady=2
            )
            value = tk.Label(body, text="—", bg=PANEL, fg=TEXT, font=("Segoe UI", 10, "bold"))
            value.grid(row=row, column=1, sticky="e", padx=(16, 0), pady=2)
            self._status_values[key] = value
            body.grid_columnconfigure(1, weight=1)
        return window

    def _build_command_window(self) -> tk.Toplevel:
        window = tk.Toplevel(self.root)
        window.title("Drone commands")
        window.configure(bg="#10140F")
        window.minsize(420, 320)
        window.protocol("WM_DELETE_WINDOW", window.withdraw)

        tk.Label(
            window,
            text="Drone commands",
            bg="#10140F",
            fg="#8FBF9A",
            font=("Segoe UI", 16, "bold"),
        ).pack(anchor="w", padx=16, pady=(14, 2))
        tk.Label(
            window,
            text="Each line is a command sent to the aircraft, then the reply and the round trip. Stick lines appear when the stick values change.",
            bg="#10140F",
            fg="#C5D0C2",
            font=("Segoe UI", 9),
            wraplength=460,
            justify="left",
        ).pack(anchor="w", padx=16, pady=(0, 8))

        clear = self._button(window, "Clear", self._clear_commands)
        clear.pack(anchor="w", padx=16, pady=(0, 8))

        frame = tk.Frame(window, bg="#10140F")
        frame.pack(fill="both", expand=True, padx=16, pady=(0, 16))
        scroll = tk.Scrollbar(frame)
        scroll.pack(side="right", fill="y")
        self._command_text = tk.Text(
            frame,
            bg="#0C100E",
            fg="#D7F5D0",
            insertbackground="#D7F5D0",
            relief="flat",
            font=("Consolas", 11),
            wrap="word",
            yscrollcommand=scroll.set,
            padx=8,
            pady=8,
        )
        self._command_text.pack(side="left", fill="both", expand=True)
        scroll.configure(command=self._command_text.yview)
        self._command_text.configure(state="disabled")
        return window

    def _place_windows(self) -> None:
        self.root.update_idletasks()
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        main_w = min(1100, max(980, screen_w - 420))
        main_h = min(780, screen_h - 80)
        self.root.geometry(f"{main_w}x{main_h}+16+16")
        status_x = min(16 + main_w + 12, max(16, screen_w - 380))
        self._status_win.geometry(f"360x560+{status_x}+16")
        command_y = min(16 + 560 + 12, max(16, screen_h - 360))
        self._command_win.geometry(f"460x320+{status_x}+{command_y}")

    def _button(self, parent: tk.Widget, text: str, command, color: str = BUTTON) -> tk.Button:
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=color,
            fg=TEXT,
            activebackground="#3C4658",
            activeforeground=TEXT,
            relief="flat",
            bd=0,
            padx=10,
            pady=6,
            font=("Segoe UI", 10),
            cursor="hand2",
            takefocus=0,
        )

    def _hold_button(self, parent: tk.Widget, text: str, axis: str, direction: int) -> tk.Button:
        button = self._button(parent, text, None)
        button.configure(width=12)
        button.bind("<ButtonPress-1>", lambda _event: self._set_pad(axis, direction))
        button.bind("<ButtonRelease-1>", lambda _event: self._set_pad(axis, 0))
        button.bind("<Leave>", lambda _event: self._set_pad(axis, 0) if self._pad[axis] == direction else None)
        self._pad_buttons[(axis, direction)] = button
        return button

    def _metric(self, parent: tk.Widget, caption: str) -> tk.Label:
        row = tk.Frame(parent, bg=PANEL)
        row.pack(fill="x", pady=1)
        tk.Label(row, text=caption, bg=PANEL, fg=MUTED, font=("Segoe UI", 9)).pack(side="left")
        value = tk.Label(row, text="—", bg=PANEL, fg=TEXT, font=("Segoe UI", 11, "bold"))
        value.pack(side="right")
        return value

    def _set_pad(self, axis: str, direction: int) -> None:
        self._pad[axis] = direction
        self._space = False
        self.root.focus_set()
        self._push_channels()

    def _hover(self) -> None:
        self._space = True
        for axis in self._pad:
            self._pad[axis] = 0
        self._push_channels()
        self.root.focus_set()

    def _push_channels(self) -> None:
        if self._space:
            self.session.hover()
            return
        speed = int(self._speed.get())
        left_right = _combine(self._axis_from_keys("lr"), self._pad["lr"]) * speed
        forward = _combine(self._axis_from_keys("fb"), self._pad["fb"]) * speed
        vertical = _combine(self._axis_from_keys("ud"), self._pad["ud"])
        yaw = _combine(self._axis_from_keys("yaw"), self._pad["yaw"]) * speed
        self.session.set_channels(left_right, forward, vertical, yaw)

    def _axis_from_keys(self, axis: str) -> int:
        total = 0
        for key, (mapped_axis, direction) in KEY_AXES.items():
            if mapped_axis == axis and key in self._held:
                total += direction
        return max(-1, min(1, total))

    def _on_key_press(self, event: tk.Event) -> None:
        if self._typing_in_entry():
            return
        key = event.keysym
        lowered = key.lower()
        if key == "1":
            if key not in self._held:
                self._held.add(key)
                self._takeoff()
            return
        if key == "2":
            if key not in self._held:
                self._held.add(key)
                self._land()
            return
        if key == "space":
            self._held.add(key)
            self._hover()
            return
        mapped = KEY_AXES.get(lowered) or KEY_AXES.get(key)
        if mapped is None:
            return
        self._held.add(lowered if lowered in KEY_AXES else key)
        self._space = False
        self._push_channels()

    def _on_key_release(self, event: tk.Event) -> None:
        key = event.keysym
        lowered = key.lower()
        self._held.discard(key)
        self._held.discard(lowered)
        if key == "space":
            self._space = False
        if self._typing_in_entry():
            return
        self._push_channels()

    def _typing_in_entry(self) -> bool:
        focused = self.root.focus_get()
        return isinstance(focused, tk.Entry)

    def _connect(self) -> None:
        host = self._host.get().strip()
        self._run_drone_action("connect", lambda: self.session.connect(host))

    def _disconnect(self) -> None:
        self._run_drone_action("disconnect", lambda: self.session.disconnect(land_first=True))

    def _takeoff(self) -> None:
        self._run_drone_action("takeoff", self.session.takeoff)

    def _land(self) -> None:
        self._run_drone_action("land", self.session.land)

    def _emergency(self) -> None:
        if not messagebox.askyesno(
            "Emergency stop",
            "Cut the motors immediately? The aircraft will drop.",
            parent=self.root,
        ):
            return
        self._run_drone_action("emergency", self.session.emergency)

    def _run_drone_action(self, label: str, action) -> None:
        """Blocking SDK calls stay off the UI thread."""

        def work() -> None:
            try:
                action()
            except Exception as ex:
                self.root.after(0, lambda: self._report_action_error(label, ex))

        threading.Thread(target=work, name=f"tello-{label}", daemon=True).start()

    def _report_action_error(self, label: str, error: Exception) -> None:
        if self._closing:
            return
        messagebox.showerror(label, str(error), parent=self.root)

    def _show_status(self) -> None:
        self._status_win.deiconify()
        self._status_win.lift()

    def _show_commands(self) -> None:
        self._command_win.deiconify()
        self._command_win.lift()

    def _clear_commands(self) -> None:
        self._command_text.configure(state="normal")
        self._command_text.delete("1.0", "end")
        self._command_text.configure(state="disabled")

    def _tick(self) -> None:
        if self._closing:
            return
        try:
            self._push_channels()
            snapshot = self.session.snapshot()
            self._paint_video(snapshot)
            self._paint_latency(snapshot)
            self._paint_status(snapshot)
            self._paint_commands()
            self._paint_buttons(snapshot)
            self._message.configure(text=self._ground_hint(snapshot))
        except Exception as ex:
            self._message.configure(text=f"Display error: {ex}")
        self.root.after(33, self._tick)

    def _ground_hint(self, snapshot: DroneSnapshot) -> str:
        stick = snapshot.stick
        moving = any(stick)
        if snapshot.connected and not snapshot.flying and not snapshot.busy and moving:
            return "Sticks move the drone only after takeoff. Press 1 to take off."
        return snapshot.message

    def _paint_latency(self, snapshot: DroneSnapshot) -> None:
        telemetry = self._latency_labels["telemetry"]
        frame = self._latency_labels["frame"]
        command = self._latency_labels["command"]
        telemetry.configure(text=snapshot.telemetry_age, fg=_latency_color(snapshot.telemetry_age_ms))
        frame.configure(text=snapshot.frame_age, fg=_latency_color(snapshot.frame_age_ms))
        # Takeoff and land include the aircraft action, so a multi-second value is not link latency.
        command_color = MUTED
        if snapshot.command_rtt_ms is not None and snapshot.command_rtt_ms < 1000:
            command_color = _latency_color(snapshot.command_rtt_ms)
        command.configure(text=snapshot.command_rtt, fg=command_color)
        self._battery.configure(text=f"Battery {snapshot.battery}" if snapshot.connected else "Battery —")

    def _paint_status(self, snapshot: DroneSnapshot) -> None:
        if snapshot.connecting:
            link = "Connecting"
        elif snapshot.connected:
            link = "Connected"
        else:
            link = "Disconnected"
        flight = "Airborne" if snapshot.flying else "On the ground"
        if snapshot.busy:
            flight += "  (busy)"
        left_right, forward, _vertical, yaw = snapshot.stick
        # Vertical is a step, not an rc percent. Show the command the loop will send.
        if snapshot.stick[2] > 0:
            stick = f"up {ALTITUDE_STEP_CM}"
        elif snapshot.stick[2] < 0:
            stick = f"down {ALTITUDE_STEP_CM}"
        else:
            stick = f"rc {left_right} {forward} 0 {yaw}"
        values = {
            "link": link,
            "flight": flight,
            "battery": snapshot.battery,
            "height": snapshot.height_cm,
            "tof": snapshot.tof_cm,
            "baro": snapshot.barometer,
            "pitch": snapshot.pitch,
            "roll": snapshot.roll,
            "yaw": snapshot.yaw,
            "vgx": snapshot.speed_x,
            "vgy": snapshot.speed_y,
            "vgz": snapshot.speed_z,
            "temp": snapshot.temperature,
            "time": snapshot.flight_time,
            "telemetry": snapshot.telemetry_age,
            "frame_age": snapshot.frame_age,
            "frame_interval": snapshot.frame_interval,
            "rtt": snapshot.command_rtt,
            "command": snapshot.last_command,
            "stick": stick if snapshot.flying else f"{stick}  (held)",
        }
        for key, text in values.items():
            color = TEXT
            if key == "telemetry":
                color = _latency_color(snapshot.telemetry_age_ms)
            elif key == "frame_age":
                color = _latency_color(snapshot.frame_age_ms)
            self._status_values[key].configure(text=text, fg=color)

    def _paint_commands(self) -> None:
        lines = self.session.drain_log()
        if not lines:
            return
        self._command_text.configure(state="normal")
        for line in lines:
            self._command_text.insert("end", line + "\n")
        # Keep the window from growing without limit.
        line_count = int(self._command_text.index("end-1c").split(".")[0])
        if line_count > 800:
            self._command_text.delete("1.0", "200.0")
        self._command_text.see("end")
        self._command_text.configure(state="disabled")

    def _paint_buttons(self, snapshot: DroneSnapshot) -> None:
        connected = snapshot.connected and not snapshot.connecting
        self._connect_btn.configure(state="disabled" if connected or snapshot.connecting or snapshot.busy else "normal")
        self._disconnect_btn.configure(state="normal" if connected and not snapshot.busy else "disabled")
        self._takeoff_btn.configure(
            state="normal" if connected and not snapshot.flying and not snapshot.busy else "disabled"
        )
        self._land_btn.configure(state="normal" if connected and snapshot.flying and not snapshot.busy else "disabled")
        self._emergency_btn.configure(state="normal" if connected else "disabled")

        axes = {
            "lr": _combine(self._axis_from_keys("lr"), self._pad["lr"]),
            "fb": _combine(self._axis_from_keys("fb"), self._pad["fb"]),
            "ud": _combine(self._axis_from_keys("ud"), self._pad["ud"]),
            "yaw": _combine(self._axis_from_keys("yaw"), self._pad["yaw"]),
        }
        for (axis, direction), button in self._pad_buttons.items():
            active = not self._space and axes[axis] == direction and direction != 0
            button.configure(bg=BUTTON_ACTIVE if active else BUTTON)
        self._hover_btn.configure(bg=BUTTON_ACTIVE if self._space else BUTTON)

    def _paint_video(self, snapshot: DroneSnapshot) -> None:
        fresh, frame = self.session.copy_frame()
        if frame is None:
            self.session.note_video_frame(False)
            self._photo = None
            self._video.configure(image="", text=self.session.video_placeholder())
            return

        self.session.note_video_frame(fresh)

        image = Image.fromarray(frame)
        width = max(self._video_frame.winfo_width(), 640)
        height = max(self._video_frame.winfo_height(), 360)
        image.thumbnail((width, height), Image.Resampling.BILINEAR)
        self._draw_overlay(image, snapshot)
        self._photo = ImageTk.PhotoImage(image=image, master=self.root)
        self._video.configure(image=self._photo, text="")
        self._video.image = self._photo

    def _draw_overlay(self, image: Image.Image, snapshot: DroneSnapshot) -> None:
        draw = ImageDraw.Draw(image)
        lines = [
            (f"Battery {snapshot.battery}    Height {snapshot.height_cm}", TEXT),
            (f"Telemetry {snapshot.telemetry_age}", _latency_color(snapshot.telemetry_age_ms)),
            (f"Video frame {snapshot.frame_age}", _latency_color(snapshot.frame_age_ms)),
            (f"Command {snapshot.last_command}  {snapshot.command_rtt}", MUTED),
        ]
        box_height = 18 * len(lines) + 16
        draw.rectangle((12, image.height - box_height - 12, 360, image.height - 12), fill="#10141C")
        y = image.height - box_height - 4
        for text, color in lines:
            draw.text((22, y), text, font=self._overlay_font, fill=color)
            y += 18

    def _on_close(self) -> None:
        if self._closing:
            return
        snapshot = self.session.snapshot()
        if snapshot.flying or snapshot.busy:
            if not messagebox.askyesno(
                "Land",
                "The drone is still in a flight command. Land and close?",
                parent=self.root,
            ):
                return
        self._closing = True
        self._message.configure(text="Closing the link.")

        def work() -> None:
            try:
                self.session.shutdown()
            finally:
                self.root.after(0, self.root.destroy)

        threading.Thread(target=work, name="tello-shutdown", daemon=True).start()


def _combine(key_direction: int, pad_direction: int) -> int:
    return max(-1, min(1, key_direction + pad_direction))


def main() -> None:
    _enable_sharp_text()
    try:
        FlightApp().run()
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
