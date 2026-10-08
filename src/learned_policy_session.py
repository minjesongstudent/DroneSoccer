"""Offline learned-policy command session; deliberately no physical flight path.

Camera producers supply increasing sequence numbers and original host receipt
timestamps in the session's monotonic clock domain. Exposure time and upstream
network delay are unmeasured. Recorded playback uses its recorded timeline as
the injected clock, never fresh timestamps on cached images.
Only selected, checksum-checked LearnedPolicy instances produce movement. The
memory transport is a test double and LoopbackLink permits local UDP only.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import threading
import time

import numpy as np

from src.command_rehearsal import LoopbackLink, encode_action
from src.policy_runtime import LearnedPolicy, SPECS


class PhysicalFlightBlocked(RuntimeError):
    """The software interface exists, but physical flight is not released."""


def open_physical_transport(*args, **kwargs):
    """Fail before SDK import, socket creation or hardware connection.

    There is intentionally no approval boolean, environment override or takeoff
    option. An independently reviewed release is still required after policy
    acceptance, calibration and physical command-path verification.
    """
    raise PhysicalFlightBlocked(
        'Physical learned-policy flight remains disabled: policy acceptance, '
        'RC calibration and physical stop/landing verification are incomplete.')


class MemoryTransport:
    """In-process command sink for rehearsals; never opens a network connection."""

    def __init__(self):
        self.commands = []
        self.closed = False

    def send(self, command):
        if self.closed:
            raise OSError('Transport closed')
        if command != 'land':
            words = command.split()
            if len(words) != 5 or words[0] != 'rc':
                raise ValueError('Only RC and land supported')
            if any(abs(int(value)) > 100 for value in words[1:]):
                raise ValueError('RC exceeds SDK range')
        self.commands.append(command)

    def close(self):
        self.closed = True


@dataclass(frozen=True)
class CameraSample:
    sequence: int
    received_at: float
    detection: np.ndarray


@dataclass(frozen=True)
class CommandEvent:
    at: float
    command: str
    sent: bool
    error: str = ''


@dataclass(frozen=True)
class PolicyStep:
    sequence: int
    received_at: float
    observation: np.ndarray
    proposed_action: np.ndarray
    issued_action: np.ndarray


@dataclass(frozen=True)
class ControlTick:
    scheduled_at: float
    started_at: float
    completed_at: float
    issued: bool
    skipped_after: int


class LearnedPolicySession:
    """Explicit arm/start and latching stop around the existing learned runtime.

    Inference and image detection run outside the state lock, so the independent
    watchdog and operator stop can interrupt a stalled model/detector. Transport
    sends must be nonblocking (both permitted transports satisfy that contract).
    `last_issued` means locally handed to the transport, not physically received.
    A failed neutral command never falsely overwrites that history with zeros.
    Each session must own its policy instance; do not share its mutable history
    with another session. `from_checkpoint` always creates a fresh instance.
    """

    FRAME_AGE = .25
    TELEMETRY_AGE = .5
    OPERATOR_AGE = .5
    PREDICTION_AGE = .25
    CONTROL_PERIOD = .1  # Optional runner uses 100 ms tick-start intervals.
    MAX_DURATION = 30.
    WATCHDOG_PERIOD = .02

    def __init__(self, policy, transport=None, *, clock=time.monotonic):
        if not isinstance(policy, LearnedPolicy):
            raise TypeError('A selected LearnedPolicy is required')
        if policy.spec not in SPECS.values():
            raise ValueError('Unrecognized learned-policy profile')
        if transport is None:
            transport = MemoryTransport()
        # Do not allow a duck-typed SDK or subclass to silently gain hardware I/O.
        if type(transport) not in (MemoryTransport, LoopbackLink):
            raise TypeError('Only MemoryTransport or IPv4 LoopbackLink is enabled')
        self.policy, self.transport, self.clock = policy, transport, clock
        self.state, self.reason = 'disarmed', ''
        self.latest = None
        self.telemetry_at = self.operator_at = self.prediction_at = None
        self.deadline = self.last_step_at = self.last_sequence = None
        self.last_issued = np.zeros(4, dtype=np.float64)
        self.events, self.errors, self.steps = [], [], []
        self.control_ticks = []
        self._lock = threading.RLock()
        self._inference_lock = threading.Lock()
        self._closed_event = threading.Event()
        self._watchdog = None
        self._control_thread = None

    @classmethod
    def from_checkpoint(cls, path: Path, profile: str, transport=None, **kwargs):
        if profile not in SPECS:
            raise ValueError('Choose an explicit named learned-policy profile')
        return cls(LearnedPolicy(path, SPECS[profile]), transport, **kwargs)

    @staticmethod
    def _finite_stamp(value):
        return (not isinstance(value, (bool, np.bool_))
                and isinstance(value, (int, float, np.integer, np.floating))
                and bool(np.isfinite(value)))

    def _fresh(self, stamp, age, now):
        return self._finite_stamp(stamp) and 0 <= now - stamp <= age

    def _problem(self, now, *, prediction=False):
        if not self._finite_stamp(now):
            return 'invalid session clock'
        if self.latest is None or not self._fresh(self.latest.received_at, self.FRAME_AGE, now):
            return 'stale video'
        if not self._fresh(self.telemetry_at, self.TELEMETRY_AGE, now):
            return 'stale telemetry'
        if not self._fresh(self.operator_at, self.OPERATOR_AGE, now):
            return 'operator heartbeat lost'
        if prediction and not self._fresh(self.prediction_at, self.PREDICTION_AGE, now):
            return 'inference watchdog'
        if self.deadline is not None and now >= self.deadline:
            return 'session time limit'
        return ''

    def _reject(self, reason):
        if self.state in ('armed', 'active'):
            self.stop(reason)
        raise ValueError(reason)

    def receive_detection(self, sample: CameraSample):
        """Accept a producer-stamped detector result from either live or replay input."""
        with self._lock:
            if self.state in ('stopped', 'closed'):
                return False
            if not isinstance(sample, CameraSample):
                self._reject('CameraSample required')
            try:
                obs = np.asarray(sample.detection, dtype=np.float32)
            except (TypeError, ValueError, OverflowError):
                self._reject('invalid detector features')
            if obs.shape != (8,) or not np.isfinite(obs).all():
                self._reject('eight finite detector features required')
            if type(sample.sequence) is not int or sample.sequence < 0:
                self._reject('invalid camera sequence')
            if not self._fresh(sample.received_at, self.FRAME_AGE, self.clock()):
                self._reject('stale or future video timestamp')
            if self.latest is not None and (
                    sample.sequence <= self.latest.sequence
                    or sample.received_at <= self.latest.received_at):
                self._reject('duplicate or out-of-order video')
            immutable = obs.copy()
            immutable.setflags(write=False)
            self.latest = CameraSample(sample.sequence, float(sample.received_at), immutable)
            return True

    def receive_frame(self, frame_bgr, *, sequence, received_at, detector):
        """Detect a raw BGR image without replacing its producer timestamp.

        `detector` may be the current HoopDetector; its output must expose `obs`.
        Source adapters own decoding and timestamping. This API opens no camera.
        """
        with self._lock:
            if self.state in ('stopped', 'closed'):
                return False
        try:
            frame = np.asarray(frame_bgr)
            if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8 or not frame.size:
                raise ValueError('A nonempty uint8 BGR frame is required')
            result = detector(frame.copy())
            return self.receive_detection(CameraSample(sequence, received_at, result.obs))
        except BaseException:
            self.stop('detector failure')
            raise

    def _receive_stamp(self, field, stamp, limit, reason):
        with self._lock:
            if self.state in ('stopped', 'closed'):
                return False
            previous = getattr(self, field)
            if (not self._fresh(stamp, limit, self.clock())
                    or (previous is not None and stamp <= previous)):
                self._reject(reason)
            setattr(self, field, float(stamp))
            return True

    def receive_telemetry(self, received_at, *, healthy=True):
        if healthy is not True:
            with self._lock:
                self._reject('unhealthy telemetry')
        return self._receive_stamp('telemetry_at', received_at, self.TELEMETRY_AGE,
                                   'stale, future or duplicate telemetry')

    def operator_heartbeat(self, received_at):
        return self._receive_stamp('operator_at', received_at, self.OPERATOR_AGE,
                                   'stale, future or duplicate operator heartbeat')

    def arm(self):
        """Operator action: validate inputs and reset memory; issue no command."""
        with self._lock:
            if self.state != 'disarmed':
                raise RuntimeError('Arm requires a new disarmed session')
            problem = self._problem(self.clock())
            if problem:
                raise ValueError(problem)
            self.policy.reset()
            self.state = 'armed'

    def start(self):
        """Separate operator action: enable local RC processing; never take off."""
        with self._lock:
            if self.state != 'armed':
                raise RuntimeError('Explicit arm is required before start')
            now = self.clock()
            problem = self._problem(now)
            if problem:
                self._reject(problem)
            self.prediction_at, self.deadline = now, now + self.MAX_DURATION
            self.state = 'active'
            self._watchdog = threading.Thread(target=self._watch, daemon=True,
                                               name='learned-policy-watchdog')
            try:
                self._watchdog.start()
            except Exception:
                self._watchdog = None
                self.stop('watchdog failed to start')
                raise

    def _send(self, command):
        try:
            self.transport.send(command)
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            self.errors.append(error)
            self.events.append(CommandEvent(self.clock(), command, False, error))
            return False
        self.events.append(CommandEvent(self.clock(), command, True))
        return True

    def start_control_loop(self):
        """Opt into the 10 Hz software runner after explicit arm() and start().

        Timing is anchored to tick starts, never prediction completions. Late
        work skips elapsed intervals rather than issuing catch-up predictions.
        Once enabled, the runner owns inference; external manual polls do
        nothing. The existing watchdog remains independent. There is no pause
        or automatic restart: stopping the runner stops the whole session.
        """
        with self._lock:
            if self.state != 'active':
                raise RuntimeError('Explicit arm and start are required before the control loop')
            if self._control_thread is not None:
                raise RuntimeError('Control loop already started')
            if self._inference_lock.locked():
                # Query only: acquiring this lock while holding _lock would
                # invert the order used by process_latest and risk deadlock.
                raise RuntimeError('Wait for the in-flight manual prediction before starting the control loop')
            problem = self._problem(self.clock(), prediction=True)
            if problem:
                self._reject(problem)
            first_due = self.clock()
            if self.last_step_at is not None:
                first_due = max(first_due, self.last_step_at + self.CONTROL_PERIOD)
            self._control_thread = threading.Thread(
                target=self._run_control_loop, args=(first_due,), daemon=True,
                name='learned-policy-control')
            try:
                self._control_thread.start()
            except Exception:
                self._control_thread = None
                self.stop('control loop failed to start')
                raise

    @classmethod
    def _next_control_tick(cls, started_at, completed_at):
        """Next future start interval; constant inference cost adds no drift."""
        elapsed = completed_at - started_at
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError('Control clock must advance monotonically')
        periods = max(1, math.ceil(elapsed / cls.CONTROL_PERIOD - 1e-9))
        return started_at + periods * cls.CONTROL_PERIOD, periods - 1

    def _run_control_loop(self, due):
        try:
            while not self._closed_event.is_set():
                delay = due - self.clock()
                if not np.isfinite(delay):
                    raise ValueError('Invalid control clock')
                if delay > 0:
                    if self._closed_event.wait(delay):
                        return
                    continue  # Recheck the clock if the wait ended early.
                started_at = self.clock()
                # An overslept tick begins now. The next interval is measured
                # from this start, preventing short catch-up bursts after a
                # delayed wakeup as well as after a slow prediction.
                step = self.process_latest()
                completed_at = self.clock()
                next_due, skipped = self._next_control_tick(started_at, completed_at)
                with self._lock:
                    self.control_ticks.append(ControlTick(
                        due, started_at, completed_at, step is not None, skipped))
                due = next_due
        except BaseException as exc:
            with self._lock:
                self.errors.append(f'Control loop: {type(exc).__name__}: {exc}')
            self.stop('control loop failure: ' + type(exc).__name__)

    @property
    def control_loop_running(self):
        return self._control_thread is not None and self._control_thread.is_alive()

    def process_latest(self):
        """One control tick, at most 10 Hz; duplicate polls do not advance history.

        Returns a PolicyStep only if its RC was handed to the local transport.
        Concurrent callers cannot duplicate inference or issue delayed actions.
        Manual polling retains the original completion-based cap. The opt-in
        runner instead owns the tick-start cadence; only its own thread may
        infer while enabled, avoiding an extra completion-based gate that would
        otherwise halve the rate when inference latency varies.
        """
        if not self._inference_lock.acquire(blocking=False):
            return None
        try:
            with self._lock:
                if self.state != 'active':
                    return None
                scheduled = self._control_thread is not None
                if scheduled and threading.current_thread() is not self._control_thread:
                    return None
                now = self.clock()
                problem = self._problem(now, prediction=True)
                if problem:
                    self.stop(problem)
                    return None
                if (self.latest.sequence == self.last_sequence or
                        (not scheduled and self.last_step_at is not None and
                         now - self.last_step_at < self.CONTROL_PERIOD - 1e-9)):
                    return None
                sample = self.latest
                previous = self.last_issued.copy()
            try:
                observation, action = self.policy.predict(sample.detection.copy(), previous)
                observation = np.asarray(observation, dtype=np.float32)
                if observation.shape != (self.policy.spec.inputs,) or not np.isfinite(observation).all():
                    raise ValueError('invalid learned observation')
                command, issued = encode_action(action)
            except BaseException as exc:
                self.stop('inference failure: ' + type(exc).__name__)
                if not isinstance(exc, Exception):
                    raise
                self.errors.append(f'{type(exc).__name__}: {exc}')
                return None
            with self._lock:
                if self.state != 'active':
                    return None
                now = self.clock()
                problem = self._problem(now, prediction=True)
                if not self._fresh(sample.received_at, self.FRAME_AGE, now):
                    problem = 'video became stale during inference'
                if problem:
                    self.stop(problem)
                    return None
                try:
                    sent = self._send(command)
                except BaseException:
                    self.stop('transport interrupted')
                    raise
                if not sent:
                    self.stop('RC transport failure')
                    return None
                self.last_issued = issued.copy()
                self.prediction_at = self.last_step_at = now
                self.last_sequence = sample.sequence
                step = PolicyStep(sample.sequence, sample.received_at, observation.copy(),
                                  np.asarray(action).copy(), issued.copy())
                self.steps.append(step)
                return step
        finally:
            self._inference_lock.release()

    def stop(self, reason='operator stop'):
        """Latch first, then independently attempt neutral RC and land exactly once."""
        with self._lock:
            if self.state in ('stopped', 'closed'):
                return
            had_control = self.state in ('armed', 'active')
            self.state, self.reason = 'stopped', reason
            self._closed_event.set()
            if had_control:
                if self._send('rc 0 0 0 0'):
                    self.last_issued = np.zeros(4, dtype=np.float64)
                self._send('land')

    def handle_key(self, key):
        if key in (' ', 'Space', 'Escape', 'l', 'L', 'q', 'Q', 'Ctrl+C'):
            self.stop('operator ' + repr(key))
            return True
        return False

    def watchdog_once(self):
        with self._lock:
            if self.state == 'active':
                problem = self._problem(self.clock(), prediction=True)
                if problem:
                    self.stop(problem)

    def _watch(self):
        while not self._closed_event.wait(self.WATCHDOG_PERIOD):
            try:
                self.watchdog_once()
            except Exception as exc:
                self.stop('watchdog failure: ' + type(exc).__name__)
                return

    def close(self):
        with self._lock:
            if self.state == 'closed':
                return
            self.stop('shutdown')
            self.state = 'closed'
            self._closed_event.set()
        # A blocked model cannot hold shutdown indefinitely. It remains daemon
        # code after this bounded join; state was already latched closed so its
        # eventual return cannot issue any command against the closed transport.
        join_until = time.monotonic() + 1.
        for worker in (self._watchdog, self._control_thread):
            if worker is not None and worker is not threading.current_thread():
                worker.join(timeout=max(0., join_until - time.monotonic()))
        if self.control_loop_running:
            self.errors.append('Control inference still running after bounded shutdown; output remains latched')
        self.transport.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
