"""Offline fault injection only; fake outputs never stand in for flight evidence."""
from pathlib import Path
from types import SimpleNamespace
import json
import socket
import threading
import time

import numpy as np
import pytest

from src.command_rehearsal import LoopbackLink, encode_action
from src.learned_policy_session import (
    CameraSample, LearnedPolicySession, MemoryTransport, PhysicalFlightBlocked,
    open_physical_transport,
)
from src.policy_runtime import LearnedPolicy, SPECS


class ModelDouble:
    """Deterministic model-interface double for exact fault and history assertions."""

    def __init__(self):
        self.action = np.array([.513, -.499, .257, -.255])
        self.observations = []

    def predict(self, obs, deterministic):
        assert deterministic
        self.observations.append(obs.copy())
        return self.action.copy(), None


@pytest.fixture
def rig():
    # Bypass only disk loading in the fault-injection fixture. Production factory
    # and real-checkpoint tests below retain checksum/dimension verification.
    policy = LearnedPolicy.__new__(LearnedPolicy)
    policy.spec = SPECS['seenflag-s1-level1']
    policy.model = ModelDouble()
    policy.reset()
    now = [100.]
    session = LearnedPolicySession(policy, clock=lambda: now[0])
    yield session, now
    session.close()


def inputs(session, now, sequence=0):
    session.receive_detection(CameraSample(sequence, now, np.arange(8) / 8))
    session.receive_telemetry(now)
    session.operator_heartbeat(now)


def activate(session, now):
    inputs(session, now)
    session.arm()
    session.start()


def assert_stopped(session):
    assert session.state == 'stopped'
    assert [event.command for event in session.events][-2:] == ['rc 0 0 0 0', 'land']
    before = len(session.events)
    assert session.process_latest() is None
    session.stop()
    assert len(session.events) == before
    with pytest.raises(RuntimeError):
        session.arm()
    with pytest.raises(RuntimeError):
        session.start()


def test_construction_arm_and_start_never_send_or_predict(rig):
    session, now = rig
    assert session.state == 'disarmed'
    assert session.process_latest() is None
    with pytest.raises(RuntimeError):
        session.start()
    with pytest.raises(ValueError, match='video'):
        session.arm()
    inputs(session, now[0])
    session.arm()
    assert session.state == 'armed'
    assert session.process_latest() is None
    assert session.transport.commands == []
    assert session.policy.model.observations == []
    session.start()
    assert session.state == 'active'
    assert session.transport.commands == []
    assert session.policy.model.observations == []
    assert session._watchdog.is_alive()


def test_separate_start_revalidates_freshness(rig):
    session, now = rig
    inputs(session, now[0]); session.arm()
    now[0] += .26
    with pytest.raises(ValueError, match='video'):
        session.start()
    assert_stopped(session)


def test_every_history_uses_successful_quantized_commands(rig):
    session, now = rig
    activate(session, now[0])
    issued = []
    for sequence in range(5):
        if sequence:
            now[0] += .1
            inputs(session, now[0], sequence)
        session.policy.model.action += .001
        step = session.process_latest()
        expected = np.zeros((3, 4))
        history = issued[-3:]
        if history:
            expected[-len(history):] = history
        np.testing.assert_allclose(step.observation[-13:-1].reshape(3, 4), expected)
        command, actual = encode_action(step.proposed_action)
        assert session.transport.commands[-1] == command
        np.testing.assert_array_equal(step.issued_action, actual)
        np.testing.assert_array_equal(session.last_issued, actual)
        issued.append(actual)
    assert not np.array_equal(issued[0], session.steps[0].proposed_action)


def test_duplicate_poll_and_fast_camera_do_not_advance_control_history(rig):
    session, now = rig
    activate(session, now[0])
    first = session.process_latest()
    assert first is not None
    assert session.process_latest() is None
    now[0] += .01; inputs(session, now[0], 1)
    assert session.process_latest() is None
    assert len(session.policy.model.observations) == 1
    now[0] += .09
    assert session.process_latest().sequence == 1
    assert len(session.policy.model.observations) == 2


@pytest.mark.parametrize('key', [' ', 'Space', 'Escape', 'l', 'L', 'q', 'Q', 'Ctrl+C'])
def test_operator_stop_keys_latch(rig, key):
    session, now = rig
    activate(session, now[0]); session.process_latest()
    assert session.handle_key(key)
    assert_stopped(session)
    np.testing.assert_array_equal(session.last_issued, np.zeros(4))


def test_unmapped_key_does_not_arm_start_or_stop(rig):
    session, _ = rig
    assert not session.handle_key('Enter')
    assert session.state == 'disarmed' and not session.events


@pytest.mark.parametrize('bad', [np.zeros(7), [float('nan')] * 8, ['bad'] * 8])
def test_invalid_detection_latches(rig, bad):
    session, now = rig
    activate(session, now[0]); now[0] += .1
    with pytest.raises(ValueError):
        session.receive_detection(CameraSample(1, now[0], bad))
    assert_stopped(session)


@pytest.mark.parametrize('sequence,stamp', [(0, 100.1), (-1, 100.1), (1, 100.),
                                         (True, 100.1), (1, 99.), (1, 101.),
                                         (1, float('nan')), (1, '100.1')])
def test_stale_duplicate_regressing_and_future_frames(rig, sequence, stamp):
    session, now = rig
    activate(session, now[0]); now[0] += .1
    with pytest.raises(ValueError):
        session.receive_detection(CameraSample(sequence, stamp, np.zeros(8)))
    assert_stopped(session)


@pytest.mark.parametrize('source', ['receive_telemetry', 'operator_heartbeat'])
@pytest.mark.parametrize('stamp', [99., 100., 101., float('inf'), True, None])
def test_bad_heartbeat_timestamps_latch(rig, source, stamp):
    session, now = rig
    activate(session, now[0]); now[0] += .1
    with pytest.raises(ValueError):
        getattr(session, source)(stamp)
    assert_stopped(session)


def test_unhealthy_telemetry_latches(rig):
    session, now = rig
    activate(session, now[0])
    with pytest.raises(ValueError, match='unhealthy'):
        session.receive_telemetry(now[0], healthy=False)
    assert_stopped(session)


@pytest.mark.parametrize('missing,reason', [('frame', 'video'), ('telemetry', 'telemetry'),
                                         ('operator', 'heartbeat'), ('inference', 'inference')])
def test_individual_watchdog_faults(rig, missing, reason):
    session, now = rig
    activate(session, now[0])
    session.process_latest()
    # Sustain the unrelated feeds and inference long enough to isolate each fault.
    for seq in range(1, 7):
        if session.state != 'active':
            break
        now[0] += .1
        if missing != 'frame':
            session.receive_detection(CameraSample(seq, now[0], np.zeros(8)))
        if missing != 'telemetry':
            session.receive_telemetry(now[0])
        if missing != 'operator':
            session.operator_heartbeat(now[0])
        if missing != 'inference':
            session.process_latest()
        session.watchdog_once()
    assert_stopped(session)
    assert reason in session.reason


def test_expiry_checked_even_with_fresh_feeds(rig):
    session, now = rig
    activate(session, now[0])
    with session._lock:
        now[0] += session.MAX_DURATION
        inputs(session, now[0], 1)
        session.prediction_at = now[0]
        session.watchdog_once()
    assert_stopped(session)
    assert session.reason == 'session time limit'


@pytest.mark.parametrize('bad', [np.zeros(3), [float('nan')] * 4, [float('inf')] * 4,
                               [1.001, 0, 0, 0]])
def test_invalid_model_action_stops_before_motion(rig, bad):
    session, now = rig
    activate(session, now[0])
    session.policy.model.action = np.asarray(bad)
    assert session.process_latest() is None
    assert_stopped(session)
    assert session.transport.commands == ['rc 0 0 0 0', 'land']


@pytest.mark.parametrize('error', [RuntimeError('model failed'), ValueError('bad history')])
def test_inference_exception_stops(rig, monkeypatch, error):
    session, now = rig
    activate(session, now[0])
    def fail(*args):
        raise error
    monkeypatch.setattr(session.policy, 'predict', fail)
    assert session.process_latest() is None
    assert_stopped(session)


def test_keyboard_interrupt_in_inference_stops_then_propagates(rig, monkeypatch):
    session, now = rig
    activate(session, now[0])
    def fail(*args):
        raise KeyboardInterrupt()
    monkeypatch.setattr(session.policy, 'predict', fail)
    with pytest.raises(KeyboardInterrupt):
        session.process_latest()
    assert_stopped(session)


def test_input_copy_cannot_be_changed_after_receive(rig):
    session, now = rig
    features = np.arange(8, dtype=np.float32)
    session.receive_detection(CameraSample(0, now[0], features))
    features[:] = float('nan')
    assert np.isfinite(session.latest.detection).all()
    assert not session.latest.detection.flags.writeable


def test_detector_frame_api_preserves_timestamp_and_features(rig):
    session, now = rig
    calls = []
    def detector(frame):
        calls.append(frame.shape)
        return SimpleNamespace(obs=np.arange(8))
    assert session.receive_frame(np.zeros((24, 32, 3), np.uint8), sequence=4,
                                 received_at=now[0] - .1, detector=detector)
    assert calls == [(24, 32, 3)]
    assert session.latest.received_at == now[0] - .1
    np.testing.assert_array_equal(session.latest.detection, np.arange(8))


def test_slow_detector_cannot_refresh_stale_capture(rig):
    session, now = rig
    activate(session, now[0])
    def detector(frame):
        now[0] += .3
        return SimpleNamespace(obs=np.zeros(8))
    # Hold the lock here only to isolate the detector-return validation from the
    # separate watchdog. The independent-thread stop is exercised below.
    with session._lock:
        with pytest.raises(ValueError):
            session.receive_frame(np.zeros((24, 32, 3), np.uint8), sequence=1,
                                  received_at=100.01, detector=detector)
    assert_stopped(session)


def test_detector_exception_latches(rig):
    session, now = rig
    activate(session, now[0])
    def detector(frame):
        raise RuntimeError('detector failed')
    with pytest.raises(RuntimeError):
        session.receive_frame(np.zeros((24, 32, 3), np.uint8), sequence=1,
                              received_at=100.01, detector=detector)
    assert_stopped(session)


@pytest.mark.parametrize('stop_kind', ['operator', 'watchdog', 'close'])
def test_stop_interrupts_blocked_inference_and_discards_late_result(rig, monkeypatch, stop_kind):
    session, now = rig
    activate(session, now[0])
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    original = session.policy.predict
    def blocked(*args):
        entered.set()
        assert release.wait(2)
        return original(*args)
    monkeypatch.setattr(session.policy, 'predict', blocked)
    send = session.transport.send
    def track(command):
        send(command)
        if command == 'land':
            stopped.set()
    monkeypatch.setattr(session.transport, 'send', track)
    results = []
    worker = threading.Thread(target=lambda: results.append(session.process_latest()))
    worker.start()
    try:
        assert entered.wait(1)
        assert session.process_latest() is None  # second caller cannot race history
        if stop_kind == 'operator':
            session.handle_key('Escape')
        elif stop_kind == 'close':
            session.close()
        else:
            now[0] += .26  # real watchdog thread, no manual poll
        assert stopped.wait(1)
        assert session.transport.commands == ['rc 0 0 0 0', 'land']
    finally:
        release.set(); worker.join(2)
    assert not worker.is_alive()
    assert results == [None]
    assert len(session.steps) == 0
    assert session.transport.commands == ['rc 0 0 0 0', 'land']


def test_input_fresh_at_entry_but_old_after_inference_cannot_send(rig, monkeypatch):
    session, now = rig
    activate(session, now[0])
    original = session.policy.predict
    def delayed(*args):
        now[0] += .3
        return original(*args)
    monkeypatch.setattr(session.policy, 'predict', delayed)
    assert session.process_latest() is None
    assert_stopped(session)
    assert not session.steps


@pytest.mark.parametrize('fail_command', ['motion', 'neutral', 'land', 'all'])
def test_send_failures_preserve_truth_and_still_attempt_land(rig, monkeypatch, fail_command):
    session, now = rig
    activate(session, now[0]); session.process_latest()
    prior = session.last_issued.copy()
    attempted = []
    send = session.transport.send
    def fault(command):
        attempted.append(command)
        category = ('land' if command == 'land' else 'neutral'
                    if command == 'rc 0 0 0 0' else 'motion')
        if fail_command in ('all', category):
            raise OSError('injected ' + category)
        send(command)
    monkeypatch.setattr(session.transport, 'send', fault)
    if fail_command in ('motion', 'all'):
        now[0] += .1; inputs(session, now[0], 1)
        assert session.process_latest() is None
    else:
        session.stop()
    assert_stopped(session)
    assert attempted[-2:] == ['rc 0 0 0 0', 'land']
    assert any(not event.sent for event in session.events)
    expected = prior if fail_command in ('neutral', 'all') else np.zeros(4)
    np.testing.assert_array_equal(session.last_issued, expected)


def test_close_before_arm_and_idempotent_close_do_not_send(rig):
    session, _ = rig
    session.close(); session.close()
    assert session.state == 'closed'
    assert session.transport.closed
    assert not session.events


def test_context_manager_stops_on_application_exception(rig):
    session, now = rig
    with pytest.raises(RuntimeError):
        with session:
            activate(session, now[0])
            raise RuntimeError('application failed')
    assert session.state == 'closed'
    assert session.transport.commands == ['rc 0 0 0 0', 'land']


def test_physical_transport_is_blocked_before_any_socket(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('A physical interface must not create a socket')
    monkeypatch.setattr(socket, 'socket', forbidden)
    for options in ({}, {'approved': True}, {'host': '192.168.10.1'}):
        with pytest.raises(PhysicalFlightBlocked, match='acceptance'):
            open_physical_transport(**options)


def test_unknown_transport_and_scripted_policy_are_rejected(rig):
    session, _ = rig
    with pytest.raises(TypeError, match='LoopbackLink'):
        LearnedPolicySession(session.policy, SimpleNamespace(send=lambda _: None))
    with pytest.raises(TypeError, match='LearnedPolicy'):
        LearnedPolicySession(SimpleNamespace(predict=lambda *_: np.zeros(4)))


@pytest.mark.parametrize('command', ['takeoff', 'emergency', 'command', 'streamon'])
def test_memory_transport_has_no_launch_or_sdk_configuration(command):
    with pytest.raises(ValueError):
        MemoryTransport().send(command)


def test_loopback_wire_uses_the_same_session(rig):
    template, now = rig
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1', 0)); receiver.settimeout(1)
    try:
        link = LoopbackLink('127.0.0.1', receiver.getsockname()[1])
        with LearnedPolicySession(template.policy, link, clock=lambda: now[0]) as session:
            activate(session, now[0])
            step = session.process_latest()
            assert receiver.recv(1024).decode() == encode_action(step.proposed_action)[0]
            session.handle_key('Escape')
            assert receiver.recv(1024).decode() == 'rc 0 0 0 0'
            assert receiver.recv(1024).decode() == 'land'
    finally:
        receiver.close()


@pytest.mark.parametrize('profile', list(SPECS))
def test_real_unchanged_checkpoint_through_quantized_session(profile):
    root = Path(__file__).resolve().parents[1]
    now = [100.]
    with LearnedPolicySession.from_checkpoint(root / 'trainedModels' / (profile + '.zip'),
                                              profile, clock=lambda: now[0]) as session:
        activate(session, now[0])
        first = session.process_latest()
        assert first.observation.shape == (SPECS[profile].inputs,)
        np.testing.assert_array_equal(first.observation[-13:-1], np.zeros(12))
        now[0] += .1; inputs(session, now[0], 1)
        second = session.process_latest()
        np.testing.assert_allclose(second.observation[-5:-1], first.issued_action)
        assert np.isfinite(second.proposed_action).all()
        assert len(session.steps) == 2


def test_selected_profile_and_model_checksum_required():
    root = Path(__file__).resolve().parents[1]
    with pytest.raises(ValueError, match='named'):
        LearnedPolicySession.from_checkpoint(root / 'missing.zip', 'automatic')
    with pytest.raises(ValueError, match='checksum'):
        LearnedPolicySession.from_checkpoint(root / 'trainedModels/lags60-scratch.zip',
                                              'seenflag-s1-level1')


def preview_fixture(tmp_path, *, gaps=(0, .1, .2), age=.02):
    source = SPECS['lags60-scratch']
    observations = np.arange(source.inputs, dtype=float)
    path = tmp_path / 'policy_preview.jsonl'
    rows = [dict(frame_id=sequence + 1, receipt_monotonic=100. + gap,
                 frame_age_s=age, observation=observations.tolist())
            for sequence, gap in enumerate(gaps)]
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    path.with_name('preview_metadata.json').write_text(json.dumps({
        'model_sha256': source.sha256, 'policy_inputs': source.inputs}), encoding='utf-8')
    return path, rows


def test_replay_extracts_documented_current_features_and_preserves_receipt_age(tmp_path):
    from tools.rehearse_learned_policy import load_samples
    path, rows = preview_fixture(tmp_path)
    records = load_samples(path, 'lags60-scratch')
    for (observed_at, sample), row in zip(records, rows):
        assert observed_at == row['receipt_monotonic'] + row['frame_age_s']
        assert sample.received_at == row['receipt_monotonic']
        np.testing.assert_array_equal(sample.detection, row['observation'][88:96])


def test_replay_requires_explicit_arm_and_start_before_reading_source(tmp_path):
    from tools.rehearse_learned_policy import rehearse
    for armed, started in [(False, False), (True, False), (False, True)]:
        with pytest.raises(ValueError, match='arm and start'):
            rehearse(tmp_path / 'absent', 'lags60-scratch', 'seenflag-s1-level1',
                      tmp_path / 'absent', armed=armed, started=started)


def test_replay_refuses_wrong_source_profile(tmp_path):
    from tools.rehearse_learned_policy import load_samples
    path, _ = preview_fixture(tmp_path)
    with pytest.raises(ValueError, match='metadata'):
        load_samples(path, 'seenflag-s1-level1')


@pytest.mark.parametrize('mutation', ['dimension', 'duplicate', 'future_age'])
def test_replay_rejects_malformed_sources(tmp_path, mutation):
    from tools.rehearse_learned_policy import load_samples
    path, rows = preview_fixture(tmp_path)
    if mutation == 'dimension':
        rows[0]['observation'] = [0] * 85
    elif mutation == 'duplicate':
        rows[1]['frame_id'] = rows[0]['frame_id']
    else:
        rows[0]['frame_age_s'] = -1
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
    with pytest.raises(ValueError):
        load_samples(path, 'lags60-scratch')


@pytest.mark.parametrize('gaps,reason,steps', [((0, .1, .2), 'recording complete', 3),
                                           ((0, .3), 'stale video', 1)])
def test_replay_reports_synthetic_feeds_and_does_not_hide_gaps(
        rig, tmp_path, monkeypatch, gaps, reason, steps):
    from tools.rehearse_learned_policy import rehearse
    template, _ = rig
    path, _ = preview_fixture(tmp_path, gaps=gaps)
    def factory(path, profile, *, clock):
        return LearnedPolicySession(template.policy, clock=clock)
    monkeypatch.setattr(LearnedPolicySession, 'from_checkpoint', factory)
    report = rehearse(path, 'lags60-scratch', 'seenflag-s1-level1', tmp_path / 'fixture',
                       armed=True, started=True)
    assert report['stop_reason'] == reason
    assert report['issued_policy_steps'] == steps
    assert report['physical_flight_enabled'] is False
    assert 'synthetic' in report['telemetry'] and 'synthetic' in report['operator']
    assert [entry['command'] for entry in report['commands']][-2:] == ['rc 0 0 0 0', 'land']
    if steps > 1:
        np.testing.assert_allclose(report['policy_steps'][1]['observation'][-5:-1],
                                   report['policy_steps'][0]['issued_action'])


@pytest.mark.parametrize('elapsed,periods', [(.01, 1), (.08, 1), (.1, 1), (.15, 2),
                                          (.21, 3), (.47, 5)])
def test_scheduler_deadline_is_anchored_to_start_and_skips_elapsed_intervals(elapsed, periods):
    due, skipped = LearnedPolicySession._next_control_tick(100., 100. + elapsed)
    assert due == pytest.approx(100. + periods * .1)
    assert skipped == periods - 1


@pytest.mark.parametrize('latencies,expected_starts,expected_skips', [
    ([.02, .06, .01, .04], [100., 100.1, 100.2, 100.3], [0, 0, 0, 0]),
    ([.15, .01, .15, .01], [100., 100.2, 100.3, 100.5], [1, 0, 1, 0]),
])
def test_runner_variable_latency_does_not_halve_rate_or_burst(
        rig, monkeypatch, latencies, expected_starts, expected_skips):
    session, now = rig
    # Isolate scheduler arithmetic with virtual time. Real independent watchdog
    # behavior under a blocked runner is exercised separately below.
    monkeypatch.setattr(session, '_watch', lambda: None)
    activate(session, now[0])
    original = session.policy.predict
    starts, receipts = [], []
    def delayed(*args):
        starts.append(now[0]); receipts.append(session.latest.received_at)
        result = original(*args)
        now[0] += latencies[len(starts) - 1]
        return result
    monkeypatch.setattr(session.policy, 'predict', delayed)
    class VirtualWait:
        closed = False
        sequence = 0
        def is_set(self):
            return self.closed or len(session.control_ticks) >= len(latencies)
        def wait(self, seconds):
            now[0] += seconds
            self.sequence += 1
            inputs(session, now[0], self.sequence)
            return self.closed
        def set(self):
            self.closed = True
    session._closed_event = VirtualWait()
    session._control_thread = threading.current_thread()
    try:
        session._run_control_loop(now[0])
        assert starts == pytest.approx(expected_starts)
        assert [tick.skipped_after for tick in session.control_ticks] == expected_skips
        assert len(session.steps) == len(latencies)
        assert [step.received_at for step in session.steps] == receipts
        for previous, step in zip(session.steps, session.steps[1:]):
            np.testing.assert_allclose(step.observation[-5:-1], previous.issued_action)
    finally:
        session._control_thread = None


@pytest.fixture
def scheduled_rig(rig):
    session, _ = rig
    session.clock = time.monotonic
    activate(session, session.clock())
    done = threading.Event()
    producer_errors = []
    def produce():
        sequence = 0
        try:
            while not done.wait(.02):
                sequence += 1
                inputs(session, session.clock(), sequence)
        except BaseException as exc:
            producer_errors.append(exc)
    producer = threading.Thread(target=produce, daemon=True)
    producer.start()
    yield session
    done.set(); producer.join(1)
    assert not producer_errors


def test_control_loop_requires_start_and_cannot_be_started_twice(rig, scheduled_rig):
    # scheduled_rig shares this fixture's session, so use a separate unstarted one.
    other = LearnedPolicySession(scheduled_rig.policy)
    try:
        with pytest.raises(RuntimeError, match='arm and start'):
            other.start_control_loop()
    finally:
        other.close()
    assert not scheduled_rig.policy.model.observations  # runner remains opt-in
    scheduled_rig.start_control_loop()
    with pytest.raises(RuntimeError, match='already started'):
        scheduled_rig.start_control_loop()
    assert scheduled_rig.process_latest() is None  # scheduler exclusively owns inference


def test_running_scheduler_never_reprocesses_one_frame(rig):
    session, _ = rig
    session.clock = time.monotonic
    activate(session, session.clock())
    session.start_control_loop()
    assert session._closed_event.wait(1)  # unrefreshed camera forces watchdog stop
    assert len(session.steps) == 1
    assert len(session.policy.model.observations) == 1
    assert_stopped(session)


def test_control_thread_start_failure_stops(rig, monkeypatch):
    session, now = rig
    activate(session, now[0])
    def fail(*args):
        raise RuntimeError('injected thread failure')
    monkeypatch.setattr(threading.Thread, 'start', fail)
    with pytest.raises(RuntimeError, match='thread failure'):
        session.start_control_loop()
    assert_stopped(session)


def test_control_loop_cannot_take_ownership_during_manual_inference(scheduled_rig, monkeypatch):
    session = scheduled_rig
    entered, release = threading.Event(), threading.Event()
    original = session.policy.predict
    def blocked(*args):
        entered.set()
        assert release.wait(1)
        return original(*args)
    monkeypatch.setattr(session.policy, 'predict', blocked)
    manual = threading.Thread(target=session.process_latest)
    manual.start()
    try:
        assert entered.wait(1)
        with pytest.raises(RuntimeError, match='in-flight manual'):
            session.start_control_loop()
        assert session._control_thread is None
        assert session.state == 'active'
    finally:
        release.set(); manual.join(1)
    assert len(session.steps) == 1
    session.start_control_loop()
    assert session.control_loop_running


@pytest.mark.parametrize('stop_kind', ['operator', 'watchdog', 'close'])
def test_blocked_control_runner_stops_independently_and_discards_result(
        scheduled_rig, monkeypatch, stop_kind):
    session = scheduled_rig
    entered, release = threading.Event(), threading.Event()
    original = session.policy.predict
    def blocked(*args):
        entered.set()
        assert release.wait(4)
        return original(*args)
    monkeypatch.setattr(session.policy, 'predict', blocked)
    session.start_control_loop()
    try:
        assert entered.wait(1)
        if stop_kind == 'operator':
            session.handle_key('Escape')
        elif stop_kind == 'close':
            before = time.monotonic()
            session.close()
            assert time.monotonic() - before < 1.4
            assert session.control_loop_running
            assert any('bounded shutdown' in error for error in session.errors)
        else:
            assert session._closed_event.wait(1)
            assert session.reason == 'inference watchdog'
        assert session.transport.commands == ['rc 0 0 0 0', 'land']
    finally:
        release.set()
        session._control_thread.join(1)
    assert not session.control_loop_running
    assert not session.steps
    assert session.transport.commands == ['rc 0 0 0 0', 'land']
