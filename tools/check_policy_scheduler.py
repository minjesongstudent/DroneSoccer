"""Short real-clock, real-checkpoint scheduler check using only synthetic inputs.

The transport is an in-memory sink. No camera, network or hardware is opened.
This measures software scheduling on the current host, not Mac/flight readiness.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path
import threading
import time

import numpy as np

from src.learned_policy_session import CameraSample, LearnedPolicySession
from src.policy_runtime import SPECS


ROOT = Path(__file__).resolve().parents[1]


def check(profile, checkpoint, *, seconds=2., armed=False, started=False):
    if not armed or not started:
        raise ValueError('Explicit arm and start are required for this offline check')
    if not np.isfinite(seconds) or not .5 <= seconds <= 5:
        raise ValueError('Choose a bounded duration from 0.5 to 5 seconds')
    session = LearnedPolicySession.from_checkpoint(checkpoint, profile)
    producer_done = threading.Event()
    producer_errors = []
    def feed(sequence):
        receipt = time.monotonic()
        session.receive_detection(CameraSample(sequence, receipt, np.zeros(8, np.float32)))
        session.receive_telemetry(receipt)
        session.operator_heartbeat(receipt)
    def produce():
        sequence = 0
        try:
            while not producer_done.wait(.02):
                sequence += 1
                feed(sequence)
        except BaseException as exc:
            producer_errors.append(f'{type(exc).__name__}: {exc}')
            session.stop('synthetic producer failure')
    producer = threading.Thread(target=produce, daemon=True, name='synthetic-check-inputs')
    with session:
        feed(0)
        session.arm()
        session.start()
        started_at = time.monotonic()
        try:
            producer.start()
            session.start_control_loop()
            stopped_early = session._closed_event.wait(seconds)
        finally:
            producer_done.set()
            session.stop('synthetic scheduler check complete')
            if producer.ident is not None:
                producer.join(1)
    intervals = np.diff([tick.started_at for tick in session.control_ticks])
    latencies = [tick.completed_at - tick.started_at for tick in session.control_ticks]
    stats = lambda values: (None if len(values) == 0 else {
        'minimum_s': float(np.min(values)), 'median_s': float(np.median(values)),
        'maximum_s': float(np.max(values)),
    })
    return {
        'mode': 'real-clock software timing; synthetic features/telemetry/operator; memory commands only',
        'physical_flight_enabled': False,
        'claim_limit': 'current-host software measurement only; not a Mac, detector or flight acceptance test',
        'profile': profile,
        'model_sha256': SPECS[profile].sha256,
        'requested_duration_s': seconds,
        'elapsed_s': time.monotonic() - started_at,
        'stopped_early': stopped_early,
        'stop_reason': session.reason,
        'issued_policy_steps': len(session.steps),
        'control_ticks': [asdict(tick) for tick in session.control_ticks],
        'tick_start_intervals': stats(intervals),
        'tick_work_durations': stats(latencies),
        'skipped_intervals_after_work': sum(tick.skipped_after for tick in session.control_ticks),
        'all_workers_exited': not producer.is_alive() and not session.control_loop_running,
        'errors': session.errors + producer_errors,
        'implementation_sha256': {
            name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            for name in ('src/learned_policy_session.py', 'src/command_rehearsal.py',
                         'src/policy_runtime.py', 'src/camera_history.py',
                         'tools/check_policy_scheduler.py')
        },
        'commands': [asdict(event) for event in session.events],
        'policy_steps': [{
            'sequence': step.sequence, 'received_at': step.received_at,
            'observation': step.observation.tolist(),
            'proposed_action': step.proposed_action.tolist(),
            'issued_action': step.issued_action.tolist(),
        } for step in session.steps],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=tuple(SPECS), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--seconds', type=float, default=2.)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args(argv)
    if not args.arm or not args.start:
        parser.error('--arm and --start are required; this is an offline memory-only check')
    if args.out.exists():
        parser.error('Choose a new output file to preserve prior evidence')
    report = check(args.profile,
                   args.checkpoint or ROOT / 'trainedModels' / (args.profile + '.zip'),
                   seconds=args.seconds, armed=args.arm, started=args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in (
        'mode', 'profile', 'issued_policy_steps', 'tick_start_intervals',
        'tick_work_durations', 'skipped_intervals_after_work', 'stopped_early',
        'stop_reason', 'all_workers_exited', 'errors')}, indent=2))


if __name__ == '__main__':
    main()
