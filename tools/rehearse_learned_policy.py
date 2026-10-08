"""Replay saved preview features through a selected policy and memory commands.

No sockets, camera connection or hardware calls. Explicit --arm and --start
authorize only this offline memory rehearsal. Telemetry/operator heartbeats are
synthetic fixtures, not measurements. The recorded video receipt time and age
are retained. A stop or the 30-second limit ends this one session permanently.
"""
from dataclasses import asdict
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from src.learned_policy_session import CameraSample, LearnedPolicySession
from src.policy_runtime import SPECS


ROOT = Path(__file__).resolve().parents[1]


def load_samples(path, source_profile):
    """Extract the documented newest feature block, never infer lags from size."""
    spec = SPECS[source_profile]
    metadata_path = path.with_name('preview_metadata.json')
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('model_sha256') != spec.sha256 or metadata.get('policy_inputs') != spec.inputs:
        raise ValueError('Preview metadata does not match the explicit source profile')
    records = []
    offset = (len(spec.lags) - 1) * 8
    with path.open(encoding='utf-8') as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            observation = np.asarray(row['observation'], dtype=np.float32)
            if observation.shape != (spec.inputs,) or not np.isfinite(observation).all():
                raise ValueError(f'Invalid source observation on line {line_no}')
            stamp, age = row['receipt_monotonic'], row['frame_age_s']
            if (not LearnedPolicySession._finite_stamp(stamp)
                    or not LearnedPolicySession._finite_stamp(age) or age < 0):
                raise ValueError(f'Invalid recorded timing on line {line_no}')
            observed_at = stamp + age
            sequence = row['frame_id']
            if type(sequence) is not int or sequence < 0:
                raise ValueError(f'Invalid frame ID on line {line_no}')
            if records and (observed_at <= records[-1][0]
                            or stamp <= records[-1][1].received_at
                            or sequence <= records[-1][1].sequence):
                raise ValueError(f'Regressing recorded sequence/timing on line {line_no}')
            records.append((observed_at, CameraSample(sequence, stamp,
                                                      observation[offset:offset + 8].copy())))
    if not records:
        raise ValueError('No recorded preview samples')
    return records


def rehearse(path, source_profile, profile, checkpoint, *, armed=False, started=False):
    if not armed or not started:
        raise ValueError('Explicit arm and start are required for offline command rehearsal')
    records = load_samples(path, source_profile)
    now = [records[0][0]]
    consumed, startup_error = 0, None
    with LearnedPolicySession.from_checkpoint(checkpoint, profile, clock=lambda: now[0]) as session:
        try:
            for observed_at, sample in records:
                # Original recorded receipt age is preserved, including any gap.
                # No sleep: this is a virtual-clock software replay, not a timing
                # benchmark. The manual watchdog tick makes gaps deterministic.
                with session._lock:
                    now[0] = observed_at
                    session.watchdog_once()
                    if session.state == 'stopped':
                        break
                    session.receive_detection(sample)
                    session.receive_telemetry(observed_at)  # explicit synthetic fixture
                    session.operator_heartbeat(observed_at)  # explicit synthetic fixture
                    if session.state == 'disarmed':
                        session.arm()
                        session.start()
                consumed += 1
                session.process_latest()
                if session.state == 'stopped':
                    break
        except ValueError as exc:
            startup_error = str(exc)
            session.stop('recorded input rejected: ' + str(exc))
        if session.state in ('armed', 'active'):
            session.stop('recording complete')
        report = {
            'mode': 'offline memory transport; no hardware or network',
            'physical_flight_enabled': False,
            'explicit_operator_actions': ['arm', 'start'],
            'profile': profile,
            'model_sha256': SPECS[profile].sha256,
            'source_profile': source_profile,
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'source_metadata_sha256': hashlib.sha256(
                path.with_name('preview_metadata.json').read_bytes()).hexdigest(),
            'implementation_sha256': {
                name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                for name in ('src/learned_policy_session.py', 'src/command_rehearsal.py',
                             'src/policy_runtime.py', 'src/camera_history.py',
                             'tools/rehearse_learned_policy.py')
            },
            'source': str(path.resolve()),
            'source_samples': len(records),
            'consumed_samples': consumed,
            'issued_policy_steps': len(session.steps),
            'replay_elapsed_s': now[0] - records[0][0],
            'state': session.state,
            'stop_reason': session.reason,
            'input_error': startup_error,
            'telemetry': 'synthetic fresh heartbeat at each recorded sample; not telemetry evidence',
            'operator': 'synthetic fresh heartbeat at each recorded sample; not operator-interface evidence',
            'video_timing': 'original recorded host receipt plus recorded receipt age; exposure/network age unknown',
            'timing_limit': 'virtual clock does not measure current inference speed or physical delivery',
            'control_cadence': 'event-driven replay capped at 10 Hz; no claim of regular 10 Hz operation',
            'causal_limit': 'hand-carried recorded images do not respond to the newly issued memory commands',
            'acknowledgments': 'memory sink only; no physical acknowledgment or landing confirmation',
            'errors': session.errors,
            'commands': [asdict(event) for event in session.events],
            'policy_steps': [{
                'sequence': step.sequence,
                'received_at': step.received_at,
                'observation': step.observation.tolist(),
                'proposed_action': step.proposed_action.tolist(),
                'issued_action': step.issued_action.tolist(),
            } for step in session.steps],
        }
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preview', type=Path, required=True)
    parser.add_argument('--source-profile', choices=tuple(SPECS), required=True)
    parser.add_argument('--profile', choices=tuple(SPECS), required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--arm', action='store_true')
    parser.add_argument('--start', action='store_true')
    args = parser.parse_args(argv)
    if not args.arm or not args.start:
        parser.error('--arm and --start are both required; only offline memory commands are enabled')
    if args.out.exists():
        parser.error('Use a new output file; prior evidence is preserved')
    checkpoint = args.checkpoint or ROOT / 'trainedModels' / (args.profile + '.zip')
    report = rehearse(args.preview, args.source_profile, args.profile, checkpoint,
                       armed=args.arm, started=args.start)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({key: report[key] for key in
                      ('mode', 'profile', 'source_samples', 'consumed_samples',
                       'issued_policy_steps', 'state', 'stop_reason')}, indent=2))


if __name__ == '__main__':
    main()
