"""Build the Mac camera recorder from branch source; excludes all user sessions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.camera_calibration import default_calibration, validate_calibration_bytes

SOURCE = ROOT / 'experiments/faisal/lab-tools/Mac-Lab-Recorder'
OUT = ROOT.parent / 'Mac-Lab-Recorder-Live'


def build_package(profile: str, out: Path, calibration_path: Path | None = None) -> Path:
    out = out.expanduser().resolve()
    archive = out.with_suffix('.zip')
    if out.exists() or archive.exists():
        raise FileExistsError(f'Package output already exists; choose a new --out directory: {out}')
    if profile not in ('lags60-scratch', 'seenflag-s1-level1'):
        raise ValueError(f'Unknown policy profile: {profile}')
    input_count = 109 if profile == 'lags60-scratch' else 85
    files = {}
    calibration_status = default_calibration().status()
    calibration_status['bundled_file'] = None
    if calibration_path is not None:
        calibration_path = calibration_path.expanduser().resolve()
        # Validate the same bytes we package, not a second read of a changing file.
        payload = calibration_path.read_bytes()
        calibration_status = validate_calibration_bytes(payload, calibration_path.name).status()
        calibration_status['bundled_file'] = 'calib.npz'
        files['calib.npz'] = payload
    for name in ['recorder.py', 'test_recorder.py', 'live_preview.py', 'test_live_preview.py',
                 'test_calibration.py', 'CALIBRATION.md', 'export_latest.py', 'replay_latest.py',
                 'mac-common.sh', 'MEASUREMENTS.md', '3-Export-Latest.command',
                 'preview-requirements.txt']:
        files[name] = (SOURCE / name).read_bytes()
    for name in ('compare_session.py', 'test_compare_session.py', 'COMPARISON-GUIDE.txt'):
        files[name] = (SOURCE / name).read_bytes()
    files['1-Setup.command'] = (SOURCE / '4-Setup-Camera-Window.command').read_text().replace(
        '5-Record-With-Camera.command', '2-Record.command').replace(
        'test_recorder.py test_live_preview.py',
        'test_recorder.py test_live_preview.py test_calibration.py test_compare_session.py').encode()
    files['2-Record.command'] = (SOURCE / '5-Record-With-Camera.command').read_text().replace(
        '4-Setup-Camera-Window.command', '1-Setup.command').encode()
    files['4-Replay-Latest.command'] = (SOURCE / '6-Replay-Latest-Camera.command').read_text().replace(
        '4-Setup-Camera-Window.command', '1-Setup.command').encode()
    for name in ['__init__.py', 'contract.py', 'params.py', 'detect.py', 'camera_history.py',
                 'camera_calibration.py', 'policy_runtime.py', 'command_rehearsal.py',
                 'learned_policy_session.py']:
        files['src/' + name] = (ROOT / 'src' / name).read_bytes()
    files['tools/__init__.py'] = b''
    files['tools/rehearse_learned_policy.py'] = (ROOT/'tools/rehearse_learned_policy.py').read_bytes()
    files['tools/check_policy_scheduler.py'] = (ROOT/'tools/check_policy_scheduler.py').read_bytes()
    files['5-Check-Control-Timing.command'] = (
        '#!/bin/bash\nsource "$(dirname "$0")/mac-common.sh"\n'
        "echo 'Offline policy timing check. No camera connection or drone commands.'\n"
        'if [ ! -x .preview-venv/bin/python ]; then\n'
        "  echo 'Run 1-Setup.command first.'\n"
        'else\n'
        '  mkdir -p checks || exit 1\n'
        '  CHECK_FILE="checks/timing-$(date +%Y%m%d-%H%M%S)-$$.json"\n'
        f'  .preview-venv/bin/python -m tools.check_policy_scheduler --profile {profile} '
        '--checkpoint preview-policy.zip --seconds 2 --arm --start --out "$CHECK_FILE"\n'
        'fi\npause_window\n'
    ).encode()
    files['OFFLINE-COMMAND-CHECK.txt'] = (
        'Offline learned-policy command check\n\n'
        'Run 1-Setup.command once. Double-click 5-Check-Control-Timing.command\n'
        'for a two-second software timing check using the learned model and\n'
        'explicit synthetic input fixtures. It saves a report in checks/.\n'
        'This measures this computer only, not camera or drone response.\n\n'
        'For recorded-feature replay, open Terminal in this package folder.\n'
        'Keep your policy_preview.jsonl and its preview_metadata.json in the same folder.\n'
        'This check reads saved features, uses a memory-only command sink and ends\n'
        'after at most 30 seconds of recorded time. It cannot connect to a drone.\n\n'
        '.preview-venv/bin/python -m tools.rehearse_learned_policy \\\n'
        '  --preview /path/to/policy_preview.jsonl \\\n'
        '  --source-profile SOURCE_PROFILE \\\n'
        f'  --profile {profile} --checkpoint preview-policy.zip \\\n'
        '  --arm --start --out command-check.json\n\n'
        'Replace SOURCE_PROFILE with the source recording model: lags60-scratch\n'
        '(109 inputs) or seenflag-s1-level1 (85 inputs). The metadata is verified.\n'
        'Use a new output filename each time; previous evidence is preserved.\n'
        'The arm/start flags authorize this offline check only. Synthetic telemetry\n'
        'and operator heartbeats are labeled in the report. Recorded images cannot\n'
        'respond to these new memory commands: this is not a flight-success test.\n'
    ).encode()
    files['preview-policy.zip'] = (ROOT / 'trainedModels' / (profile + '.zip')).read_bytes()
    files['preview-manifest.json'] = json.dumps(dict(
        model_sha256=hashlib.sha256(files['preview-policy.zip']).hexdigest(),
        model=profile, group_master='71c577e', policy_inputs=input_count, flight_control=False,
        camera_calibration=calibration_status), indent=2).encode()
    intro = (f'BUNDLED POLICY: {profile}, {input_count} inputs.\n'
             'SHADOW PREVIEW ONLY: not cleared for physical flight.\n'
             f'CAMERA INTRINSICS: {calibration_status["label"]}.\n'
             'See CALIBRATION.md for calibration selection and provenance.\n\n')
    guide = (SOURCE / 'LIVE-START-HERE.txt').read_text().replace(
        'Mac-Lab-Recorder-Live.zip', out.name + '.zip').replace(
        'compatible 109-input SAC checkpoint', f'compatible {input_count}-input SAC checkpoint').replace(
        'Camera intrinsics remain the defaults recorded by the team unless a calib.npz is\nprovided.',
        'The manifest identifies bundled calibration or explicit team defaults. To select\n'
        'another calibration file, use the --calibration option described in CALIBRATION.md.')
    files['START-HERE.txt'] = (intro + guide).encode()
    files['LICENSE'] = (ROOT / 'LICENSE').read_bytes()
    hashes = {}
    out.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        if name.endswith(('.command', '.sh')):
            data = data.replace(b'\r\n', b'\n')
            files[name] = data
        path = out / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        hashes[name] = hashlib.sha256(data).hexdigest()
    files['PACKAGE-SHA256.json'] = json.dumps(hashes, indent=2).encode()
    (out / 'PACKAGE-SHA256.json').write_bytes(files['PACKAGE-SHA256.json'])
    with ZipFile(archive, 'x', ZIP_DEFLATED) as z:
        for name, data in files.items():
            entry = ZipInfo(out.name + '/' + name)
            entry.create_system = 3
            entry.external_attr = (0o100755 if name.endswith(('.command', '.sh')) else 0o100644) << 16
            entry.compress_type = ZIP_DEFLATED
            z.writestr(entry, data)
    with ZipFile(archive) as z:
        if z.testzip() is not None:
            raise RuntimeError('Package archive failed its CRC check')
        for name, digest in hashes.items():
            if hashlib.sha256(z.read(out.name + '/' + name)).hexdigest() != digest:
                raise RuntimeError(f'Package hash mismatch: {name}')
    print('Verified', archive, len(files), 'files', archive.stat().st_size, 'bytes')
    print('Camera intrinsics:', calibration_status['label'])
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['lags60-scratch', 'seenflag-s1-level1'],
                        default='seenflag-s1-level1')
    parser.add_argument('--out', type=Path, default=OUT, help='New directory (existing output is preserved)')
    parser.add_argument('--calibration', type=Path,
                        help='Measured calib.npz from src.labcheck; omit to package explicit defaults')
    args = parser.parse_args()
    try:
        build_package(args.profile, args.out, args.calibration)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))


if __name__ == '__main__':
    main()
