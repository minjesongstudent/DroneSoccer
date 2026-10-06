"""Ground-only Tello EDU data recorder. Python 3.10+, standard library only."""
import argparse
import hashlib
import json
import platform
import select
import socket
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ALLOWED = frozenset({'command', 'streamon', 'streamoff'})

def control_reply(data, address):
    """Accept only SDK acknowledgements from the expected command endpoint."""
    if address != ('192.168.10.1', 8889):
        return None
    try:
        text = data.decode('ascii').strip(' \r\n')
    except UnicodeDecodeError:
        return None
    if not text or any(ord(c) < 32 or ord(c) > 126 for c in text):
        return None
    if text.lower() == 'ok' or text.lower() == 'error' or text.lower().startswith('error '):
        return text
    return None

def request_sdk(control, command, pump, commands, clock=time.monotonic, timeout=8):
    send(control, command)
    commands.write(json.dumps({'utc': utc(), 'command': command}) + '\n')
    commands.flush()
    deadline = clock() + timeout
    while clock() < deadline:
        for reply in pump(.1):
            if reply.lower() == 'ok':
                return
            raise RuntimeError(command + ': ' + reply)
    raise TimeoutError(command + ': no valid SDK acknowledgement within eight seconds. '
                       'Unexpected packets, if any, are saved in commands.jsonl. '
                       'Check drone Wi-Fi, close other controllers, and retry with motors off.')

def utc():
    return datetime.now(timezone.utc).isoformat()

def parse_state(raw):
    fields = {}
    for item in raw.strip().split(';'):
        key, sep, value = item.partition(':')
        if sep:
            try:
                value = float(value) if '.' in value else int(value)
            except ValueError:
                pass
            fields[key] = value
    return fields

def send(sock, command):
    if command not in ALLOWED:
        raise ValueError('Flight and configuration commands are disabled')
    sock.sendto(command.encode('ascii'), ('192.168.10.1', 8889))

def export(folder):
    folder = Path(folder).resolve()
    if not (folder / 'session.json').is_file():
        raise ValueError('Choose a session folder containing session.json')
    files = sorted(p for p in folder.rglob('*') if p.is_file() and not p.is_symlink()
                   and p.name != 'SHA256.json')
    manifest = {}
    for p in files:
        h = hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda: f.read(1024 * 1024), b''):
                h.update(block)
        manifest[p.relative_to(folder).as_posix()] = h.hexdigest()
    (folder / 'SHA256.json').write_text(json.dumps(manifest, indent=2))
    destination = folder.with_name(folder.name + '-export-' + str(time.time_ns()) + '.zip')
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as z:
        for p in files + [folder / 'SHA256.json']:
            z.write(p, folder.name + '/' + p.relative_to(folder).as_posix())
    print('Transfer this file:', destination)
    return destination

def record(seconds):
    root = Path(__file__).resolve().parent
    folder = root / 'sessions' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    folder.mkdir(parents=True)
    (folder / 'measurements.md').write_text((root / 'MEASUREMENTS.md').read_text())
    info = {'started_utc': utc(), 'platform': platform.platform(),
            'python': platform.python_version(), 'mode': 'motors-off hand-carried capture',
            'timing': 'Host receipt timestamps; not exposure time or measured network latency',
            'requested_seconds': seconds, 'state_packets': 0, 'video_packets': 0,
            'video_bytes': 0, 'ignored_control_packets': 0, 'recorder_version': 2, 'error': None}
    sockets = []
    start = time.monotonic()
    try:
        for port in (8889, 8890, 11111):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sockets.append(s)
            s.bind(('', port))
            s.setblocking(False)
        control, state, video = sockets
        with (folder / 'telemetry.jsonl').open('w') as states, \
             (folder / 'video_packets.jsonl').open('w') as packets, \
             (folder / 'commands.jsonl').open('w') as commands, \
             (folder / 'camera.h264').open('wb') as movie:
            def pump(timeout):
                replies = []
                for s in select.select(sockets, [], [], timeout)[0]:
                    data, address = s.recvfrom(65535)
                    if address[0] != '192.168.10.1':
                        continue
                    stamp = {'utc': utc(), 'elapsed_s': time.monotonic() - start}
                    if s is state:
                        raw = data.decode('utf-8', errors='replace')
                        states.write(json.dumps(dict(stamp, raw=raw, fields=parse_state(raw))) + '\n')
                        states.flush()
                        info['state_packets'] += 1
                    elif s is video:
                        packets.write(json.dumps(dict(stamp, offset=movie.tell(), size=len(data))) + '\n')
                        movie.write(data)
                        info['video_packets'] += 1
                        info['video_bytes'] += len(data)
                    else:
                        reply = control_reply(data, address)
                        commands.write(json.dumps(dict(stamp, response=reply,
                            source_ip=address[0], source_port=address[1],
                            raw_hex=data.hex(), accepted=reply is not None)) + '\n')
                        commands.flush()
                        if reply is not None:
                            replies.append(reply)
                        else:
                            info['ignored_control_packets'] += 1
                return replies

            def request(command):
                request_sdk(control, command, pump, commands)

            try:
                request('command')
                request('streamon')
                print('Recording with motors OFF. Ctrl+C stops and saves. Folder:', folder, flush=True)
                deadline = time.monotonic() + seconds
                report_at = 0
                while time.monotonic() < deadline:
                    pump(.1)
                    if time.monotonic() >= report_at:
                        print('Telemetry packets:', info['state_packets'], 'Video bytes:', info['video_bytes'], flush=True)
                        report_at = time.monotonic() + 5
            finally:
                # No landing, motor, movement or Wi-Fi configuration commands are sent.
                send(control, 'streamoff')
                commands.write(json.dumps({'utc': utc(), 'command': 'streamoff', 'reply_waited': False}) + '\n')
    except KeyboardInterrupt:
        info['stopped_by_user'] = True
    except Exception as exc:
        info['error'] = str(exc)
        print('Recording issue:', exc)
    finally:
        for s in sockets:
            s.close()
        info['ended_utc'] = utc()
        info['elapsed_s'] = time.monotonic() - start
        info['capture_present'] = bool(info['state_packets'] and info['video_bytes'])
        (folder / 'session.json').write_text(json.dumps(info, indent=2))
        print('Saved:', folder)
        if not info['capture_present']:
            print('INCOMPLETE: video or telemetry missing. Keep this folder for troubleshooting.')
        export(folder)
    return 1 if info['error'] or not info['capture_present'] else 0

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seconds', type=int, default=120)
    parser.add_argument('--export', type=Path, help='ZIP a finished session after adding measurements/photos')
    args = parser.parse_args()
    if args.export:
        export(args.export)
    else:
        if not 1 <= args.seconds <= 300:
            parser.error('Use 1 to 300 seconds per ground capture')
        raise SystemExit(record(args.seconds))
