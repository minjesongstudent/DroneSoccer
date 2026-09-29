import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import Mock
import recorder

class RecorderTests(unittest.TestCase):
    def test_no_flight_commands(self):
        sock = Mock()
        for command in ['takeoff', 'land', 'rc 0 0 0 0', 'emergency', 'wifi x y', 'command\ntakeoff']:
            with self.assertRaises(ValueError):
                recorder.send(sock, command)
        sock.sendto.assert_not_called()
        recorder.send(sock, 'streamon')
        sock.sendto.assert_called_once_with(b'streamon', ('192.168.10.1', 8889))

    def test_missing_and_unknown_telemetry(self):
        result = recorder.parse_state('yaw:-12;baro:1.23;extra:unknown;broken;')
        self.assertEqual(result, {'yaw': -12, 'baro': 1.23, 'extra': 'unknown'})
        self.assertNotIn('h', result)

    def test_zip_preserves_data_and_hashes(self):
        with tempfile.TemporaryDirectory() as td:
            folder = Path(td) / 'session'
            folder.mkdir()
            (folder / 'session.json').write_text('{}')
            data = b'\x00\x00\x00\x01test'
            (folder / 'camera.h264').write_bytes(data)
            archive = recorder.export(folder)
            with zipfile.ZipFile(archive) as z:
                self.assertEqual(z.read('session/camera.h264'), data)
                hashes = json.loads(z.read('session/SHA256.json'))
                self.assertEqual(hashes['camera.h264'], hashlib.sha256(data).hexdigest())

if __name__ == '__main__':
    unittest.main()
