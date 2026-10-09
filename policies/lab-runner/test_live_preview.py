"""Offline camera/policy integration and stale-data checks; no drone sockets."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from urllib.request import urlopen

import numpy as np
import cv2
from live_preview import Preview, ROOT, select_calibration
from src.camera_calibration import default_calibration, load_calibration


class PreviewTests(unittest.TestCase):
    def test_camera_model_browser_and_log(self):
        preview = Preview()
        try:
            with tempfile.TemporaryDirectory() as td:
                preview.start(Path(td))
                with patch('live_preview.webbrowser.open'):
                    preview.open_browser()
                base = f'http://127.0.0.1:{preview.server.server_port}'
                img = np.full((720,960,3),(40,200,220),np.uint8)
                cv2.circle(img,(480,330),170,(200,120,30),45)
                deadline = time.monotonic()+15
                while time.monotonic()<deadline:
                    preview.decoded(img,time.monotonic())
                    time.sleep(.12)
                    captured = [json.loads(line) for line in (Path(td)/'policy_preview.jsonl').read_text().splitlines()]
                    if len(captured)>=3 and any(r['observation'][70] for r in captured):
                        break
                state = json.loads(urlopen(base+'/state',timeout=3).read())
                self.assertTrue(state['decoded'])
                self.assertNotIn('error',state)
                self.assertEqual(state['camera_calibration'], preview.calibration_status)
                self.assertEqual(len(urlopen(base+'/frame.jpg',timeout=3).read())>1000,True)
                self.assertIn(b'No movement commands',urlopen(base,timeout=3).read())
                preview.close()
                outcome_path=Path(td)/'preview_outcome.json'
                first_outcome=outcome_path.read_bytes()
                preview.close()
                self.assertEqual(outcome_path.read_bytes(),first_outcome)
                records = [json.loads(s) for s in (Path(td)/'policy_preview.jsonl').read_text().splitlines()]
                metadata = json.loads((Path(td)/'preview_metadata.json').read_text())
                self.assertEqual(metadata['camera_calibration'], preview.calibration_status)
                self.assertGreaterEqual(len(records),3)
                self.assertTrue(any(r['observation'][70] for r in records))
                for r in records:
                    self.assertEqual(len(r['camera_features']),8)
                    self.assertLessEqual(r['receipt_monotonic'],r['analysis_started_monotonic'])
                    self.assertLessEqual(r['analysis_started_monotonic'],r['analysis_completed_monotonic'])
                    self.assertEqual(len(r['raw_policy_action']),4)
                    self.assertEqual(len(r['observation']),preview.spec.inputs)
                    self.assertEqual(r['issued_action'],[0,0,0,0])
                    self.assertEqual(r['observation'][72:84],[0]*12)
                    self.assertTrue(np.isfinite(r['suggested_action']).all())
                self.assertTrue(preview.snapshot()['stale'])
                self.assertEqual(preview.snapshot()['suggestion'],[])
        finally:
            preview.close()
            if preview.server:
                preview.server.shutdown();preview.server.server_close()

    def test_recent_camera_cannot_refresh_old_suggestion(self):
        preview = Preview()
        preview.decoded(np.zeros((4,4,3),np.uint8),time.monotonic())
        preview.state.update(suggestion=[1,1,1,1],suggestion_received=time.monotonic()-2)
        self.assertTrue(preview.snapshot()['stale'])
        self.assertEqual(preview.snapshot()['suggestion'],[])

    def test_calibration_selection_and_tampered_bundle(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            path = root / 'calib.npz'
            defaults = default_calibration()
            np.savez(path, K=defaults.K, dist=defaults.dist, size=defaults.size, rms=.4)
            measured = load_calibration(path)
            # A stray file cannot silently change an old or default package.
            chosen, status = select_calibration({}, root=root)
            self.assertIsNone(chosen.sha256)
            self.assertEqual(status['status'], 'defaults')
            manifest = {'camera_calibration': dict(measured.status(), bundled_file='calib.npz')}
            chosen, status = select_calibration(manifest, root=root)
            self.assertEqual(chosen.sha256, measured.sha256)
            self.assertEqual(status['selected_by'], 'package manifest')
            manifest['camera_calibration']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'hash'):
                select_calibration(manifest, root=root)
            chosen, status = select_calibration(manifest, explicit_path=path, root=root)
            self.assertEqual(chosen.sha256, measured.sha256)
            self.assertEqual(status['selected_by'], 'explicit --calibration')
            with self.assertRaises(FileNotFoundError):
                select_calibration({}, explicit_path=root/'missing.npz', root=root)

    def test_calibration_manifest_cannot_escape_package_or_fall_back(self):
        with self.assertRaisesRegex(ValueError, 'path'):
            select_calibration({'camera_calibration': {'bundled_file': '../calib.npz'}})
        with self.assertRaisesRegex(ValueError, 'without'):
            select_calibration({'camera_calibration': {'status': 'calibration_file'}})

    def test_stream_calibration_aspect_error_is_visible(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            calibration = root/'wide-camera.npz'
            np.savez(calibration, K=[[920.,0.,640.],[0.,920.,360.],[0.,0.,1.]],
                     dist=np.zeros(5), size=[1280,720], rms=.4)
            preview = Preview(calibration)
            try:
                preview.start(root)
                preview.decoded(np.zeros((720,1280,3),np.uint8),time.monotonic())
                deadline = time.monotonic()+3
                while 'frame_id' not in preview.snapshot() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertIn('frame_id',preview.snapshot())
                self.assertNotIn('error',preview.snapshot())
                # A changed crop must stop analysis and report the actual problem.
                preview.decoded(np.zeros((720,960,3),np.uint8),time.monotonic())
                deadline = time.monotonic()+3
                while 'error' not in preview.snapshot() and time.monotonic()<deadline:
                    time.sleep(.02)
                self.assertIn('aspect ratio',preview.snapshot().get('error',''))
            finally:
                preview.close()

if __name__ == '__main__':
    unittest.main()
