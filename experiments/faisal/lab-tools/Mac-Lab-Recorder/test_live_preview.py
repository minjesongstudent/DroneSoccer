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
from live_preview import Preview, ROOT


class PreviewTests(unittest.TestCase):
    def test_camera_model_browser_and_log(self):
        preview = Preview(ROOT/'preview-policy.zip')
        try:
            with tempfile.TemporaryDirectory() as td:
                preview.start(Path(td))
                with patch('live_preview.webbrowser.open'):
                    preview.open_browser()
                base = f'http://127.0.0.1:{preview.server.server_port}'
                img = np.full((720,960,3),(40,200,220),np.uint8)
                cv2.circle(img,(480,330),170,(200,120,30),45)
                deadline = time.monotonic()+5
                while time.monotonic()<deadline:
                    preview.decoded(img,time.monotonic())
                    time.sleep(.12)
                    if len((Path(td)/'policy_preview.jsonl').read_text().splitlines())>=3:
                        break
                state = json.loads(urlopen(base+'/state',timeout=3).read())
                self.assertTrue(state['decoded'])
                self.assertNotIn('error',state)
                self.assertEqual(len(urlopen(base+'/frame.jpg',timeout=3).read())>1000,True)
                self.assertIn(b'No movement commands',urlopen(base,timeout=3).read())
                preview.close()
                records = [json.loads(s) for s in (Path(td)/'policy_preview.jsonl').read_text().splitlines()]
                self.assertGreaterEqual(len(records),3)
                self.assertTrue(any(r['observation'][6] or r['observation'][94] for r in records))
                for r in records:
                    self.assertEqual(len(r['observation']),109)
                    self.assertEqual(r['issued_action'],[0,0,0,0])
                    self.assertEqual(r['observation'][-13:-1],[0]*12)
                    self.assertTrue(np.isfinite(r['suggested_action']).all())
                self.assertTrue(preview.snapshot()['stale'])
                self.assertEqual(preview.snapshot()['suggestion'],[])
        finally:
            preview.close()
            if preview.server:
                preview.server.shutdown();preview.server.server_close()

    def test_recent_camera_cannot_refresh_old_suggestion(self):
        preview = Preview(ROOT/'preview-policy.zip')
        preview.decoded(np.zeros((4,4,3),np.uint8),time.monotonic())
        preview.state.update(suggestion=[1,1,1,1],suggestion_received=time.monotonic()-2)
        self.assertTrue(preview.snapshot()['stale'])
        self.assertEqual(preview.snapshot()['suggestion'],[])

if __name__ == '__main__':
    unittest.main()
