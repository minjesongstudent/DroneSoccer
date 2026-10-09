"""Local HTTP and mocked UDP verification, never a drone connection."""
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.request import Request,urlopen
from urllib.error import HTTPError
from supervised_flight import CaptureLink,FlightPreview


class WiringTests(unittest.TestCase):
    def test_selected_real_policy_produces_logged_commands_to_mock_only(self):
        from flight_control import FlightControl
        from src.policy_runtime import LearnedPolicy,SPECS
        from live_preview import ROOT
        import numpy as np
        now=[100.];commands=[];rows=[]
        runtime=LearnedPolicy(ROOT/'preview-policy.zip',SPECS['seenflag-s1-level1'])
        c=FlightControl(runtime,commands.append,rows.append,clock=lambda:now[0])
        c.capture_ready=True;c.receive_camera(100.);c.receive_telemetry(dict(bat=80,h=0,temph=30),100.)
        c.heartbeat();c.start();now[0]=100.1
        c.receive_camera(now[0]);c.receive_telemetry(dict(bat=80,h=80,temph=30),now[0]);c.heartbeat()
        c.acknowledge('ok',now[0]);c.watch()
        row=c.step(1,now[0],np.zeros(8,np.float32))
        self.assertEqual(len(row['observation']),85);self.assertTrue(row['sent'])
        self.assertTrue(commands[-1].startswith('rc '))
        self.assertEqual(rows[-1]['issued_action'],row['issued_action'])

    def test_capture_construction_opens_no_motor_command_and_send_logs_exact_bytes(self):
        with tempfile.TemporaryDirectory() as d:
            with patch('supervised_flight.socket.socket',side_effect=lambda *a:Mock()):
                link=CaptureLink(Path(d))
                try:
                    link.sockets[0].sendto.assert_not_called()
                    link.send('rc 10 -20 30 -40')
                    link.sockets[0].sendto.assert_called_once_with(b'rc 10 -20 30 -40',('192.168.10.1',8889))
                    row=json.loads((Path(d)/'commands.jsonl').read_text())
                    self.assertTrue(row['sent']);self.assertFalse(row['delivery_confirmed'])
                    with self.assertRaises(ValueError):link.send('wifi other password')
                finally:link.close()

    def test_local_ui_rejects_missing_token_and_never_starts_on_get(self):
        link=Mock();preview=FlightPreview(link)
        try:
            with patch('supervised_flight.webbrowser.open'):preview.open_browser()
            base='http://127.0.0.1:'+str(preview.server.server_port)
            self.assertIn(b'HOLD: take off',urlopen(base).read())
            link.send.assert_not_called()
            with self.assertRaises(HTTPError) as e:
                urlopen(Request(base+'/control',data=b'{"action":"start"}',method='POST'))
            self.assertEqual(e.exception.code,403);link.send.assert_not_called()
            c=preview.control;c.capture_ready=True
            stamp=time.monotonic();c.receive_camera(stamp);c.receive_telemetry(dict(bat=80,h=0,temph=30),stamp)
            def post(action):
                return urlopen(Request(base+'/control',data=json.dumps(dict(action=action)).encode(),
                    headers={'X-Flight-Token':preview.token,'Content-Type':'application/json'},method='POST')).read()
            c.frame_at=None
            with self.assertRaises(HTTPError) as rejected:post('start')
            failure=json.loads(rejected.exception.read())
            self.assertEqual(failure['control_state'],'ground')
            self.assertEqual(failure['error'],'stale camera');link.send.assert_not_called()
            c.receive_camera(time.monotonic())
            post('heartbeat');post('start');link.send.assert_called_with('takeoff')
            post('land');self.assertEqual(c.state,'landing');link.send.assert_called_with('land')
            with self.assertRaises(HTTPError):post('start')
        finally:
            preview.close()
            if preview.server:preview.server.shutdown();preview.server.server_close()

    def test_capture_countdown_uses_monotonic_deadline(self):
        preview=FlightPreview(Mock())
        try:
            self.assertIsNone(preview.snapshot()['capture_remaining_s'])
            with patch('supervised_flight.time.monotonic',return_value=100.):
                preview.capture_deadline=120.
                self.assertEqual(preview.snapshot()['capture_remaining_s'],20.)
                preview.capture_deadline=99.
                self.assertEqual(preview.snapshot()['capture_remaining_s'],0.)
        finally:preview.close()

    def test_csv_labels_physical_records_without_inventing_delivery(self):
        from compare_session import analyse_session
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            (folder/'preview_metadata.json').write_text(json.dumps(dict(flight_control=True)))
            (folder/'policy_preview.jsonl').write_text(json.dumps(dict(frame_id=1,receipt_monotonic=1.,
                camera_features=[0]*8,issued_action=[],suggested_action=[0]*4,sent=False,command='rc 0 0 0 0'))+'\n')
            result=analyse_session(folder)
            self.assertEqual(result['mode'],'physical supervised policy test')
            self.assertFalse(any('were not sent' in s for s in result['limitations']))


if __name__=='__main__':unittest.main()
