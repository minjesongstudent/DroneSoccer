import csv
import json
from pathlib import Path
import tempfile
import unittest
from compare_session import analyse_session


class ComparisonTests(unittest.TestCase):
    def make(self, folder, telemetry):
        rows = [dict(frame_id=i, receipt_monotonic=t, camera_features=[0,0,0,0,0,0,1,0],
                     suggested_action=[.1,.2,.3,.4], issued_action=[0]*4)
                for i,t in enumerate([10.,10.1,11.])]
        (folder/'policy_preview.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        (folder/'telemetry.jsonl').write_text('\n'.join(map(json.dumps, telemetry)))

    def test_causal_join_never_uses_future_or_stale_values(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d)
            self.make(folder, [dict(receipt_monotonic=10.05, fields={'yaw':77})])
            result=analyse_session(folder)
            self.assertEqual(result['matched_frames'],1)
            with (folder/'comparison.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual([r['telemetry_yaw'] for r in rows],['','77',''])
            self.assertEqual(rows[1]['issued_right'],'0')
            self.assertEqual(rows[1]['suggested_right'],'0.1')

    def test_legacy_times_are_not_invented(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.make(folder,[dict(elapsed_s=.1,fields={'yaw':77})])
            self.assertEqual(analyse_session(folder)['matched_frames'],0)

    def test_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.make(folder,[])
            result=analyse_session(folder)
            original=(folder/'comparison.csv').read_bytes()
            with self.assertRaises(FileExistsError): analyse_session(folder)
            self.assertEqual((folder/'comparison.csv').read_bytes(),original)
            self.assertIn('policy_preview.jsonl',result['source_sha256'])

    def test_missing_telemetry_fields_remain_missing(self):
        with tempfile.TemporaryDirectory() as d:
            folder=Path(d);self.make(folder,[dict(receipt_monotonic=10.,fields={'bat':80})])
            self.assertEqual(analyse_session(folder)['matched_frames'],2)
            with (folder/'comparison.csv').open() as f: first=next(csv.DictReader(f))
            self.assertEqual(first['telemetry_yaw'],'')
            self.assertEqual(first['telemetry_bat'],'80')


if __name__ == '__main__': unittest.main()
