"""Exercise packet decoding and the group policy on a saved recorder session."""
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package',type=Path,required=True)
    parser.add_argument('--session',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    sys.path.insert(0,str(args.package.resolve()))
    from live_preview import Preview
    preview = Preview(args.package.resolve()/'preview-policy.zip')
    args.out.mkdir(parents=True,exist_ok=True)
    preview.start(args.out)
    # Recorded payload sizes and offsets are replayed; no network connection.
    records = [json.loads(s) for s in (args.session/'video_packets.jsonl').read_text().splitlines()]
    try:
        with (args.session/'camera.h264').open('rb') as f:
            for record in records:
                while preview.chunks.qsize()>200:
                    time.sleep(.005)
                f.seek(record['offset'])
                preview.feed(f.read(record['size']),time.monotonic())
        deadline=time.monotonic()+15
        while not preview.chunks.empty() and time.monotonic()<deadline:
            time.sleep(.05)
        time.sleep(.5)
    finally:
        preview.close()
    rows=[json.loads(s) for s in (args.out/'policy_preview.jsonl').read_text().splitlines()]
    assert rows, 'No camera-to-policy predictions'
    assert preview.state['decoded']>3000, preview.state
    assert not preview.chunks.qsize(), 'Decoder failed to drain'
    assert 'error' not in preview.state, preview.state
    assert all(len(r['observation'])==109 and r['issued_action']==[0,0,0,0] for r in rows)
    report=dict(decoded=preview.state['decoded'],prediction_records=len(rows),
                preview_packet_drops=preview.state['dropped'],
                decode_error_events=preview.state.get('decode_errors',0),
                physical_commands_sent=0,mode='accelerated packet replay; not timing validation')
    (args.out/'check.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':
    main()
