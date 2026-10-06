"""Replay every decoded camera frame through the live detector, without flight I/O."""
import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import detect


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    detector = detect.HoopDetector(confirm=detect.CONFIRM_FRAMES)
    cap = cv2.VideoCapture(str(args.video))
    counts, times, runs = Counter(), [], []
    n, begin = 0, None
    with (args.out/'frames.jsonl').open('w') as log:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            start = time.perf_counter()
            result = detector(frame)
            elapsed = time.perf_counter()-start
            times.append(elapsed)
            visible = bool(result.obs[6])
            counts['visible'] += visible
            counts['marker'] += bool(result.obs[7])
            counts[result.rejected or ('accepted' if visible else 'no_candidate')] += 1
            if visible and begin is None:
                begin = n
            if not visible and begin is not None:
                runs.append([begin, n-1])
                begin = None
            log.write(json.dumps(dict(frame=n, obs=result.obs.tolist(), rejected=result.rejected,
                                      inference_s=elapsed))+'\n')
            n += 1
    cap.release()
    if begin is not None:
        runs.append([begin, n-1])
    if not n:
        raise RuntimeError('No decodable frames')
    report = dict(video_sha256=hashlib.sha256(args.video.read_bytes()).hexdigest(),
                  detector_sha256=hashlib.sha256(Path(detect.__file__).read_bytes()).hexdigest(),
                  confirm=detect.CONFIRM_FRAMES, decoded_frames=n, counts=dict(counts),
                  visible_runs_frames=runs, inference_ms_percentiles=dict(zip(
                      ['p50','p95','p99'], (1000*np.percentile(times,[50,95,99])).tolist())),
                  timing_note='Frame indices only; no exposure timestamps or flight calibration.',
                  evaluation_note='Detection output counts, not accuracy; no exhaustive labels.')
    (args.out/'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k:v for k,v in report.items() if k not in ('counts','visible_runs_frames')},indent=2))
    print('Ring frames:', counts['visible'], 'marker frames:',counts['marker'], 'runs:',len(runs))

if __name__ == '__main__':
    main()
