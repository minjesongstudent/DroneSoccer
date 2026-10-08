"""Read-only classification of saved learned trajectories; no flight or training."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def classify(row, path):
    with np.load(path) as trace:
        obs = trace['actor_observations']
        assert obs.shape[1] in (85,121) and np.isfinite(obs).all()
        frames = obs[:, :72].reshape(-1, 9, 8)
        visible = frames[:, -1, 6] > .5
        empty = ~(frames[:, :, 6] > .5).any(axis=1)
        blind = 0
        for value in visible[::-1]:
            if value: break
            blind += 1
        last_sighting = np.flatnonzero(visible)
        # Observations precede each recorded action; terminal position is after it.
        pos = trace['positions'][-1]
        normal = trace['hoop_normal']
        delta = pos - trace['hoop_centre']
        signed = float(delta @ normal)
        radial = float(np.linalg.norm(delta - signed*normal))
    return dict(seed=row['seed'], outcome=row['outcome'], crashed=row['crashed'],
                passes=row['full_body_passes'], returns=row['returns_to_front'],
                terminal_blind_steps=blind, terminal_history_empty=bool(empty[-1]),
                never_seen=not len(last_sighting), empty_history_steps=int(empty.sum()),
                terminal_plane_distance=signed, terminal_radial_distance=radial,
                phase='after_pass' if row['full_body_passes'] else 'before_first_pass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--candidate-label',choices=['boundary','adapter'],default='boundary')
    args = parser.parse_args()
    assert (args.run/'DONE').exists(), 'Comparison must finish before audit'
    args.out.mkdir(parents=True, exist_ok=False)
    records = {}
    hashes = {}
    for model in ['original', args.candidate_label]:
        for suite in ['aligned', 'nominal', 'randomised']:
            folder = args.run/model/suite
            log = folder/'episodes.jsonl'
            hashes[str(log)] = hashlib.sha256(log.read_bytes()).hexdigest()
            rows = [json.loads(line) for line in log.read_text().splitlines()]
            values = [classify(row, folder/f"{row['seed']}.npz") for row in rows]
            groups = defaultdict(list)
            for value in values:
                if value['crashed']:
                    groups[value['outcome']+'/'+value['phase']].append(value)
            records[model+'/'+suite] = dict(
                groups={key: dict(count=len(group),
                                 terminal_history_empty=sum(v['terminal_history_empty'] for v in group),
                                 never_seen=sum(v['never_seen'] for v in group),
                                 median_terminal_blind_steps=float(np.median([v['terminal_blind_steps'] for v in group])))
                        for key, group in groups.items()}, rows=values)
    (args.out/'audit.json').write_text(json.dumps(dict(records=records, input_hashes=hashes,
        note='Pre-action visibility; associations do not prove causation. Development only.'), indent=2))
    shutil.copy2(__file__,args.out/'audit_source.py')
    for key, value in records.items():
        print(key, json.dumps(value['groups']))


if __name__ == '__main__':
    main()
