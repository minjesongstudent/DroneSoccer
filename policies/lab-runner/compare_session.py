"""Join recorded policy frames to preceding telemetry; never connects to hardware."""
import argparse
from bisect import bisect_right
import csv
import hashlib
import json
import math
from pathlib import Path

AXES = ('right', 'forward', 'up', 'clockwise_yaw')
FIELDS = ('pitch', 'roll', 'yaw', 'vgx', 'vgy', 'vgz', 'h', 'tof', 'baro', 'bat')


def rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def analyse_session(folder, max_age=.25):
    folder = Path(folder)
    if not number(max_age) or max_age <= 0:
        raise ValueError('max_age must be positive and finite')
    preview = rows(folder/'policy_preview.jsonl')
    telemetry = rows(folder/'telemetry.jsonl')
    startup = rows(folder/'startup_search.jsonl')
    metadata_path = folder/'preview_metadata.json'
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    physical = metadata.get('flight_control') is True
    outcome_path = folder/'preview_outcome.json'
    outcome = json.loads(outcome_path.read_text()) if outcome_path.exists() else {}
    valid = sorted((r for r in telemetry if number(r.get('receipt_monotonic'))),
                   key=lambda r: r['receipt_monotonic'])
    times = [r['receipt_monotonic'] for r in valid]
    output, matched, visible, resets, stale = [], 0, 0, 0, 0
    for row in preview:
        stamp = row.get('receipt_monotonic')
        idx = bisect_right(times, stamp)-1 if number(stamp) else -1
        measured = valid[idx] if idx >= 0 else None
        age = stamp-measured['receipt_monotonic'] if measured else None
        usable = measured is not None and age <= max_age
        matched += int(usable)
        features = row.get('camera_features', [])
        ring = features[6] if len(features) == 8 else None
        visible += int(ring == 1)
        resets += int(bool(row.get('history_reset')))
        suggested = row.get('suggested_action', [])
        stale += int(len(suggested) != 4)
        item = dict(frame_id=row.get('frame_id'), frame_receipt_monotonic=stamp,
                    analysis_started_monotonic=row.get('analysis_started_monotonic'),
                    analysis_completed_monotonic=row.get('analysis_completed_monotonic'),
                    frame_age_s=row.get('frame_age_s'), tick_gap_s=row.get('tick_gap_s'),
                    history_reset=row.get('history_reset'), ring_visible=ring,
                    command=row.get('command'),sent=row.get('sent'),
                    marker_visible=features[7] if len(features) == 8 else None,
                    telemetry_matched=usable, telemetry_age_s=age,
                    telemetry_receipt_monotonic=measured['receipt_monotonic'] if measured else None)
        for label, values in (('suggested', suggested), ('issued', row.get('issued_action', []))):
            for i, axis in enumerate(AXES):
                item[label+'_'+axis] = values[i] if len(values) == 4 else None
        for field in FIELDS:
            item['telemetry_'+field] = measured.get('fields', {}).get(field) if usable else None
        output.append(item)
    # Exclusive creation preserves any previous report and source evidence.
    csv_path, summary_path = folder/'comparison.csv', folder/'comparison-summary.json'
    if csv_path.exists() or summary_path.exists():
        raise FileExistsError('Comparison already exists; preserve it and use a fresh session')
    with csv_path.open('x', newline='') as f:
        if output:
            writer = csv.DictWriter(f, fieldnames=list(output[0]))
            writer.writeheader(); writer.writerows(output)
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (folder/'policy_preview.jsonl', folder/'startup_search.jsonl', folder/'telemetry.jsonl', metadata_path, outcome_path)
                     if p.exists()}
    summary = dict(schema_version=1, mode=('physical supervised policy test' if physical else 'recorded shadow-policy comparison'),
        model=metadata.get('model'), model_sha256=metadata.get('model_sha256'),
        policy_identity=metadata.get('policy_identity'),
        startup_search_sends=sum(r.get('event')=='startup_send_result' and r.get('sent') is True for r in startup),
        startup_search_counted_as_learned_policy=False,
        camera_calibration=metadata.get('camera_calibration'), frames=len(preview),
        preview_error=outcome.get('error'), decoded_frames=outcome.get('decoded'),
        dropped_preview_packets=outcome.get('dropped'),
        telemetry_packets=len(telemetry), timestamped_telemetry_packets=len(valid),
        matched_frames=matched, unmatched_frames=len(preview)-matched,
        frames_with_visible_ring=visible, history_resets=resets,
        frames_without_fresh_suggestion=stale, max_telemetry_age_s=max_age,
        source_sha256=source_hashes,
        timing='Latest telemetry received at or before the frame receipt; host clock only, not exposure or latency measurement.',
        telemetry_units='Raw SDK field values, unchanged; no conversion to policy action units.',
        limitations=['No measured global 3D trajectory is reconstructed.',
            ('UDP send is not delivery or motor-state confirmation; telemetry is not a measured global path.' if physical
             else 'Suggested actions were not sent: these records cannot measure their physical response.'),
            'Legacy logs without telemetry monotonic timestamps are left unmatched.',
            'No simulation-to-real error or physical flight success is inferred.'])
    with summary_path.open('x') as f:
        json.dump(summary, f, indent=2, allow_nan=False)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('session', type=Path)
    args = parser.parse_args()
    print(json.dumps(analyse_session(args.session), indent=2))
