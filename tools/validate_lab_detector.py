"""Freeze blind image labels and score sequential detector replay, without flight I/O.

prepare exports deterministic raw frames and an unlabelled manifest. Review the
images before filling labels; evaluate never infers truth from detector outputs.
Existing tuning footage is retrospective, even if these particular frames are new.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(video, out, stride=75, offset=17):
    """Use sequential decode, never seeking in the damaged elementary stream."""
    if stride < 1 or not 0 <= offset < stride:
        raise ValueError("Require stride > 0 and 0 <= offset < stride")
    if out.exists() and any(out.iterdir()):
        raise ValueError("Use a new output directory; a frozen package is never overwritten")
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir()
    frames, n = [], 0
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if n >= offset and (n - offset) % stride == 0:
                name = f"frames/frame-{n:06d}.png"
                if not cv2.imwrite(str(out / name), frame):
                    raise RuntimeError(f"Cannot save frame {n}")
                frames.append(dict(frame=n, image=name, image_sha256=sha256(out / name),
                                   pixels_sha256=hashlib.sha256(frame.tobytes()).hexdigest(),
                                   width=frame.shape[1], height=frame.shape[0],
                                   reviewed=False, ring=None, marker=None, ring_view="unknown",
                                   reviewer="", notes=""))
            n += 1
    finally:
        cap.release()
    if not frames:
        raise ValueError("No selected decodable frames")
    manifest = dict(schema_version=1, video_sha256=sha256(video), decoded_frames=n,
                    decoder_opencv=cv2.__version__,
                    selection=dict(stride=stride, offset=offset, method="uniform sequential frame indices"),
                    independence="retrospective; source recording previously used for tuning",
                    purpose="Freeze before evaluating future changes; never tune on these labels",
                    label_policy={
                        "ring": "true: unmistakable ring visible, including clipped arcs; false: no ring visible; null: uncertain",
                        "marker": "true: front disc with dark cross visibly identifiable; false: absent or clearly blank/back-facing; null: ambiguous, blurred or too small",
                        "ring_view": "full, clipped, absent, or unknown; net/pole occlusion alone does not mean clipped",
                        "reviewed": "true only after visually inspecting the actual image; reviewer identifies AI or human provenance",
                    }, frames=frames)
    write_json(out / "labels.json", manifest)
    write_json(out / "selection.json", {k: v for k, v in manifest.items() if k != "frames"} | {
        "frames": [{k: row[k] for k in ("frame", "image", "image_sha256", "pixels_sha256", "width", "height")}
                   for row in frames]})
    cards = [f'<figure><a href="{html.escape(r["image"])}"><img src="{html.escape(r["image"])}"></a>'
             f'<figcaption>Decoded frame {r["frame"]}</figcaption></figure>' for r in frames]
    (out / "review.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Blind detector annotation review</title>'
        '<style>body{font:18px system-ui;background:#222;color:white}main{display:flex;flex-wrap:wrap}'
        'figure{margin:8px}img{width:480px;max-width:95vw}</style>'
        '<h1>Blind image review</h1><p>No predictions or suggested labels. Click for full resolution.</p>'
        '<main>' + ''.join(cards) + '</main>', encoding="utf-8")
    for start in range(0, len(frames), 6):
        sheet = np.full((780, 1440, 3), 35, np.uint8)
        for i, row in enumerate(frames[start:start + 6]):
            image = cv2.imread(str(out / row["image"]))
            tile = cv2.resize(image, (480, 360))
            x, y = (i % 3) * 480, (i // 3) * 390
            sheet[y:y + 360, x:x + 480] = tile
            cv2.putText(sheet, f'frame {row["frame"]}', (x + 10, y + 382),
                        cv2.FONT_HERSHEY_SIMPLEX, .6, (255, 255, 255), 1)
        cv2.imwrite(str(out / f"contact-{start // 6:02d}.jpg"), sheet)
    return {"decoded_frames": n, "selected_frames": len(frames), "out": str(out)}


def validate_labels(manifest):
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported label schema")
    seen = set()
    for row in manifest["frames"]:
        idx = row["frame"]
        if type(idx) is not int or idx < 0 or idx in seen:
            raise ValueError("Frame indices must be unique nonnegative integers")
        seen.add(idx)
        if type(row["reviewed"]) is not bool:
            raise ValueError("reviewed must be a boolean")
        for key in ("ring", "marker"):
            if row[key] is not None and type(row[key]) is not bool:
                raise ValueError(f"{key} labels must be true, false or null")
            if row[key] is not None and not row["reviewed"]:
                raise ValueError("Unreviewed frames cannot contain truth labels")
        if row["reviewed"] and not row["reviewer"].strip():
            raise ValueError("Reviewed labels require reviewer provenance")
        if row["ring_view"] not in {"full", "clipped", "absent", "unknown"}:
            raise ValueError("Invalid ring_view")
        if row["ring"] is False and row["ring_view"] != "absent":
            raise ValueError("Absent ring requires ring_view=absent")
        if row["ring"] is True and row["ring_view"] not in {"full", "clipped"}:
            raise ValueError("Visible ring requires full or clipped view")
        if row["marker"] is True and row["ring"] is False:
            raise ValueError("A front marker cannot belong to an absent ring")


def binary_metrics(truth_prediction_pairs):
    counts = Counter({"tp": 0, "fp": 0, "tn": 0, "fn": 0})
    for truth, prediction in truth_prediction_pairs:
        counts[("t" if truth == prediction else "f") + ("p" if prediction else "n")] += 1
    tp, fp, tn, fn = (counts[k] for k in ("tp", "fp", "tn", "fn"))
    divide = lambda a, b: a / b if b else None
    return dict(counts, labelled=tp + fp + tn + fn, positives=tp + fn, negatives=tn + fp,
                precision=divide(tp, tp + fp), recall=divide(tp, tp + fn),
                false_positive_rate=divide(fp, fp + tn), accuracy=divide(tp + tn, tp + fp + tn + fn))


def summarize(manifest, predictions):
    """Null and unreviewed labels never enter any accuracy denominator."""
    validate_labels(manifest)
    frames = manifest["frames"]
    metrics = {}
    for target in ("ring", "marker"):
        verified = [r for r in frames if r["reviewed"] and r[target] is not None]
        missing = [r["frame"] for r in verified if r["frame"] not in predictions]
        if missing:
            raise ValueError(f"Missing predictions for reviewed frames: {missing}")
        metrics[target] = binary_metrics((r[target], predictions[r["frame"]][target]) for r in verified)
        metrics[target]["excluded_unreviewed_or_unknown"] = len(frames) - len(verified)
    metrics["ring_by_view"] = {
        view: binary_metrics((r["ring"], predictions[r["frame"]]["ring"])
                             for r in frames if r["reviewed"] and r["ring"] is not None
                             and r["ring_view"] == view)
        for view in ("full", "clipped", "absent")}
    return metrics


def evaluate(video, labels, out):
    from src import detect
    from src.params import CALIB_PATH

    if out.exists():
        raise ValueError("Choose a new report file; previous evidence is never overwritten")
    manifest = json.loads(labels.read_text(encoding="utf-8"))
    labels_hash = sha256(labels)
    validate_labels(manifest)
    video_hash = sha256(video)
    if video_hash != manifest["video_sha256"]:
        raise ValueError("Video hash does not match frozen labels")
    selection_path = labels.with_name("selection.json")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    selection_hash = sha256(selection_path)
    frozen = {r["frame"]: r for r in selection["frames"]}
    if (selection["video_sha256"] != video_hash or len(frozen) != len(manifest["frames"])
            or selection["decoded_frames"] != manifest["decoded_frames"]
            or set(frozen) != {r["frame"] for r in manifest["frames"]}):
        raise ValueError("Labels do not match frozen selection")
    rows = {r["frame"]: r for r in manifest["frames"]}
    for row in rows.values():
        if any(row[k] != frozen[row["frame"]][k] for k in frozen[row["frame"]]):
            raise ValueError("Frozen frame metadata changed")
        if sha256(labels.parent / row["image"]) != row["image_sha256"]:
            raise ValueError(f'Image hash mismatch for frame {row["frame"]}')
    source_hashes = {name: sha256(ROOT / name) for name in (
        "src/detect.py", "src/contract.py", "src/params.py", "src/camera_calibration.py",
        "tools/validate_lab_detector.py")}
    calibration_hash = sha256(CALIB_PATH) if CALIB_PATH.exists() else None
    detector = detect.HoopDetector(confirm=detect.CONFIRM_FRAMES)
    predictions, n = {}, 0
    cap = cv2.VideoCapture(str(video))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = detector(frame)  # Include every prior frame: temporal state matters.
            if n in rows:
                if hashlib.sha256(frame.tobytes()).hexdigest() != rows[n]["pixels_sha256"]:
                    raise ValueError(f"Decoded pixels changed at frame {n}; re-review required")
                predictions[n] = dict(ring=bool(result.obs[6]), marker=bool(result.obs[7]),
                                      rejected=result.rejected)
            n += 1
    finally:
        cap.release()
    if n != manifest["decoded_frames"]:
        raise ValueError("Decoded frame count changed")
    changed = [name for name, digest in source_hashes.items() if sha256(ROOT / name) != digest]
    if changed:
        raise ValueError("Detector sources changed during evaluation; rerun against a stable snapshot: "
                         + ", ".join(changed))
    if calibration_hash != (sha256(CALIB_PATH) if CALIB_PATH.exists() else None):
        raise ValueError("Calibration changed during evaluation")
    if labels_hash != sha256(labels) or selection_hash != sha256(selection_path):
        raise ValueError("Frozen labels or selection changed during evaluation")
    report = dict(video_sha256=video_hash, labels_sha256=labels_hash,
                  selection_sha256=selection_hash, detector_sources_sha256=source_hashes,
                  calibration_sha256=calibration_hash, opencv=cv2.__version__,
                  calibration_status=getattr(detector, "calibration_status", None),
                  confirm=detect.CONFIRM_FRAMES, replayed_frames=n, selected_frames=len(rows),
                  independence=manifest["independence"],
                  metrics=summarize(manifest, predictions),
                  caveats=["Single tuning capture; retrospective sample metrics are not unseen-test estimates.",
                           "Frame samples are correlated; no independent-frame confidence interval claimed.",
                           "Visual presence labels only; no ellipse, pose, flight-clearance or timing accuracy measured.",
                           "Ring-presence recall includes heavily clipped views; it is not fitted-ellipse accuracy.",
                           "Reviewer provenance is retained; AI-reviewed labels need human adjudication before acceptance."],
                  predictions=[dict(frame=k, **v) for k, v in sorted(predictions.items())])
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, report)
    return {"out": str(out), "metrics": report["metrics"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("video", type=Path)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--stride", type=int, default=75)
    prep.add_argument("--offset", type=int, default=17)
    score = sub.add_parser("evaluate")
    score.add_argument("video", type=Path)
    score.add_argument("--labels", type=Path, required=True)
    score.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.video, args.out, args.stride, args.offset)
    else:
        result = evaluate(args.video, args.labels, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
