import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools.validate_lab_detector import binary_metrics, evaluate, prepare, summarize, validate_labels


def row(frame, ring, marker, view, reviewed=True):
    return dict(frame=frame, ring=ring, marker=marker, ring_view=view,
                reviewed=reviewed, reviewer="test reviewer" if reviewed else "")


def test_unknown_and_unreviewed_are_excluded_per_target():
    labels = dict(schema_version=1, frames=[
        row(1, True, None, "clipped"), row(2, False, False, "absent"),
        row(3, None, None, "unknown", reviewed=False), row(4, True, True, "full")])
    metrics = summarize(labels, {1: dict(ring=False, marker=False),
                                2: dict(ring=False, marker=False),
                                4: dict(ring=True, marker=False)})
    assert metrics["ring"]["recall"] == .5
    assert metrics["ring"]["labelled"] == 3
    assert metrics["marker"]["labelled"] == 2
    assert metrics["marker"]["fn"] == 1
    assert metrics["ring_by_view"]["clipped"]["fn"] == 1
    assert metrics["ring_by_view"]["full"]["tp"] == 1


def test_no_denominator_is_null_not_perfect_accuracy():
    metrics = binary_metrics([])
    assert metrics["labelled"] == 0
    assert all(metrics[key] is None for key in ("precision", "recall", "accuracy", "false_positive_rate"))
    assert binary_metrics([(False, True)])["fp"] == 1


@pytest.mark.parametrize("change", [
    {"reviewed": False}, {"reviewer": ""}, {"ring": 1},
    {"frame": -1}, {"ring_view": "absent"}, {"marker": "unknown"},
])
def test_invalid_truth_labels_fail_closed(change):
    entry = row(1, True, True, "clipped") | change
    with pytest.raises(ValueError):
        validate_labels(dict(schema_version=1, frames=[entry]))


def test_duplicate_frame_or_missing_prediction_is_rejected():
    entry = row(1, True, True, "clipped")
    with pytest.raises(ValueError, match="unique"):
        validate_labels(dict(schema_version=1, frames=[entry, entry]))
    with pytest.raises(ValueError, match="Missing predictions"):
        summarize(dict(schema_version=1, frames=[entry]), {})


class FakeVideo:
    def __init__(self, frames):
        self.frames = iter(frames)

    def read(self):
        image = next(self.frames, None)
        return image is not None, image

    def release(self):
        pass


def test_replay_uses_unsampled_history_and_rejects_tampered_images(tmp_path, monkeypatch):
    from tools import validate_lab_detector as validation
    from src import detect

    images = [np.full((24, 32, 3), n, np.uint8) for n in range(4)]
    monkeypatch.setattr(validation.cv2, "VideoCapture", lambda _: FakeVideo(images))
    video = tmp_path / "recording.h264"
    video.write_bytes(b"test fixture")
    package = tmp_path / "package"
    result = prepare(video, package, stride=2, offset=1)
    assert result["selected_frames"] == 2
    path = package / "labels.json"
    manifest = json.loads(path.read_text())
    assert all(not r["reviewed"] and r["ring"] is None for r in manifest["frames"])
    for entry in manifest["frames"]:
        entry.update(ring=True, marker=False, ring_view="full", reviewed=True, reviewer="test reviewer")
    path.write_text(json.dumps(manifest))
    seen = []

    class TemporalDetector:
        def __init__(self, confirm):
            assert confirm == detect.CONFIRM_FRAMES

        def __call__(self, frame):
            seen.append(int(frame[0, 0, 0]))
            obs = np.zeros(8)
            obs[6] = len(seen) >= 2
            return SimpleNamespace(obs=obs, rejected="")

    monkeypatch.setattr(detect, "HoopDetector", TemporalDetector)
    out = tmp_path / "report.json"
    evaluate(video, path, out)
    assert seen == [0, 1, 2, 3]
    assert json.loads(out.read_text())["metrics"]["ring"]["tp"] == 2
    with pytest.raises(ValueError, match="never overwritten"):
        evaluate(video, path, out)
    image = package / manifest["frames"][0]["image"]
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Image hash mismatch"):
        evaluate(video, path, tmp_path / "second.json")


def test_prepare_and_evaluation_never_replace_frozen_package(tmp_path):
    package = tmp_path / "package"
    package.mkdir()
    (package / "labels.json").write_text("original")
    with pytest.raises(ValueError, match="never overwritten"):
        prepare(tmp_path / "missing.h264", package)
    assert (package / "labels.json").read_text() == "original"
