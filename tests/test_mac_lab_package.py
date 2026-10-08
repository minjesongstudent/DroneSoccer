"""Artifact integrity checks; calibration fixtures are explicitly synthetic."""
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

from tools.build_mac_lab_preview import build_package


def test_default_package_records_defaults_and_contains_validation(tmp_path):
    out = tmp_path / 'preview-defaults'
    archive = build_package('lags60-scratch', out)
    with ZipFile(archive) as z:
        members = z.namelist()
        assert not any('/sessions/' in member or member.endswith('/calib.npz') for member in members)
        manifest = json.loads(z.read(out.name + '/preview-manifest.json'))
        assert manifest['camera_calibration']['status'] == 'defaults'
        assert manifest['camera_calibration']['bundled_file'] is None
        assert out.name + '/src/camera_calibration.py' in members
        assert out.name + '/test_calibration.py' in members
        assert out.name + '/CALIBRATION.md' in members
        assert b'test_calibration.py' in z.read(out.name + '/1-Setup.command')
        checksums = json.loads(z.read(out.name + '/PACKAGE-SHA256.json'))
        for name, digest in checksums.items():
            assert hashlib.sha256(z.read(out.name + '/' + name)).hexdigest() == digest


def test_supplied_calibration_is_bundled_exactly_and_identified(tmp_path):
    source = tmp_path / 'synthetic-test-only.npz'
    np.savez(source, K=[[800., 0, 480], [0, 810, 360], [0, 0, 1]],
             dist=np.zeros(5), size=[960, 720], rms=.2)
    out = tmp_path / 'preview-calibration'
    archive = build_package('seenflag-s1-level1', out, source)
    with ZipFile(archive) as z:
        payload = z.read(out.name + '/calib.npz')
        assert payload == source.read_bytes()
        manifest = json.loads(z.read(out.name + '/preview-manifest.json'))
        calibration = manifest['camera_calibration']
        assert calibration['status'] == 'calibration_file'
        assert calibration['bundled_file'] == 'calib.npz'
        assert calibration['source_file'] == source.name
        assert calibration['sha256'] == hashlib.sha256(payload).hexdigest()
        assert calibration['size'] == [960, 720]
        assert calibration['rms_px'] == .2
        assert calibration['physical_camera_match_verified'] is False
        assert manifest['policy_inputs'] == 85


def test_missing_or_invalid_calibration_cannot_create_a_package(tmp_path):
    source = tmp_path / 'bad.npz'
    out = tmp_path / 'rejected'
    with pytest.raises(FileNotFoundError):
        build_package('lags60-scratch', out, source)
    source.write_bytes(b'invalid calibration')
    with pytest.raises(ValueError, match='Invalid camera calibration'):
        build_package('lags60-scratch', out, source)
    assert not out.exists()
    assert not out.with_suffix('.zip').exists()


@pytest.mark.parametrize('kind', ['directory', 'zip'])
def test_existing_output_is_preserved(tmp_path, kind):
    out = tmp_path / 'existing'
    if kind == 'directory':
        out.mkdir()
        preserved = out / 'user-recording.txt'
    else:
        preserved = out.with_suffix('.zip')
    preserved.write_bytes(b'preserve this existing package or recording')
    with pytest.raises(FileExistsError, match='choose a new --out'):
        build_package('lags60-scratch', out)
    assert preserved.read_bytes() == b'preserve this existing package or recording'
