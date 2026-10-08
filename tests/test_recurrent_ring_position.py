"""Offline adapter checks; synthetic checkpoints are test fixtures, never actors."""
import hashlib
import inspect
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from src.camera_history import CameraHistory
from src import recurrent_ring_position as runtime
from src.recurrent_ring_position import RingPositionSpec, RecurrentRingPosition


PROFILE = 'seenflag-s1-level1'
REAL_SPEC = 'gru64-ring-position-20261008'
REAL_CHECKPOINT = Path('D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt')


class BatchReference(nn.Module):
    """Independent copy of the frozen experiment's packed whole-episode forward."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(85, 64, batch_first=True)
        self.output = nn.Linear(64, 3)

    def forward(self, observations):
        lengths = torch.full((len(observations),), observations.shape[1], dtype=torch.long)
        packed = pack_padded_sequence(observations, lengths, batch_first=True, enforce_sorted=False)
        hidden, _ = self.gru(packed)
        hidden, _ = pad_packed_sequence(hidden, batch_first=True,
                                        total_length=observations.shape[1])
        return self.output(hidden)


def public_sequence(length=17, seed=42):
    """Synthetic public observations with the explicit documented lag/action layout."""
    rng = np.random.default_rng(seed)
    history = CameraHistory((24, 20, 16, 12, 8, 4, 2, 1, 0))
    observations = []
    for index in range(length):
        detection = np.array([.1*np.sin(index), .2, .4, .8, 0., .2, 1., index % 2], np.float32)
        if index % 5 == 0:
            detection[:] = 0
        previous_issued = np.zeros(4, np.float32) if index == 0 else rng.uniform(-.2, .2, 4).astype(np.float32)
        observations.append(history.update(detection, previous_issued))
    return np.stack(observations)


@pytest.fixture
def synthetic_checkpoint(tmp_path, monkeypatch):
    counter = [0]
    def make(mutate=None):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(371)
            reference = BatchReference()
        payload = dict(model=reference.state_dict(), updates=0, normalizers={
            'x_mean': np.linspace(-.2, .3, 85, dtype=np.float32),
            'x_scale': np.linspace(.2, 1., 85, dtype=np.float32),
            'y_mean': np.array([1., -2., 3.], np.float32),
            'y_scale': np.array([2., 4., 8.], np.float32),
        })
        if mutate:
            mutate(payload)
        counter[0] += 1
        path = tmp_path / f'synthetic-fixture-{counter[0]}.pt'
        torch.save(payload, path)
        name = f'test-only-gru64-{counter[0]}'
        spec = RingPositionSpec(name, hashlib.sha256(path.read_bytes()).hexdigest())
        monkeypatch.setattr(runtime, 'RING_POSITION_SPECS', MappingProxyType({
            **runtime.RING_POSITION_SPECS, name: spec,
        }))
        return path, name, payload
    return make


def load_fixture(synthetic_checkpoint, mutate=None, **kwargs):
    path, spec, payload = synthetic_checkpoint(mutate)
    return RecurrentRingPosition(path, spec=spec, profile=PROFILE, device=kwargs.pop('device', 'cpu'),
                           **kwargs), path, payload


def reference_predictions(payload, observations):
    with torch.random.fork_rng(devices=[]):
        reference = BatchReference().eval()
    reference.load_state_dict(payload['model'])
    normalizers = payload['normalizers']
    normalized = (observations-normalizers['x_mean'])/normalizers['x_scale']
    with torch.inference_mode():
        standardized = reference(torch.from_numpy(normalized[None]))[0].numpy()
    return standardized, standardized*normalizers['y_scale']+normalizers['y_mean']


def test_streaming_matches_packed_batch_with_frozen_normalization(synthetic_checkpoint):
    estimator, _, payload = load_fixture(synthetic_checkpoint)
    observations = public_sequence()
    standardized, expected = reference_predictions(payload, observations)
    result = [estimator.step(row) for row in observations]
    # Same causal computation; batched vs one-row GEMM may differ by float32 roundoff.
    np.testing.assert_allclose(np.stack([row.position_mm for row in result]), expected,
                               rtol=2e-6, atol=5e-6)
    np.testing.assert_allclose(np.stack([row.standardized_position for row in result]),
                               standardized, rtol=2e-6, atol=5e-7)
    assert [row.valid_history for row in result[:3]] == [False, False, True]
    assert [row.frame_count for row in result] == list(range(1, len(observations)+1))


def test_reset_zeros_state_and_repeats_episode_independently(synthetic_checkpoint):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    observations = public_sequence()
    first = [estimator.step(row).position_mm for row in observations]
    assert estimator.hidden_state.any()
    estimator.reset()
    assert not estimator.hidden_state.any() and estimator.frame_count == 0
    second = [estimator.step(row) for row in observations]
    np.testing.assert_array_equal(np.stack(first), np.stack([row.position_mm for row in second]))
    assert not second[0].valid_history and second[2].valid_history


def test_instances_have_independent_state_and_future_suffix_cannot_change_prefix(synthetic_checkpoint):
    path, spec, _ = synthetic_checkpoint()
    left = RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cpu')
    right = RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cpu')
    first, second = public_sequence(14), public_sequence(14)
    second[8:] = public_sequence(6, seed=93)
    left_predictions, right_predictions = [], []
    for index in range(14):
        left_predictions.append(left.step(first[index]).position_mm)
        right_predictions.append(right.step(second[index]).position_mm)
    np.testing.assert_array_equal(left_predictions[:8], right_predictions[:8])
    assert not np.array_equal(left_predictions[8:], right_predictions[8:])
    state = right.hidden_state.copy()
    left.reset()
    np.testing.assert_array_equal(right.hidden_state, state)


def test_public85_exactly_preserved_separately_from_estimate(synthetic_checkpoint):
    estimator, _, payload = load_fixture(synthetic_checkpoint)
    observation = public_sequence(1)[0]
    original = observation.copy()
    result = estimator.step(observation)
    np.testing.assert_array_equal(result.public85, original)
    assert result.public85.tobytes() == original.tobytes()
    assert result.public85.shape == (85,) and result.position_mm.shape == (3,)
    assert result.standardized_position.shape == (3,)
    np.testing.assert_array_equal(result.position_mm,
                                  result.standardized_position*payload['normalizers']['y_scale']
                                  +payload['normalizers']['y_mean'])
    observation[:] = 999
    np.testing.assert_array_equal(result.public85, original)
    for value in (result.public85, result.position_mm, result.standardized_position):
        assert not value.flags.writeable
    assert tuple(inspect.signature(RecurrentRingPosition.step).parameters) == ('self', 'public85')


def test_first_two_predictions_are_returned_but_explicitly_unqualified(synthetic_checkpoint):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    rows = [estimator.step(row) for row in public_sequence(3)]
    assert [row.valid_history for row in rows] == [False, False, True]
    assert all(np.isfinite(row.position_mm).all() for row in rows)
    assert rows[0].position_mm.any()  # no fabricated zero-output fallback


@pytest.mark.parametrize('shape', [(84,), (88,), (89,), (109,), (1, 85), (0,)])
def test_wrong_shape_rejected_without_state_advance(synthetic_checkpoint, shape):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    with pytest.raises(ValueError, match='shape'):
        estimator.step(np.zeros(shape, np.float32))
    assert estimator.frame_count == 0 and not estimator.hidden_state.any()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -float('inf'), np.finfo(np.float32).max])
def test_nonfinite_or_normalization_overflow_has_no_fallback(synthetic_checkpoint, bad):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    estimator.step(public_sequence(1)[0])
    previous = estimator.hidden_state.copy()
    observation = np.zeros(85, np.float32)
    observation[0] = bad
    with pytest.raises(ValueError, match='finite'):
        estimator.step(observation)
    assert estimator.frame_count == 1
    np.testing.assert_array_equal(estimator.hidden_state, previous)


@pytest.mark.parametrize('value', [np.zeros(85), [0.] * 85, np.zeros(85, np.int32)])
def test_no_implicit_precision_conversion(synthetic_checkpoint, value):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    with pytest.raises(TypeError, match='float32'):
        estimator.step(value)


def test_unknown_spec_wrong_profile_and_hash_fail_before_deserialization(synthetic_checkpoint, monkeypatch):
    path, spec, _ = synthetic_checkpoint()
    def forbidden(*args, **kwargs):
        pytest.fail('Rejected specs/profiles/hashes must not be deserialized')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ValueError, match='spec'):
        RecurrentRingPosition(path, spec='guess-from-dimension', profile=PROFILE, device='cpu')
    with pytest.raises(ValueError, match='profile'):
        RecurrentRingPosition(path, spec=spec, profile='lags60-scratch', device='cpu')
    path.write_bytes(path.read_bytes()+b'changed test fixture')
    with pytest.raises(ValueError, match='checksum'):
        RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cpu')


@pytest.mark.parametrize('change', ['shape', 'nan', 'zero_scale', 'negative_scale', 'dtype', 'missing'])
def test_bad_normalizers_fail_instead_of_refitting(synthetic_checkpoint, change):
    def mutate(payload):
        norms = payload['normalizers']
        if change == 'shape': norms['x_mean'] = np.zeros(84, np.float32)
        if change == 'nan': norms['y_mean'][1] = np.nan
        if change == 'zero_scale': norms['x_scale'][0] = 0
        if change == 'negative_scale': norms['y_scale'][0] = -1
        if change == 'dtype': norms['x_mean'] = norms['x_mean'].astype(np.float64)
        if change == 'missing': del norms['y_scale']
    path, spec, _ = synthetic_checkpoint(mutate)
    with pytest.raises(ValueError, match='normalizer'):
        RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cpu')


@pytest.mark.parametrize('change', ['shape', 'nan', 'extra'])
def test_wrong_architecture_or_weights_rejected(synthetic_checkpoint, change):
    def mutate(payload):
        if change == 'shape': payload['model']['output.weight'] = torch.zeros(4, 64)
        if change == 'nan': payload['model']['output.bias'][0] = float('nan')
        if change == 'extra': payload['model']['other.weight'] = torch.zeros(1)
    path, spec, _ = synthetic_checkpoint(mutate)
    with pytest.raises(ValueError, match='architecture|weights'):
        RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cpu')


@pytest.mark.parametrize('bad_state', [False, True])
def test_nonfinite_prediction_or_state_does_not_commit(synthetic_checkpoint, monkeypatch, bad_state):
    estimator, _, _ = load_fixture(synthetic_checkpoint)
    estimator.step(public_sequence(1)[0])
    previous = estimator.hidden_state.copy()
    def failed(observation, state):
        prediction, next_state = torch.ones(1, 1, 3), state.clone()
        (next_state if bad_state else prediction).fill_(float('nan'))
        return prediction, next_state
    monkeypatch.setattr(estimator._model, 'forward', failed)
    with pytest.raises(ValueError, match='prediction or hidden state'):
        estimator.step(public_sequence(1)[0])
    assert estimator.frame_count == 1
    np.testing.assert_array_equal(estimator.hidden_state, previous)


def test_normalizers_are_exact_readonly_copies_and_checkpoint_never_mutates(synthetic_checkpoint):
    estimator, path, payload = load_fixture(synthetic_checkpoint)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    weights = {key: value.clone() for key, value in estimator._model.state_dict().items()}
    for key, original in payload['normalizers'].items():
        exposed = estimator.normalizers[key]
        np.testing.assert_array_equal(exposed, original)
        assert exposed.dtype == original.dtype and not exposed.flags.writeable
        exposed.setflags(write=True); exposed[:] = 999  # modifying a copy cannot change runtime
        np.testing.assert_array_equal(estimator.normalizers[key], original)
    with pytest.raises(TypeError):
        estimator.normalizers['x_mean'] = np.zeros(85)
    assert estimator.normalization_metadata['output_axes'] == ('forward', 'right', 'up')
    assert 'FIT episodes only' in estimator.normalization_metadata['source']
    assert estimator.spec.inference_contract == 'strict_float32'
    assert estimator.normalization_metadata['inference_contract'] == 'strict_float32'
    for row in public_sequence(): estimator.step(row)
    estimator.reset()
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not estimator._model.training
    assert not any(parameter.requires_grad for parameter in estimator._model.parameters())
    for key, value in estimator._model.state_dict().items():
        assert torch.equal(weights[key], value)


@pytest.mark.parametrize('device', ['auto', 'cuda', 'mps', '', None])
def test_device_must_be_explicit(synthetic_checkpoint, device):
    path, spec, _ = synthetic_checkpoint()
    with pytest.raises(ValueError, match='explicitly'):
        RecurrentRingPosition(path, spec=spec, profile=PROFILE, device=device)


def test_unavailable_cuda_never_falls_back(synthetic_checkpoint, monkeypatch):
    path, spec, _ = synthetic_checkpoint()
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(RuntimeError, match='no implicit CPU fallback'):
        RecurrentRingPosition(path, spec=spec, profile=PROFILE, device='cuda:0')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='Optional explicit-CUDA fixture check')
def test_explicit_cuda_streaming_fixture(synthetic_checkpoint, monkeypatch):
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    estimator, _, payload = load_fixture(synthetic_checkpoint, device='cuda:0')
    observations = public_sequence(5)
    _, expected = reference_predictions(payload, observations)
    actual = np.stack([estimator.step(row).position_mm for row in observations])
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=5e-6)
    assert all(parameter.device.type == 'cuda' for parameter in estimator._model.parameters())
    assert not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA precision guard requires CUDA')
@pytest.mark.parametrize('cudnn_tf32,matmul_tf32', [(True, False), (False, True), (True, True)])
def test_cuda_precision_flags_rejected_before_state_advance_without_mutation(
        synthetic_checkpoint, monkeypatch, cudnn_tf32, matmul_tf32):
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    estimator, _, _ = load_fixture(synthetic_checkpoint, device='cuda:0')
    estimator.step(public_sequence(1)[0])
    previous = estimator.hidden_state.copy()
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', cudnn_tf32)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', matmul_tf32)
    with pytest.raises(RuntimeError, match='strict_float32 CUDA requires'):
        estimator.step(public_sequence(1)[0])
    assert estimator.frame_count == 1
    np.testing.assert_array_equal(estimator.hidden_state, previous)
    assert torch.backends.cudnn.allow_tf32 == cudnn_tf32
    assert torch.backends.cuda.matmul.allow_tf32 == matmul_tf32


@pytest.mark.parametrize('device,dtype', [('cpu', torch.bfloat16), ('cuda:0', torch.float16)])
def test_active_autocast_rejected_before_state_advance(synthetic_checkpoint, monkeypatch, device, dtype):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA autocast guard requires CUDA')
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    estimator, _, _ = load_fixture(synthetic_checkpoint, device=device)
    estimator.step(public_sequence(1)[0])
    previous = estimator.hidden_state.copy()
    with torch.autocast(device_type=estimator.device.type, dtype=dtype):
        with pytest.raises(RuntimeError, match='disabled .* autocast'):
            estimator.step(public_sequence(1)[0])
        assert torch.is_autocast_enabled(estimator.device.type)  # adapter did not change caller context
    assert estimator.frame_count == 1
    np.testing.assert_array_equal(estimator.hidden_state, previous)


@pytest.mark.skipif(not REAL_CHECKPOINT.exists(), reason='Frozen external checkpoint is not installed')
def test_real_frozen_checkpoint_one_offline_packed_batch_parity_check():
    before = hashlib.sha256(REAL_CHECKPOINT.read_bytes()).hexdigest()
    assert before == runtime.RING_POSITION_SPECS[REAL_SPEC].checkpoint_sha256
    payload = torch.load(REAL_CHECKPOINT, map_location='cpu', weights_only=False)
    estimator = RecurrentRingPosition(REAL_CHECKPOINT, spec=REAL_SPEC, profile=PROFILE, device='cpu')
    observations = public_sequence(24, seed=123)
    _, expected = reference_predictions(payload, observations)
    estimates = [estimator.step(row) for row in observations]
    actual = np.stack([value.position_mm for value in estimates])
    max_error_mm = float(np.max(np.abs(actual-expected)))
    np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-5)
    with np.load(REAL_CHECKPOINT.with_name('normalizers.npz'), allow_pickle=False) as saved:
        for key, value in estimator.normalizers.items():
            np.testing.assert_array_equal(value, saved[key])
    assert hashlib.sha256(REAL_CHECKPOINT.read_bytes()).hexdigest() == before
    print(f'Frozen real-checkpoint streaming/batch parity: max absolute difference {max_error_mm:.9g} mm; 24 synthetic public85 rows; CPU only')
