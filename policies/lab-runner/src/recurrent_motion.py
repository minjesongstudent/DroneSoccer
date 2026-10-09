"""Frozen offline, causal motion estimates from the named public85 profile.

The only streaming input is one existing float32 public85 observation, including
its documented *previously issued* action history. There is no current-action,
position, heading, target, network or command interface. Each instance owns one
GRU state; callers submit each actual frame once in order and reset at episode
boundaries. This module does not load an actor or train anything.

Outputs are estimated previous-interval displacement in millimetres, ordered
forward/right/up in the body frame used by the frozen estimator's targets. They
are not velocity, measured motion or exact physical state. The first two rows
initialize recurrence but had no supervised target; their returned predictions
have valid_history=False. True from row three means sufficient history only,
not a guarantee of estimation accuracy. Actor feature concatenation is deferred:
the public85 copy and estimate remain separate, with explicit validity/count.

The inference contract is strict_float32: callers disable autocast, and CUDA
callers configure both cuDNN and matrix-multiply TF32 off once at process startup.
The adapter checks these settings before advancing state and never changes
process-global precision flags. Historical packed TF32 predictions remain a
separate preserved result; they are not the strict-float32 reference.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from io import BytesIO
from pathlib import Path
import re
import threading
from types import MappingProxyType

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class MotionSpec:
    name: str
    checkpoint_sha256: str
    public_profile: str = 'seenflag-s1-level1'
    public_inputs: int = 85
    hidden_size: int = 64
    output_axes: tuple[str, ...] = ('forward', 'right', 'up')
    units: str = 'millimetres_per_previous_interval'
    unsupervised_initial_rows: int = 2
    inference_contract: str = 'strict_float32'


MOTION_SPECS = MappingProxyType({
    'gru64-preaction-20261008': MotionSpec(
        'gru64-preaction-20261008',
        'b48d6b26136a5e2e6cbfcd615a1a819eb801b495fba4cfa38130c0e2da54f554'),
})


@dataclass(frozen=True)
class MotionEstimate:
    """Separate public input and estimated motion; no actor vector is constructed.

    standardized_displacement is the direct network output in the checkpoint's
    FIT target coordinates. displacement_mm uses its unchanged y_scale/y_mean.
    Both are unqualified during the first two rows, as valid_history records.
    """

    public85: np.ndarray
    displacement_mm: np.ndarray
    standardized_displacement: np.ndarray
    frame_count: int
    valid_history: bool


class _FrozenGRU64(nn.Module):
    """State-dict-compatible architecture, independent of the experiment runner."""

    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(85, 64, batch_first=True)
        self.output = nn.Linear(64, 3)

    def forward(self, observations, state):
        features, next_state = self.gru(observations, state)
        return self.output(features), next_state


def _readonly_copy(array):
    result = array.copy()
    result.setflags(write=False)
    return result


def _explicit_device(name):
    if not isinstance(name, str) or (name != 'cpu' and re.fullmatch(r'cuda:[0-9]+', name) is None):
        raise ValueError('Choose device explicitly as cpu or cuda:<index>')
    device = torch.device(name)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no implicit CPU fallback')
        if device.index >= torch.cuda.device_count():
            raise ValueError('Requested CUDA device index does not exist')
    return device


class RecurrentMotion:
    """One hash-bound estimator stream with zero state only on explicit reset.

    Use ``RecurrentMotion(path, spec='gru64-preaction-20261008',
    profile='seenflag-s1-level1', device='cpu')`` then ``step(public85)``.
    A failed input/prediction raises without advancing state or substituting an
    estimate. No input is clipped and no normalizer is fitted or adjusted.
    """

    def __init__(self, checkpoint, *, spec: str, profile: str, device: str):
        if not isinstance(spec, str) or spec not in MOTION_SPECS:
            raise ValueError('Explicit registered motion-estimator spec required')
        selected = MOTION_SPECS[spec]
        if profile != selected.public_profile or profile != 'seenflag-s1-level1':
            raise ValueError('The explicit seenflag-s1-level1 public85 profile is required')
        if (selected.name != spec or selected.public_inputs != 85 or selected.hidden_size != 64
                or selected.output_axes != ('forward', 'right', 'up')
                or selected.units != 'millimetres_per_previous_interval'
                or selected.unsupervised_initial_rows != 2
                or selected.inference_contract != 'strict_float32'):
            raise ValueError('Unsupported motion-estimator architecture or output contract')
        self.spec, self.device = selected, _explicit_device(device)
        payload = Path(checkpoint).read_bytes()
        if hashlib.sha256(payload).hexdigest() != selected.checkpoint_sha256:
            raise ValueError('Motion-estimator checkpoint checksum does not match its spec')
        # The original training checkpoint contains NumPy normalizers/RNG data.
        # Deserialize only the exact registered bytes, after verifying their hash.
        # Read once into BytesIO so a file replacement cannot bypass that binding.
        checkpoint_data = torch.load(BytesIO(payload), map_location='cpu', weights_only=False)
        if not isinstance(checkpoint_data, dict) or not {'model', 'normalizers'} <= checkpoint_data.keys():
            raise ValueError('Checkpoint must contain model weights and frozen normalizers')
        normalizers = checkpoint_data['normalizers']
        expected_shapes = {'x_mean': (85,), 'x_scale': (85,), 'y_mean': (3,), 'y_scale': (3,)}
        if not isinstance(normalizers, dict) or set(normalizers) != set(expected_shapes):
            raise ValueError('Exactly the four frozen FIT normalizers are required')
        self._normalizers = {}
        for name, shape in expected_shapes.items():
            value = normalizers[name]
            if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                    or value.shape != shape or not np.isfinite(value).all()
                    or (name.endswith('scale') and not (value > 0).all())):
                raise ValueError(f'Invalid frozen normalizer: {name}')
            self._normalizers[name] = _readonly_copy(value)
        weights = checkpoint_data['model']
        if (not isinstance(weights, dict) or not weights
                or any(not isinstance(value, torch.Tensor) or value.dtype != torch.float32
                       or not bool(torch.isfinite(value).all()) for value in weights.values())):
            raise ValueError('Finite float32 estimator weights are required')
        # Loading fixed weights should not consume the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            model = _FrozenGRU64()
        try:
            model.load_state_dict(weights, strict=True)
        except RuntimeError as exc:
            raise ValueError('Checkpoint architecture does not match GRU85-to-64-to-3') from exc
        self._model = model.to(self.device).eval().requires_grad_(False)
        self._lock = threading.RLock()
        self.reset()

    @property
    def normalizers(self):
        """Read-only copies of the unchanged FIT coefficients from this checkpoint."""
        return MappingProxyType({name: _readonly_copy(value)
                                 for name, value in self._normalizers.items()})

    @property
    def normalization_metadata(self):
        return MappingProxyType({
            'source': 'frozen checkpoint; FIT episodes only; never fitted at runtime',
            'checkpoint_sha256': self.spec.checkpoint_sha256,
            'input': '(public85 - x_mean) / x_scale, float32',
            'output': 'normalized_prediction * y_scale + y_mean, float32',
            'output_units': self.spec.units,
            'output_axes': self.spec.output_axes,
            'public_profile': self.spec.public_profile,
            'public_lags': (24, 20, 16, 12, 8, 4, 2, 1, 0),
            'inference_contract': self.spec.inference_contract,
            'required_autocast_enabled': False,
            'required_cuda_backend_flags': (('cudnn.allow_tf32', False),
                                            ('cuda.matmul.allow_tf32', False)),
            'precision_configuration': 'caller configures once before use; adapter never mutates global flags',
        })

    @property
    def hidden_state(self):
        """Diagnostic copy of neural state, not a measured physical state."""
        with self._lock:
            return _readonly_copy(self._state.detach().cpu().numpy())

    @property
    def frame_count(self):
        with self._lock:
            return self._frame_count

    def reset(self):
        with self._lock:
            self._state = torch.zeros((1, 1, 64), dtype=torch.float32, device=self.device)
            self._frame_count = 0

    def _require_precision(self):
        if torch.is_autocast_enabled(self.device.type):
            raise RuntimeError(
                f'strict_float32 requires disabled {self.device.type} autocast; '
                'no state was advanced and no precision flag was changed')
        if self.device.type == 'cuda' and (
                torch.backends.cudnn.allow_tf32 or torch.backends.cuda.matmul.allow_tf32):
            raise RuntimeError(
                'strict_float32 CUDA requires torch.backends.cudnn.allow_tf32=False '
                'and torch.backends.cuda.matmul.allow_tf32=False; configure both once '
                'at process startup. No state was advanced and no global flag was changed')

    def step(self, public85: np.ndarray) -> MotionEstimate:
        """Consume exactly one float32 public85 row; never current action or truth.

        Float32 is required so the returned public prefix is bit-for-bit the
        supplied observation rather than an implicit precision conversion.
        """
        if not isinstance(public85, np.ndarray) or public85.dtype != np.float32:
            raise TypeError('A float32 NumPy public85 observation is required')
        if public85.shape != (85,) or not np.isfinite(public85).all():
            raise ValueError('One finite public85 observation with shape (85,) is required')
        observation = public85.copy()
        with self._lock:
            self._require_precision()
            with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
                normalized = ((observation - self._normalizers['x_mean'])
                              / self._normalizers['x_scale'])
            if not np.isfinite(normalized).all():
                raise ValueError('Non-finite normalized public85 input')
            tensor = torch.from_numpy(normalized).reshape(1, 1, 85).to(self.device)
            with torch.inference_mode():
                predicted, next_state = self._model(tensor, self._state)
            if (predicted.shape != (1, 1, 3) or next_state.shape != (1, 1, 64)
                    or not bool(torch.isfinite(predicted).all())
                    or not bool(torch.isfinite(next_state).all())):
                raise ValueError('Invalid recurrent prediction or hidden state')
            # Match the original NumPy float32 denormalization exactly.
            with np.errstate(over='ignore', invalid='ignore'):
                displacement = (predicted[0, 0].cpu().numpy() * self._normalizers['y_scale']
                                + self._normalizers['y_mean'])
            if not np.isfinite(displacement).all():
                raise ValueError('Non-finite displacement estimate')
            self._state = next_state.detach().clone()
            self._frame_count += 1
            return MotionEstimate(_readonly_copy(observation), _readonly_copy(displacement),
                                  _readonly_copy(predicted[0, 0].cpu().numpy()),
                                  self._frame_count,
                                  self._frame_count > self.spec.unsupervised_initial_rows)
