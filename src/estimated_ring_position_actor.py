"""Experimental public85 + estimated-ring-position residual actor, with no transport.

This 89-input layout is explicitly distinct from the four-command 89-input
profile. The frozen source actor sees its unchanged original85; only a small
bilinear mean correction is trainable. Recurrence belongs to RecurrentRingPosition
and advances only through observe(), never through predict() or actor updates.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn
from stable_baselines3 import SAC
from stable_baselines3.common.distributions import SquashedDiagGaussianDistribution

from src.policy_runtime import SPECS
from src.recurrent_ring_position import RING_POSITION_SPECS, RingPositionEstimate, RecurrentRingPosition


PROFILE = 'estimated-ring-position-residual-v1'
SOURCE_PROFILE = 'seenflag-s1-level1'
ESTIMATOR_PROFILE = 'gru64-ring-position-20261008'
LAYOUT = ('original_public85_unchanged', 'standardized_ring_relative_position3',
          'valid_history1')


def checkpoint_manifest():
    """Exact semantic identity; an input width alone never selects this actor."""
    return dict(format='estimated-ring-position-residual-head', version=1, profile=PROFILE,
                input_layout=list(LAYOUT), input_width=89,
                source_profile=SOURCE_PROFILE, source_sha256=SPECS[SOURCE_PROFILE].sha256,
                estimator_profile=ESTIMATOR_PROFILE,
                estimator_output_units='millimetres_relative_to_current_body',
                estimator_sha256=RING_POSITION_SPECS[ESTIMATOR_PROFILE].checkpoint_sha256,
                inference_contract='strict_float32',
                cuda_precision=dict(cudnn_allow_tf32=False, matmul_allow_tf32=False),
                normalization='frozen estimator FIT-y-standardized output; public85 unchanged',
                residual='valid * W @ row_major_outer([1, public85], estimated3)',
                weight_shape=[4, 258], frozen_log_std=True)


def _require_precision(device):
    if device.type == 'cuda' and (torch.backends.cudnn.allow_tf32
                                 or torch.backends.cuda.matmul.allow_tf32):
        raise ValueError('strict_float32 requires cudnn.allow_tf32=False and '
                         'cuda.matmul.allow_tf32=False; configure once at process startup')


def build_features(estimate: RingPositionEstimate) -> np.ndarray:
    """Concatenate adapter output once, explicitly masking unsupervised startup.

    The estimate's public85 copy is authoritative, preventing concatenation with
    a different frame. No position, heading, current action or target is accepted.
    """
    if not isinstance(estimate, RingPositionEstimate):
        raise TypeError('A RingPositionEstimate from the named public85 adapter is required')
    for value, shape in ((estimate.public85, (85,)),
                         (estimate.standardized_position, (3,)),
                         (estimate.position_mm, (3,))):
        if (not isinstance(value, np.ndarray) or value.dtype != np.float32
                or value.shape != shape or not np.isfinite(value).all()):
            raise ValueError('Finite float32 public85 and three-axis estimates are required')
    if (type(estimate.frame_count) is not int or estimate.frame_count < 1
            or type(estimate.valid_history) is not bool
            or estimate.valid_history != (estimate.frame_count >= 3)):
        raise ValueError('History validity must match the third-or-later frame contract')
    features = np.empty(89, np.float32)
    features[:85] = estimate.public85
    features[85:88] = estimate.standardized_position if estimate.valid_history else 0.
    features[88] = float(estimate.valid_history)
    return features


class EstimatedRingPositionActor(nn.Module):
    """Frozen SAC base with 1,032 trainable bilinear mean-correction weights.

    All tensor methods consume already constructed (batch, 89) features and are
    stateless. They do not advance GRU state. Existing on-policy log-probability
    helpers can use get_action_dist_params() and action_dist unchanged.
    """
    def __init__(self, source_actor):
        super().__init__()
        if (source_actor.observation_space.shape != (85,)
                or source_actor.action_space.shape != (4,)
                or source_actor.use_sde
                or not isinstance(source_actor.action_dist, SquashedDiagGaussianDistribution)):
            raise ValueError('The original public85 four-action non-SDE SAC actor is required')
        # A used distribution caches non-leaf sampled tensors; copying those is
        # invalid and unnecessary. Copy parameters/modules with a fresh identical
        # distribution, without modifying the caller's actor or cached sample.
        fresh_distribution = SquashedDiagGaussianDistribution(4, source_actor.action_dist.epsilon)
        self.base_actor = deepcopy(source_actor, {id(source_actor.action_dist): fresh_distribution})
        self.base_actor.eval().requires_grad_(False)
        device = next(self.base_actor.parameters()).device
        self.residual_weight = nn.Parameter(torch.zeros((4, 258), dtype=torch.float32, device=device))
        self.action_dist = SquashedDiagGaussianDistribution(4, source_actor.action_dist.epsilon)
        self._source_weights = {name: value.detach().cpu().clone()
                                for name, value in self.base_actor.state_dict().items()}

    @property
    def device(self):
        return self.residual_weight.device

    def train(self, mode=True):
        super().train(mode)
        self.base_actor.eval()
        return self

    def _validate(self, observations):
        _require_precision(self.device)
        if (not isinstance(observations, torch.Tensor) or observations.dtype != torch.float32
                or observations.ndim != 2 or observations.shape[1] != 89
                or observations.shape[0] == 0 or observations.device != self.device
                or not bool(torch.isfinite(observations).all())):
            raise ValueError('Finite float32 (batch, 89) features on the actor device required')
        valid = observations[:, 88]
        if not bool(((valid == 0) | (valid == 1)).all()):
            raise ValueError('History validity must be binary')
        if not bool((observations[valid == 0, 85:88] == 0).all()):
            raise ValueError('Invalid-history motion features must be explicitly zero-masked')

    def get_action_dist_params(self, observations):
        self._validate(observations)
        public = observations[:, :85].contiguous()
        # Preserve the original matrix dimensions and action-distribution path.
        with torch.no_grad():
            mean, log_std, kwargs = self.base_actor.get_action_dist_params(public)
        context = torch.cat((torch.ones_like(public[:, :1]), public), dim=1)
        bilinear = (context[:, :, None] * observations[:, None, 85:88]).flatten(start_dim=1)
        delta = nn.functional.linear(bilinear, self.residual_weight)
        mean = mean + observations[:, 88:89] * delta
        if not bool(torch.isfinite(mean).all()):
            raise ValueError('Non-finite residual action mean')
        return mean, log_std, kwargs

    def forward(self, observations, deterministic=False):
        mean, log_std, kwargs = self.get_action_dist_params(observations)
        return self.action_dist.actions_from_params(mean, log_std,
                                                    deterministic=deterministic, **kwargs)

    def action_log_prob(self, observations):
        mean, log_std, kwargs = self.get_action_dist_params(observations)
        return self.action_dist.log_prob_from_params(mean, log_std, **kwargs)

    def log_prob(self, observations, actions):
        mean, log_std, kwargs = self.get_action_dist_params(observations)
        if (not isinstance(actions, torch.Tensor) or actions.shape != mean.shape
                or actions.dtype != torch.float32 or actions.device != self.device
                or not bool(torch.isfinite(actions).all()) or not bool((actions.abs() <= 1).all())):
            raise ValueError('Finite normalized four-action rows matching the batch required')
        distribution = self.action_dist.proba_distribution(mean, log_std, **kwargs)
        return distribution.log_prob(actions.clamp(-1 + 1e-6, 1 - 1e-6))

    def assert_source_unchanged(self):
        for name, value in self.base_actor.state_dict().items():
            if not torch.equal(value.detach().cpu(), self._source_weights[name]):
                raise ValueError(f'Frozen source actor changed: {name}')
        if any(parameter.requires_grad for parameter in self.base_actor.parameters()):
            raise ValueError('Source actor parameters must remain frozen')


class EstimatedRingPositionPolicy:
    """Hash-bound source + frozen estimator + explicit experimental actor.

    Call reset() at a true episode boundary, then observe(public85) once per new
    observation. Cache the returned features for prediction and later fitting.
    predict(features89) never changes recurrence. No drone or socket interface is
    provided; this is an experimental model component, not a flight release.
    """
    profile = PROFILE

    def __init__(self, source_checkpoint, estimator_checkpoint, *, device):
        self.ring_position = RecurrentRingPosition(estimator_checkpoint, spec=ESTIMATOR_PROFILE,
                                      profile=SOURCE_PROFILE, device=device)
        payload = Path(source_checkpoint).read_bytes()
        if hashlib.sha256(payload).hexdigest() != SPECS[SOURCE_PROFILE].sha256:
            raise ValueError('Original source policy checksum does not match its profile')
        # Both deserializations are restricted to exact registered historical bytes.
        source = SAC.load(BytesIO(payload), device=device)
        if (not np.array_equal(source.action_space.low, -np.ones(4))
                or not np.array_equal(source.action_space.high, np.ones(4))):
            raise ValueError('Expected normalized original [-1,1] action space')
        self.actor = EstimatedRingPositionActor(source.actor)
        self.device = self.actor.device

    def reset(self):
        self.ring_position.reset()

    def observe(self, public85):
        return build_features(self.ring_position.step(public85))

    @torch.no_grad()
    def predict(self, features, deterministic=True):
        if (not isinstance(features, np.ndarray) or features.dtype != np.float32
                or features.ndim not in (1, 2)):
            raise ValueError('Explicit float32 estimated-ring-position 89-input features required')
        single = features.ndim == 1
        batch = features[None] if single else features
        self.actor.eval()
        actions = self.actor(torch.from_numpy(batch.copy()).to(self.device),
                             deterministic=deterministic).cpu().numpy()
        # Match SB3 BasePolicy.predict even for [-1,1]: its arithmetic unscaling
        # can change float32 ulps, so returning raw tanh would break exact parity.
        actions = self.actor.base_actor.unscale_action(actions)
        if not np.isfinite(actions).all():
            raise ValueError('Non-finite learned action')
        return (actions[0] if single else actions), None

    def save(self, path):
        """Write only a safe array head and explicit frozen-artifact identities."""
        self.actor.assert_source_unchanged()
        weights = self.actor.residual_weight.detach().cpu().numpy()
        if not np.isfinite(weights).all():
            raise ValueError('Cannot save non-finite residual weights')
        metadata = np.frombuffer(json.dumps(checkpoint_manifest(), sort_keys=True).encode(), dtype=np.uint8)
        with Path(path).open('xb') as stream:
            np.savez_compressed(stream, manifest=metadata, residual_weight=weights)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, checkpoint, source_checkpoint, estimator_checkpoint, *, device, sha256):
        """Restore safe arrays only, requiring the caller's saved candidate hash."""
        payload = Path(checkpoint).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('Candidate checkpoint checksum does not match supplied identity')
        try:
            with np.load(BytesIO(payload), allow_pickle=False) as archive:
                if set(archive.files) != {'manifest', 'residual_weight'}:
                    raise ValueError('Exactly manifest and residual_weight must be present')
                metadata, weights = archive['manifest'], archive['residual_weight']
                if metadata.dtype != np.uint8 or metadata.ndim != 1:
                    raise ValueError('Invalid manifest encoding')
                if json.loads(metadata.tobytes()) != checkpoint_manifest():
                    raise ValueError('Candidate profile, layout or frozen-artifact identity mismatch')
                if weights.dtype != np.float32 or weights.shape != (4, 258) or not np.isfinite(weights).all():
                    raise ValueError('Finite float32 residual weights with shape (4,258) required')
                weights = weights.copy()
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError('Invalid estimated-ring-position candidate archive') from exc
        policy = cls(source_checkpoint, estimator_checkpoint, device=device)
        with torch.no_grad():
            policy.actor.residual_weight.copy_(torch.from_numpy(weights).to(policy.device))
        policy.actor.assert_source_unchanged()
        return policy
