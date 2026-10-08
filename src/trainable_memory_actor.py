"""Experimental causal control memory; no simulator or physical transport.

Unlike recurrent_memory_actor, this profile permits flight-loss gradients through
the GRU. The former position decoder is discarded: this is neural control memory,
not a calibrated position estimator. Whole-episode sequence fitting starts at
zero state and never treats cached hidden features as current training inputs.
"""
from copy import deepcopy
import hashlib
from io import BytesIO
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.recurrent_memory_actor import (
    RecurrentMemoryActor, RecurrentMemoryPolicy, SOURCE_PROFILE, ESTIMATOR_PROFILE,
    checkpoint_manifest as frozen_manifest, _require_precision)

PROFILE = 'trainable-control-memory-residual-v1'


def checkpoint_manifest():
    result = frozen_manifest()
    result.update(format='trainable-control-memory-residual', profile=PROFILE,
                  memory_trainable=True,
                  estimator_role='initialization only; no physical estimate is produced',
                  input_layout=['original_public85_unchanged', 'trainable_control_memory64', 'valid_history1'],
                  normalization='frozen pretrained FIT input normalization; public85 unchanged',
                  residual='valid * W @ causal_control_memory64',
                  training_contract='complete public85 episodes from zero; full recurrent backpropagation',
                  frozen_log_std=True)
    return result


class TrainableMemoryActor(RecurrentMemoryActor):
    def __init__(self, source_actor, pretrained_gru, normalizers):
        super().__init__(source_actor)
        self.memory = deepcopy(pretrained_gru).to(self.device).requires_grad_(True)
        for name in ('x_mean', 'x_scale'):
            tensor = torch.tensor(normalizers[name].copy(), device=self.device)
            self.register_buffer(name, tensor)
        self._frozen_normalizers = {name: getattr(self, name).detach().cpu().clone()
                                    for name in ('x_mean', 'x_scale')}

    def assert_source_unchanged(self):
        super().assert_source_unchanged()
        for name, expected in self._frozen_normalizers.items():
            if not torch.equal(getattr(self, name).detach().cpu(), expected):
                raise ValueError('Frozen input normalizer changed: '+name)

    def cell(self, public, previous):
        """Explicit single-frame cell matches the reviewed streaming arithmetic."""
        normalized = (public-self.x_mean)/self.x_scale
        return self.normalized_cell(normalized, previous)

    def normalized_cell(self, normalized, previous):
        g = self.memory
        ir, iz, inn = nn.functional.linear(normalized, g.weight_ih_l0, g.bias_ih_l0).chunk(3, -1)
        hr, hz, hn = nn.functional.linear(previous, g.weight_hh_l0, g.bias_hh_l0).chunk(3, -1)
        reset, update = torch.sigmoid(ir+hr), torch.sigmoid(iz+hz)
        candidate = torch.tanh(inn+reset*hn)
        return candidate+update*(previous-candidate)

    def sequence_features(self, public, lengths):
        """Flatten genuine rows in episode order; ignore padding and reset each row.

        Call for every optimizer proposal and every trust check. Reusing hidden
        features after changing the GRU would silently train/evaluate stale state.
        No current actions, rewards, physical labels or previous hidden state are
        accepted. Backpropagation reaches all preceding observations in an episode.
        """
        _require_precision(self.device)
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError('Autocast is outside the strict float32 contract')
        if (not isinstance(public, torch.Tensor) or public.dtype != torch.float32
                or public.ndim != 3 or public.shape[2] != 85
                or public.device != self.device or not 1 <= public.shape[1] <= 600
                or public.shape[0] == 0):
            raise ValueError('Float32 (episodes, steps<=600, public85) required')
        if (not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int64
                or lengths.ndim != 1 or len(lengths) != public.shape[0]
                or not bool(((lengths > 0) & (lengths <= public.shape[1])).all())):
            raise ValueError('Integer whole-episode lengths required')
        steps = torch.arange(public.shape[1], device=self.device)[None, :]
        valid = steps < lengths.to(self.device)[:, None]
        if not bool(torch.isfinite(public[valid]).all()):
            raise ValueError('Nonfinite actual public observation')
        normalized = torch.where(valid[:, :, None], (public-self.x_mean)/self.x_scale, 0.)
        if not bool(torch.isfinite(normalized).all()):
            raise ValueError('Nonfinite normalized input')
        # Use the same explicit cell as streaming. The fused CUDA sequence kernel
        # exceeded our fixed 1e-6 stream-parity bound despite TF32 being disabled.
        # Keep full autograd links; never detach inside the episode.
        state = torch.zeros((public.shape[0],64),dtype=torch.float32,device=self.device)
        states = []
        for t in range(public.shape[1]):
            state = self.normalized_cell(normalized[:,t,:],state)
            states.append(state)
        hidden = torch.stack(states,dim=1)
        history_valid = (steps >= 2).expand_as(valid)
        hidden = torch.where(history_valid[:, :, None], hidden, 0.)
        features = torch.cat((public[valid], hidden[valid],
                              history_valid[valid][:, None].float()), dim=1)
        if not bool(torch.isfinite(features).all()):
            raise ValueError('Nonfinite recurrent features')
        return features

    def sequence_distribution(self, public, lengths):
        return self.get_action_dist_params(self.sequence_features(public, lengths))


class TrainableMemoryPolicy:
    """Explicit stateful inference plus safe checkpointing of head AND memory.

    One instance belongs to one caller. Reset at every real episode boundary.
    Optimizer work uses sequence_features and never mutates the live stream.
    """
    profile = PROFILE

    def __init__(self, source_checkpoint, estimator_checkpoint, *, device):
        initial = RecurrentMemoryPolicy(source_checkpoint, estimator_checkpoint, device=device)
        self.actor = TrainableMemoryActor(initial.actor.base_actor,
            initial.ring_position._model.gru, initial.ring_position.normalizers)
        self.device = self.actor.device
        self.reset()

    def reset(self):
        self._hidden = torch.zeros((1, 64), dtype=torch.float32, device=self.device)
        self.frame_count = 0

    @torch.no_grad()
    def observe(self, public85):
        _require_precision(self.device)
        if torch.is_autocast_enabled(self.device.type):
            raise ValueError('Autocast is outside the strict float32 contract')
        if (not isinstance(public85, np.ndarray) or public85.dtype != np.float32
                or public85.shape != (85,) or not np.isfinite(public85).all()):
            raise ValueError('One finite float32 public85 frame required')
        row = torch.from_numpy(public85.copy()).to(self.device)[None]
        hidden = self.actor.cell(row, self._hidden)
        if not bool(torch.isfinite(hidden).all()):
            raise ValueError('Nonfinite memory; live state not advanced')
        features = np.empty(150, np.float32)
        features[:85] = public85
        features[85:149] = hidden[0].cpu().numpy() if self.frame_count >= 2 else 0.
        features[149] = float(self.frame_count >= 2)
        self._hidden = hidden.detach().clone()
        self.frame_count += 1
        return features

    # Identical normalized-command conversion; this consumes explicit cached
    # features and never steps recurrence or issues any physical command.
    predict = RecurrentMemoryPolicy.predict

    def save(self, path):
        self.actor.assert_source_unchanged()
        arrays = {'residual_weight': self.actor.residual_weight.detach().cpu().numpy()}
        arrays.update({'memory.'+k: v.detach().cpu().numpy()
                       for k,v in self.actor.memory.state_dict().items()})
        if any(not np.isfinite(v).all() for v in arrays.values()):
            raise ValueError('Nonfinite candidate weights')
        arrays['manifest'] = np.frombuffer(json.dumps(checkpoint_manifest(),sort_keys=True).encode(),np.uint8)
        with Path(path).open('xb') as f:
            np.savez_compressed(f, **arrays)
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    @classmethod
    def load(cls, checkpoint, source_checkpoint, estimator_checkpoint, *, device, sha256):
        payload = Path(checkpoint).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha256:
            raise ValueError('Candidate checksum mismatch')
        policy = cls(source_checkpoint, estimator_checkpoint, device=device)
        templates = {'residual_weight': policy.actor.residual_weight}
        templates.update({'memory.'+k: v for k,v in policy.actor.memory.state_dict().items()})
        with np.load(BytesIO(payload),allow_pickle=False) as saved:
            if set(saved.files) != set(templates)|{'manifest'}:
                raise ValueError('Exact head, memory and manifest keys required')
            metadata = saved['manifest']
            if (metadata.dtype != np.uint8 or metadata.ndim != 1
                    or json.loads(metadata.tobytes()) != checkpoint_manifest()):
                raise ValueError('Candidate profile/initialization identity mismatch')
            arrays = {}
            for key, template in templates.items():
                value = saved[key]
                if (value.dtype != np.float32 or value.shape != tuple(template.shape)
                        or not np.isfinite(value).all()):
                    raise ValueError('Invalid candidate tensor: '+key)
                arrays[key] = torch.from_numpy(value.copy()).to(policy.device)
        with torch.no_grad():
            policy.actor.residual_weight.copy_(arrays.pop('residual_weight'))
            policy.actor.memory.load_state_dict({k.removeprefix('memory.'): v for k,v in arrays.items()})
        policy.actor.assert_source_unchanged()
        return policy
