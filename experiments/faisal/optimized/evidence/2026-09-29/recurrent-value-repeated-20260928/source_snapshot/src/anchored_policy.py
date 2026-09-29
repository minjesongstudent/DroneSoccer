"""SAC with prior-policy regularization and task-balanced replay.

This is deliberately not ordinary SAC: a frozen copy of the inherited actor
limits forgetting on observable approach/clearance states. It contributes no
actions or demonstrations to the environment. All environment actions still
come from the one trainable actor. The KL acceptance rule bounds the mean on
the current protected minibatch, not all possible observations globally.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update

from src.observation_memory import AGE_SCALE_SECONDS
from src.unified_policy import UnifiedSAC, start_from_checkpoint as transfer_actor


VERSION = 'prior-actor-kl-and-task-balanced-replay-v1'
ANCHOR_COEFFICIENT = 10.0
ANCHOR_MAX_KL = .10
BLIND_CLEARANCE_SECONDS = 1.8


class TaskBalancedReplayBuffer(ReplayBuffer):
    """Sample half single-pass and half lap transitions when both exist.

    A slot label follows each ring-buffer overwrite. Sampling scans this small
    boolean vector, not the strided full observation array. Ordinary SB3 sample
    construction retains normalization and terminal/time-limit semantics.
    """

    observation_dim = 100

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.n_envs != 1 or self.optimize_memory_usage:
            raise ValueError('Task-balanced replay currently requires one environment and explicit next observations.')
        if len(self.obs_shape) != 1 or self.obs_shape[0] != self.observation_dim:
            raise ValueError('Task-balanced replay requires the 100-value mission observation.')
        self._single_pass_slots = np.zeros(self.buffer_size, dtype=bool)

    def add(self, obs, next_obs, action, reward, done, infos):
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self._single_pass_slots[slot] = bool(np.asarray(obs).reshape(self.n_envs, -1)[0, 99] > .5)

    def sample(self, batch_size, env=None):
        count = self.size()
        if count == 0:
            raise ValueError('Cannot sample empty replay.')
        labels = self._single_pass_slots[:count]
        single, laps = np.flatnonzero(labels), np.flatnonzero(~labels)
        if len(single) and len(laps):
            half = batch_size//2
            indices = np.concatenate([np.random.choice(single, half),
                                      np.random.choice(laps, batch_size-half)])
            np.random.shuffle(indices)
        else:
            indices = np.random.randint(0, count, size=batch_size)
        return self._get_samples(indices, env=env)


def protected_observations(observations):
    """Use requested task, current camera and portable last-detection memory."""
    if observations.ndim != 2 or observations.shape[1] != 100:
        raise ValueError('Anchoring requires the 100-value mission observation.')
    single = observations[:, -1] > .5
    current_front = (observations[:, 70] > .5) & (observations[:, 71] > .5)
    recent_front = ((observations[:, 90] > .5) & (observations[:, 91] > .5)
                    & (observations[:, 92] > .5) & (observations[:, 93] >= 0.)
                    & (observations[:, 93] <= BLIND_CLEARANCE_SECONDS/AGE_SCALE_SECONDS))
    return single | current_front | recent_front


def gaussian_kl(mean, log_std, reference_mean, reference_log_std):
    """KL(current || reference); tanh preserves KL for these distributions."""
    return (reference_log_std-log_std
            + (log_std.mul(2).exp()+(mean-reference_mean).square())
            / (2*reference_log_std.mul(2).exp())-.5).sum(dim=1).clamp_min(0.)


class AnchoredSAC(UnifiedSAC):
    _protection_mask = staticmethod(protected_observations)

    def __init__(self, *args, **kwargs):
        self._anchor_defaults()
        kwargs.setdefault('replay_buffer_class', TaskBalancedReplayBuffer)
        kwargs.setdefault('ent_coef', .013)
        super().__init__(*args, **kwargs)

    def _anchor_defaults(self):
        self.anchor_coefficient = ANCHOR_COEFFICIENT
        self.anchor_max_kl = ANCHOR_MAX_KL
        self.actor_attempts = 0
        self.anchor_accepted = 0
        self.anchor_rejected = 0
        self.last_anchor_kl = 0.

    def _make_reference_actor(self):
        self.reference_actor = deepcopy(self.actor)
        self.reference_actor.set_training_mode(False)
        self.reference_actor.requires_grad_(False)

    def _setup_model(self):
        super()._setup_model()
        self._make_reference_actor()

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['reference_actor']

    def _get_torch_save_params(self):
        state_dicts, variables = super()._get_torch_save_params()
        return state_dicts+['reference_actor'], variables

    @classmethod
    def load(cls, path, *args, **kwargs):
        # SB3's generic legacy-optimizer fallback obscures a missing reference
        # with KeyError('policy.optimizer'). Reject that incompatible resume
        # explicitly instead of inventing a new reference from a drifted actor.
        archive_path = path
        position = path.tell() if hasattr(path, 'tell') else None
        if isinstance(path, (str, Path)):
            archive_path = Path(path)
            if not archive_path.exists() and Path(str(archive_path)+'.zip').exists():
                archive_path = Path(str(archive_path)+'.zip')
        try:
            with zipfile.ZipFile(archive_path) as archive:
                if 'reference_actor.pth' not in archive.namelist():
                    raise ValueError('Anchored resume requires a saved reference_actor; use explicit actor transfer for other checkpoints.')
        finally:
            if position is not None:
                path.seek(position)
        return super().load(path, *args, **kwargs)

    def _reference_parameters(self, observations):
        with torch.no_grad():
            mean, log_std, _ = self.reference_actor.get_action_dist_params(observations)
        return mean, log_std

    def _protected_kl(self, observations, mask, reference):
        if not mask.any():
            return observations.new_zeros(())
        mean, log_std, _ = self.actor.get_action_dist_params(observations[mask])
        return gaussian_kl(mean, log_std, *reference).mean()

    def _accept_anchor_proposal(self, before, after):
        return after <= self.anchor_max_kl

    def _actor_step(self, observations, coefficient):
        """Attempt one update and transactionally reject an excessive batch KL."""
        mask = self._protection_mask(observations)
        reference = self._reference_parameters(observations[mask]) if mask.any() else None
        actions, log_prob = self.actor.action_log_prob(observations)
        q = torch.cat(self.critic(observations, actions), dim=1).min(dim=1).values
        rl_loss = (coefficient*log_prob-q).mean()
        penalty = self._protected_kl(observations, mask, reference)
        loss = rl_loss+self.anchor_coefficient*penalty
        parameters = {k: v.detach().clone() for k, v in self.actor.state_dict().items()}
        optimizer_state = deepcopy(self.actor.optimizer.state_dict())
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        with torch.no_grad():
            after = self._protected_kl(observations, mask, reference)
            valid = all(torch.isfinite(p).all().item() for p in self.actor.parameters())
            accepted = bool(valid and torch.isfinite(after).item() and self._accept_anchor_proposal(penalty.item(), after.item()))
        self.actor_attempts += 1
        if accepted:
            self.actor_updates += 1
            self.anchor_accepted += 1
            self.last_anchor_kl = after.item()
        else:
            self.actor.load_state_dict(parameters)
            self.actor.optimizer.load_state_dict(optimizer_state)
            self.actor.optimizer.zero_grad(set_to_none=True)
            self.anchor_rejected += 1
            with torch.no_grad():
                self.last_anchor_kl = self._protected_kl(observations, mask, reference).item()
        return {'loss': loss.item(), 'rl_loss': rl_loss.item(), 'kl': self.last_anchor_kl,
                'protected_fraction': mask.float().mean().item(), 'accepted': accepted,
                'before_kl': penalty.item()}

    def _train_anchored(self, gradient_steps, batch_size):
        if self.ent_coef_optimizer is not None:
            raise ValueError('Anchored transfer requires a fixed entropy coefficient.')
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        coefficient = self.ent_coef_tensor
        critic_losses, actor_stats = [], []
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discount = data.discounts if data.discounts is not None else self.gamma
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q = next_q.min(dim=1, keepdim=True).values
                target = data.rewards+(1-data.dones)*discount*(
                    next_q-coefficient*next_log_prob.reshape(-1, 1))
            current_q = self.critic(data.observations, data.actions)
            critic_loss = .5*sum(F.mse_loss(q, target) for q in current_q)
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            self.critic.optimizer.step()
            critic_losses.append(critic_loss.item())
            actor_stats.append(self._actor_step(data.observations, coefficient))
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/critic_loss', np.mean(critic_losses))
        self.logger.record('train/ent_coef', coefficient.item())
        for key in ('loss', 'rl_loss', 'kl', 'protected_fraction', 'accepted'):
            self.logger.record('train/anchor_'+key, np.mean([s[key] for s in actor_stats]))
        self.logger.record('train/anchor_rejected_total', self.anchor_rejected)

    def train(self, gradient_steps, batch_size=64):
        if self.replay_buffer.size() < self.refill_transitions:
            return
        calibration = min(gradient_steps, max(0, self.calibration_updates-self.critic_calibration_updates))
        if calibration:
            self._calibrate_critics(calibration, batch_size)
        if gradient_steps > calibration:
            self._train_anchored(gradient_steps-calibration, batch_size)
        self.logger.record('train/critic_calibration_updates', self.critic_calibration_updates)
        self.logger.record('train/actor_updates', self.actor_updates)


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    """Reuse verified actor transfer, then install the new learning rule.

    Changing this fresh Python instance's class retains the exact transferred
    actor/critic objects and optimizer parameter references. It is allowed only
    before any experience or update. Disk resume uses normal SB3 class loading.
    """
    model, metadata = transfer_actor(path, env, device=device, seed=seed,
                                    buffer_size=buffer_size, tensorboard_log=tensorboard_log)
    if model.num_timesteps or model._n_updates or model.replay_buffer.size():
        raise ValueError('Anchored transfer can only initialize a fresh model.')
    model.__class__ = AnchoredSAC
    model._anchor_defaults()
    model._make_reference_actor()
    if metadata['source_has_frozen_reference']:
        # Preserve the ORIGINAL retention reference across reward revisions.
        # Re-anchoring to the latest actor would accumulate reference drift.
        # The current actor is copied separately by transfer_actor above.
        with zipfile.ZipFile(path) as archive:
            with archive.open('reference_actor.pth') as saved_reference:
                reference_state = torch.load(saved_reference, map_location='cpu', weights_only=True)
        model.reference_actor.load_state_dict(reference_state)
    model.replay_buffer_class = TaskBalancedReplayBuffer
    model.replay_buffer_kwargs = {}
    model.replay_buffer = TaskBalancedReplayBuffer(
        buffer_size, model.observation_space, model.action_space,
        device=model.device, n_envs=model.n_envs,
        optimize_memory_usage=model.optimize_memory_usage)
    metadata.update({
        'policy_update_rule': VERSION,
        'reference': ('original_frozen_reference_from_source_checkpoint'
                      if metadata['source_has_frozen_reference'] else 'frozen_inherited_actor'),
        'anchor_coefficient': model.anchor_coefficient, 'anchor_max_batch_mean_kl': model.anchor_max_kl,
        'anchor_mask': 'single-pass task, current front visible, or front seen within 1.8 seconds',
        'replay_sampling': '50% single-pass and 50% lap transitions when both available',
        'new_demonstrations': 0,
    })
    return model, metadata
