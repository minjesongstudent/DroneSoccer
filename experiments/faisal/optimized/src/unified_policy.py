"""Actor-preserving SAC transfer with fresh critics for a changed objective.

The inherited actor supplies all training actions. No teacher, demonstration
loss, or old-reward replay is imported. Extra observation features initially
have zero actor weights, preserving the source policy's action distribution.
"""
from __future__ import annotations

import hashlib
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.utils import polyak_update


VERSION = 'actor-transfer-fresh-critics-calibration-v2'
REFILL_TRANSITIONS = 5000
CALIBRATION_UPDATES = 5000
ACTOR_LEARNING_RATE = 3e-5
CRITIC_LEARNING_RATE = 3e-4
SOURCE_OBSERVATION_DIM = 84


class UnifiedSAC(SAC):
    """Refill replay, fit critics with a frozen actor, then run ordinary SAC.

    Calibration counters and fixed learning rates are normal model attributes,
    so SB3 checkpoints preserve them on disk resume. The actor's optimizer is
    not stepped at all during calibration, including its Adam moment estimates.
    """

    def __init__(self, *args, **kwargs):
        self.refill_transitions = REFILL_TRANSITIONS
        self.calibration_updates = CALIBRATION_UPDATES
        self.critic_calibration_updates = 0
        self.actor_updates = 0
        self.actor_learning_rate = ACTOR_LEARNING_RATE
        self.critic_learning_rate = CRITIC_LEARNING_RATE
        super().__init__(*args, **kwargs)

    @property
    def transfer_phase(self):
        if self.replay_buffer is None or self.replay_buffer.size() < self.refill_transitions:
            return 'replay_refill'
        if self.critic_calibration_updates < self.calibration_updates:
            return 'critic_calibration'
        return 'sac_training'

    def _update_learning_rate(self, optimizers):
        # SAC normally sets both optimizers to one schedule at every train call.
        # Keep the inherited actor's smaller update rate without replacing SAC's
        # losses or introducing an imitation objective.
        optimizers = optimizers if isinstance(optimizers, list) else [optimizers]
        for optimizer in optimizers:
            rate = (self.actor_learning_rate if optimizer is self.actor.optimizer
                    else self.critic_learning_rate)
            for group in optimizer.param_groups:
                group['lr'] = rate
        self.logger.record('train/learning_rate', self.critic_learning_rate)
        self.logger.record('train/actor_learning_rate', self.actor_learning_rate)

    def _calibrate_critics(self, gradient_steps, batch_size):
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.critic.optimizer])
        losses = []
        coefficient = (self.log_ent_coef.detach().exp() if self.log_ent_coef is not None
                       else self.ent_coef_tensor)
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            discount = data.discounts if data.discounts is not None else self.gamma
            with torch.no_grad():
                next_actions, next_log_prob = self.actor.action_log_prob(data.next_observations)
                next_q = torch.cat(self.critic_target(data.next_observations, next_actions), dim=1)
                next_q = next_q.min(dim=1, keepdim=True).values
                target = data.rewards + (1-data.dones)*discount*(
                    next_q-coefficient*next_log_prob.reshape(-1, 1))
            current_q = self.critic(data.observations, data.actions)
            loss = .5*sum(F.mse_loss(q, target) for q in current_q)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            losses.append(loss.item())
            if self.critic_calibration_updates % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
            self.critic_calibration_updates += 1
            self._n_updates += 1
        self.logger.record('train/critic_loss', np.mean(losses))
        self.logger.record('train/ent_coef', coefficient.item())
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')

    def train(self, gradient_steps, batch_size=64):
        if self.replay_buffer.size() < self.refill_transitions:
            return
        calibration = min(gradient_steps, max(0, self.calibration_updates-self.critic_calibration_updates))
        if calibration:
            self._calibrate_critics(calibration, batch_size)
        remaining = gradient_steps-calibration
        if remaining:
            super().train(remaining, batch_size)
            self.actor_updates += remaining
        self.logger.record('train/critic_calibration_updates', self.critic_calibration_updates)
        self.logger.record('train/actor_updates', self.actor_updates)


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    """Copy the actor only; never treat old critics as valid for new rewards."""
    from pathlib import Path

    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        has_reference = 'reference_actor.pth' in archive.namelist()
    if has_reference:
        # A previous anchored run has a separately saved frozen reference.
        # Load its actual class so SB3 validates every saved parameter group.
        from src.anchored_policy import AnchoredSAC
        old = AnchoredSAC.load(path, device='cpu')
    else:
        old = SAC.load(path, device='cpu')
    source_dim = int(np.prod(old.observation_space.shape))
    new_dim = int(np.prod(env.observation_space.shape))
    if source_dim not in (SOURCE_OBSERVATION_DIM, new_dim) or new_dim < SOURCE_OBSERVATION_DIM:
        raise ValueError('Transfer requires the original 84 observations as an unchanged prefix, or an identical current observation contract.')
    if source_dim > new_dim:
        raise ValueError('Cannot discard observations from the source actor.')
    if old.use_sde or old.action_space.shape != env.action_space.shape:
        raise ValueError('Transfer requires the original Gaussian action distribution and action shape.')
    if buffer_size < REFILL_TRANSITIONS:
        raise ValueError('Replay capacity must cover the transfer refill period.')
    coefficient = float(old.log_ent_coef.detach().exp().item()
                        if old.log_ent_coef is not None else old.ent_coef_tensor.item())
    model = UnifiedSAC(
        'MlpPolicy', env, device=device, seed=seed,
        learning_rate=CRITIC_LEARNING_RATE, buffer_size=buffer_size,
        learning_starts=0, batch_size=old.batch_size, tau=old.tau,
        gamma=old.gamma, train_freq=(1, 'step'), gradient_steps=1,
        ent_coef=coefficient, target_update_interval=old.target_update_interval,
        tensorboard_log=str(tensorboard_log),
        policy_kwargs={'net_arch': old.actor.net_arch,
                       'activation_fn': old.actor.activation_fn},
    )
    actor_state = model.actor.state_dict()
    for name, source_tensor in old.actor.state_dict().items():
        if name == 'latent_pi.0.weight':
            actor_state[name].zero_()
            actor_state[name][:, :source_dim].copy_(source_tensor)
        else:
            if actor_state[name].shape != source_tensor.shape:
                raise ValueError(f'Unsupported actor architecture at {name}.')
            actor_state[name].copy_(source_tensor)
    model.actor.load_state_dict(actor_state)
    for group in model.actor.optimizer.param_groups:
        group['lr'] = model.actor_learning_rate
    source = {
        'steps': old.num_timesteps, 'updates': old._n_updates,
        'imitation_updates': getattr(old, 'imitation_updates', 0),
        'transfer': 'actor_only', 'fresh_critics': True, 'fresh_optimizers': True,
        'entropy_coefficient': coefficient, 'entropy_mode': 'fixed_at_source_value',
        'source_observation_dim': source_dim, 'observation_dim': new_dim,
        'source_has_frozen_reference': has_reference,
        'replay_refill_transitions': model.refill_transitions,
        'critic_calibration_updates': model.calibration_updates,
        'actor_learning_rate': model.actor_learning_rate,
        'critic_learning_rate': model.critic_learning_rate,
        'source_sha256': source_hash,
    }
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source checkpoint changed during transfer.')
    return model, source
