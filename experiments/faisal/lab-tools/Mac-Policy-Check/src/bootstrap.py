"""Demonstration warm-start; only camera-history observations enter the policy.

The scripted teacher is used offline to collect training data. Prediction and
evaluation always use the learned actor alone. Demonstrations never include the
fixed evaluation seed range, and are counted separately from online steps.
"""
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import SAC


class GuidedSAC(SAC):
    """SAC with a fading imitation loss during the first 50k online steps."""
    anchor_until = 50_000

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['demo_obs', 'demo_actions']

    def set_demonstrations(self, observations, actions):
        self.demo_obs = torch.as_tensor(observations, dtype=torch.float32, device=self.device)
        self.demo_actions = torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def train(self, gradient_steps, batch_size=64):
        super().train(gradient_steps, batch_size)
        weight = max(0., 1. - self.num_timesteps / self.anchor_until)
        if not weight or gradient_steps <= 0 or not hasattr(self, 'demo_obs'):
            return
        # Keep successful experience influential while the initially untrained
        # critic learns its values. This auxiliary loss fades to zero, after
        # which learning is ordinary SAC. It never selects rollout actions.
        for _ in range(gradient_steps):
            ids = torch.randint(len(self.demo_obs), (batch_size,), device=self.device)
            prediction = self.actor(self.demo_obs[ids], deterministic=True)
            loss = weight * torch.nn.functional.mse_loss(prediction, self.demo_actions[ids])
            self.actor.optimizer.zero_grad()
            loss.backward()
            self.actor.optimizer.step()
            self.imitation_updates = getattr(self, 'imitation_updates', 0) + 1
        self.logger.record('train/imitation_weight', weight)
        self.logger.record('train/imitation_loss', loss.item())


def restore_demonstrations(model, path):
    with np.load(path, allow_pickle=False) as data:
        model.set_demonstrations(data['observations'], data['actions'])


def bootstrap(model, transitions, output, updates=1500, progress=None):
    if not transitions:
        raise ValueError('Cannot warm-start without verified successful transitions')
    if len(transitions) >= model.replay_buffer.buffer_size:
        raise ValueError('Demonstrations must fit in the replay buffer')
    observations = np.stack([t['obs'] for t in transitions]).astype(np.float32)
    actions = np.stack([t['action'] for t in transitions]).astype(np.float32)
    if not np.isfinite(observations).all() or not np.isfinite(actions).all():
        raise ValueError('Demonstration contains non-finite values')
    if np.any(np.abs(actions) > 1.):
        raise ValueError('Demonstration action outside policy limits')
    np.savez_compressed(Path(output), observations=observations, actions=actions)
    model.set_demonstrations(observations, actions)
    for transition in transitions:
        info = dict(transition['info'])
        info['TimeLimit.truncated'] = bool(transition['truncated'] and not transition['terminated'])
        model.replay_buffer.add(
            transition['obs'][None], transition['next_obs'][None],
            transition['action'][None], np.array([transition['reward']]),
            np.array([transition['terminated'] or transition['truncated']]), [info])
    model.policy.set_training_mode(True)
    with torch.no_grad():
        before = torch.nn.functional.mse_loss(
            model.actor(model.demo_obs, deterministic=True), model.demo_actions).item()
    for step in range(updates):
        ids = torch.randint(len(observations), (256,), device=model.device)
        mean, log_std, _ = model.actor.get_action_dist_params(model.demo_obs[ids])
        # Warm-start a narrow, nonzero action distribution. SAC remains
        # stochastic during online training and tunes entropy on its own.
        loss = (torch.nn.functional.mse_loss(torch.tanh(mean), model.demo_actions[ids])
                + .01 * torch.nn.functional.mse_loss(log_std, torch.full_like(log_std, -2.5)))
        model.actor.optimizer.zero_grad()
        loss.backward()
        model.actor.optimizer.step()
        if progress and (step + 1) % 100 == 0:
            progress(step + 1, updates)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        after = torch.nn.functional.mse_loss(
            model.actor(model.demo_obs, deterministic=True), model.demo_actions).item()
    return {'transitions': len(transitions), 'pretrain_updates': updates,
            'action_mse_before': before, 'action_mse_after': after,
            'anchor_until_online_step': model.anchor_until,
            'initial_log_std_target': -2.5, 'device': str(model.device)}
