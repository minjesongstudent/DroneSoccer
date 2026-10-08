"""Frozen-actor critic adaptation and reversible, reference-bounded SAC updates."""
import copy

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.utils import polyak_update


def gaussian_kl(reference_mean, reference_log_std, mean, log_std):
    # KL(reference || candidate). A shared tanh bijection preserves this KL.
    return (log_std-reference_log_std +
            (torch.exp(2*reference_log_std)+(reference_mean-mean).square()) /
            (2*torch.exp(2*log_std)) - .5).sum(dim=-1)


def episode_partition(dones, seed_base=11_000_000):
    """Group complete episodes, not adjacent transitions, across FIT/holdout."""
    done = np.asarray(dones).reshape(-1).astype(bool)
    ids = np.cumsum(np.r_[0, done[:-1]]).astype(np.int64)
    seeds = seed_base + ids
    heldout = (ids % 5) == 4
    return np.flatnonzero(~heldout), np.flatnonzero(heldout), seeds


def boundary_penalty(kl, mean_limit, max_limit):
    """Steer away before the hard guard; do not change its acceptance limits."""
    return (F.relu(kl.mean()/(mean_limit*.5)-1).square() +
            F.relu(kl.max()/(max_limit*.5)-1).square())


def guarded_optimizer_step(actor, optimizer, loss, check, max_backtracks=0):
    """Restore weights AND Adam moments when the candidate exceeds its guard."""
    weights = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    state = copy.deepcopy(optimizer.state_dict())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(actor.parameters(), .5)
    for attempt in range(max_backtracks+1):
        if attempt:
            actor.load_state_dict(weights)
            optimizer.load_state_dict(copy.deepcopy(state))
        for group, original in zip(optimizer.param_groups,state['param_groups']):
            group['lr']=original['lr']*(.5**attempt)
        optimizer.step()
        accepted, metrics = check()
        if accepted:break
    if not accepted:
        actor.load_state_dict(weights)
        optimizer.load_state_dict(state)
    else:
        for group,original in zip(optimizer.param_groups,state['param_groups']):
            group['lr']=original['lr']
    if max_backtracks:
        metrics=dict(metrics,backtracks=attempt)
    optimizer.zero_grad(set_to_none=True)
    return accepted, metrics


class ConstrainedSAC(SAC):
    def _excluded_save_params(self):
        return super()._excluded_save_params() + [
            'reference_actor', 'trust_observations', 'trust_distribution']

    def configure(self, fit_guard, actor_lr=1e-6, critic_lr=3e-4,
                  mean_limit=.01, max_limit=.1, penalty=1.0, boundary_weight=0.0,
                  uniform_actor_replay=False):
        self.reference_actor = copy.deepcopy(self.actor)
        self.reference_actor.set_training_mode(False)
        for parameter in self.reference_actor.parameters():
            parameter.requires_grad_(False)
        self.trust_observations = torch.as_tensor(fit_guard, device=self.device)
        with torch.no_grad():
            self.trust_distribution = self.reference_actor.get_action_dist_params(self.trust_observations)[:2]
        self.actor.optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic.optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.mean_limit, self.max_limit, self.kl_penalty = mean_limit, max_limit, penalty
        self.boundary_weight = boundary_weight
        self.uniform_actor_replay = uniform_actor_replay
        self.frozen_alpha = float(self.log_ent_coef.detach().exp().item())
        self.actor_enabled = False
        self.accepted_updates = self.rejected_updates = self.consecutive_rejections = 0
        self.last_stats = {}

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        return super()._sample_action(0, action_noise, n_envs)

    def trust_check(self):
        with torch.no_grad():
            mean, log_std, _ = self.actor.get_action_dist_params(self.trust_observations)
            kl = gaussian_kl(*self.trust_distribution, mean, log_std)
            metrics = dict(mean_kl=float(kl.mean()), max_kl=float(kl.max()))
        accepted = (np.isfinite(list(metrics.values())).all() and
                    metrics['mean_kl'] <= self.mean_limit and metrics['max_kl'] <= self.max_limit)
        return bool(accepted), metrics

    def critic_target_values(self, data):
        with torch.no_grad():
            action, log_prob = self.actor.action_log_prob(data.next_observations)
            q = torch.cat(self.critic_target(data.next_observations, action), dim=1).min(dim=1, keepdim=True).values
            return data.rewards + (1-data.dones)*self.gamma*(q-self.frozen_alpha*log_prob[:, None])

    def train(self, gradient_steps, batch_size=256):
        self.policy.set_training_mode(True)
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size)
            target = self.critic_target_values(data)
            current = self.critic(data.observations, data.actions)
            critic_loss = .5*sum(F.mse_loss(q, target) for q in current)
            if not torch.isfinite(critic_loss):
                raise FloatingPointError('Non-finite critic loss')
            self.critic.optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10.)
            self.critic.optimizer.step()
            metrics = dict(critic_loss=float(critic_loss.detach()))
            if self.actor_enabled:
                if self.uniform_actor_replay:
                    # Bypass the collision-biased subclass only for actor states.
                    data = ReplayBuffer.sample(self.replay_buffer, batch_size)
                for p in self.critic.parameters():
                    p.requires_grad_(False)
                try:
                    action, log_prob = self.actor.action_log_prob(data.observations)
                    q = torch.cat(self.critic(data.observations, action), dim=1).min(dim=1).values
                    with torch.no_grad():
                        rm, rs, _ = self.reference_actor.get_action_dist_params(data.observations)
                    mean, std, _ = self.actor.get_action_dist_params(data.observations)
                    penalty = gaussian_kl(rm, rs, mean, std).mean()
                    loss = (self.frozen_alpha*log_prob-q).mean()+self.kl_penalty*penalty
                    if self.boundary_weight:
                        gm, gs, _ = self.actor.get_action_dist_params(self.trust_observations)
                        gkl = gaussian_kl(*self.trust_distribution, gm, gs)
                        loss = loss + self.boundary_weight*boundary_penalty(
                            gkl, self.mean_limit, self.max_limit)
                    if not torch.isfinite(loss):
                        raise FloatingPointError('Non-finite actor loss')
                    accepted, drift = guarded_optimizer_step(self.actor, self.actor.optimizer,
                                                              loss, self.trust_check)
                    self.accepted_updates += int(accepted)
                    self.rejected_updates += int(not accepted)
                    self.consecutive_rejections = 0 if accepted else self.consecutive_rejections+1
                    metrics.update(actor_loss=float(loss.detach()), accepted=accepted, **drift)
                finally:
                    for p in self.critic.parameters():
                        p.requires_grad_(True)
            polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
            self._n_updates += 1
            self.last_stats = metrics
