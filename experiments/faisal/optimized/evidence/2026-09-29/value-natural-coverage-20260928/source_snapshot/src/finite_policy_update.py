"""Experimental finite-episode score-function update; not a SAC resume path.

Actions and pre-tanh samples must be recorded from the behavior policy.
No action-value critic gradients or bootstrapped returns enter this update.
"""
from copy import deepcopy
import numpy as np
import torch

VERSION = 'finite-episode-clipped-update-v1'


def soft_returns(rewards, log_prob, alpha, gamma):
    rewards, log_prob = np.asarray(rewards), np.asarray(log_prob)
    if rewards.ndim != 1 or rewards.shape != log_prob.shape or not len(rewards):
        raise ValueError('One complete nonempty episode required')
    if not np.isfinite(rewards).all() or not np.isfinite(log_prob).all():
        raise ValueError('Finite rewards and exact behavior densities required')
    if not np.isfinite([alpha, gamma]).all() or alpha < 0 or not 0 <= gamma <= 1:
        raise ValueError('Invalid entropy coefficient or discount')
    result = np.zeros(len(rewards), dtype=np.float64)
    tail = 0.
    for t in reversed(range(len(rewards))):
        tail = rewards[t] - alpha*log_prob[t] + gamma*tail
        result[t] = tail
    return result


def gaussian_kl(old_mu, old_ls, mu, ls):
    return (ls-old_ls + (torch.exp(2*old_ls)+(old_mu-mu)**2)/(2*torch.exp(2*ls))-.5).sum(-1)


def clipped_gain(log_ratio, weights, clip=.1):
    ratio = log_ratio.exp()
    return torch.minimum(ratio*weights, ratio.clamp(1-clip, 1+clip)*weights).mean()


def update_actor(actor, optimizer, x, actions, gaussian, weights, behavior_logp,
                 *, epochs=4, batch_size=512, mean_bound=.001, max_bound=.02):
    """Keep accepted epochs only; reject/rollback a whole epoch on KL or gain.

    Bounds always compare with the collection policy, not the preceding epoch.
    Optimizer moments are restored together with weights after rejection.
    """
    if epochs < 1 or batch_size < 1 or len(x) != len(weights):
        raise ValueError('Invalid batch contract')
    if any(t.requires_grad for t in (x, actions, gaussian, weights, behavior_logp)):
        raise ValueError('Recorded data must be detached')
    if any(not torch.isfinite(t).all() for t in (x, actions, gaussian, weights, behavior_logp)):
        raise ValueError('Nonfinite rollout data')
    if not torch.allclose(actions, gaussian.tanh(), atol=1e-6, rtol=1e-5):
        raise ValueError('Recorded pre-tanh samples disagree with actions')
    with torch.no_grad():
        old_mu, old_ls, _ = actor.get_action_dist_params(x)
        old_mu, old_ls = old_mu.clone(), old_ls.clone()
        old_lp = actor.action_dist.proba_distribution(old_mu, old_ls).log_prob(actions, gaussian).clone()
        if not torch.allclose(old_lp, behavior_logp, atol=2e-4, rtol=1e-5):
            raise ValueError('Data were not collected by the current policy')
    history = []
    previous_gain = 0.
    for epoch in range(epochs):
        state, moments = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
        for ids in torch.randperm(len(x), device=x.device).split(batch_size):
            mu, ls, _ = actor.get_action_dist_params(x[ids])
            lp = actor.action_dist.proba_distribution(mu, ls).log_prob(actions[ids], gaussian[ids])
            loss = -clipped_gain(lp-old_lp[ids], weights[ids])
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), .5, error_if_nonfinite=True)
            optimizer.step()
        with torch.no_grad():
            mu, ls, _ = actor.get_action_dist_params(x)
            div = gaussian_kl(old_mu, old_ls, mu, ls)
            lp = actor.action_dist.proba_distribution(mu, ls).log_prob(actions, gaussian)
            gain = float(clipped_gain(lp-old_lp, weights)-weights.mean())
            mean, maximum = float(div.mean()), float(div.max())
        accepted = bool(np.isfinite([gain, mean, maximum]).all() and
                        gain > previous_gain and mean <= mean_bound and maximum <= max_bound)
        history.append(dict(epoch=epoch, gain=gain, mean_kl=mean, max_kl=maximum, accepted=accepted))
        if not accepted:
            actor.load_state_dict(state);optimizer.load_state_dict(moments)
            break
        previous_gain = gain
    return history
