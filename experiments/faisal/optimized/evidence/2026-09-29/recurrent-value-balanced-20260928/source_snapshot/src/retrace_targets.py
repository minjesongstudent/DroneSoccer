"""Detached finite-horizon soft Retrace critic targets; no actor updates.

This is a soft Bellman residual construction, not a convergence claim for an
aliased observation and a neural twin critic. Q excludes its queried action's
entropy; next_soft_values already contains the next policy entropy term.
"""
import math

import torch

VERSION = 'finite-soft-retrace-residual-v1'


@torch.no_grad()
def soft_retrace_targets(rewards, terminated, truncated, valid, target_q,
                         next_soft_values, behavior_log_prob, target_log_prob,
                         *, gamma, trace_lambda):
    """Return one target per BxT sequence, stopping traces at both boundaries.

    B=min(target Q1,Q2), V'=E[min(target Q1,Q2)-alpha*log pi].
    delta=r+gamma*(1-terminated)*V'-B; G=B0+sum gamma^k*c1*...*ck*delta_k,
    with c_i=lambda*min(1,pi(a_i|o_i)/mu(a_i|o_i)). There is no c0 factor.
    Timeout tails bootstrap through V' but no trace crosses into a new flight.
    On an identical frozen behavior/target actor pass the same stored logp
    tensor twice: the ratio is exactly one without inverse-tanh roundoff.
    """
    if (not math.isfinite(gamma) or not 0 <= gamma <= 1
            or not math.isfinite(trace_lambda) or not 0 <= trace_lambda <= 1):
        raise ValueError('gamma and trace_lambda must be finite values in [0,1].')
    arrays = (rewards,terminated,truncated,valid,target_q,next_soft_values,
              behavior_log_prob,target_log_prob)
    if rewards.ndim != 2 or min(rewards.shape) <= 0:
        raise ValueError('Expected nonempty BxT sequences.')
    if any(value.shape != rewards.shape or value.device != rewards.device for value in arrays):
        raise ValueError('Every field must have the same BxT shape and device.')
    if any(value.dtype != torch.bool for value in (terminated,truncated,valid)):
        raise ValueError('Termination, truncation and validity masks must be boolean.')
    if any(not torch.isfinite(value).all().item() for value in
           (rewards,target_q,next_soft_values,behavior_log_prob,target_log_prob)):
        raise ValueError('Target fields must be finite, including padded entries.')
    if not valid[:,0].all().item():
        raise ValueError('Each sequence must contain its starting transition.')
    if (valid[:,1:] & (~valid[:,:-1] | terminated[:,:-1] | truncated[:,:-1])).any().item():
        raise ValueError('A trace cannot resume after padding or cross a flight boundary.')
    delta = rewards+gamma*(~terminated).to(rewards.dtype)*next_soft_values-target_q
    # Clamp the log ratio before exponentiation so very large ratios cannot overflow.
    coefficient = trace_lambda*torch.exp(torch.minimum(target_log_prob-behavior_log_prob,
                                                       torch.zeros_like(rewards)))
    # Cancel B0 analytically, avoiding floating-point cancellation and making
    # n=1/lambda=0 use precisely the same one-step expression as SAC.
    result = rewards[:,0]+gamma*(~terminated[:,0]).to(rewards.dtype)*next_soft_values[:,0]
    weight = torch.ones_like(result)
    for index in range(1,rewards.shape[1]):
        weight = weight*gamma*coefficient[:,index]
        result += weight*torch.where(valid[:,index],delta[:,index],torch.zeros_like(result))
    return result
