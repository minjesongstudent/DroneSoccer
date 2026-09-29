"""Bounded optimizer retry for finite-policy updates; unchanged acceptance limits."""
from copy import deepcopy
import torch
from src.finite_policy_update import update_actor

VERSION = 'finite-episode-bounded-step-v2'


def bounded_update(actor, optimizer, *data, retries=4, **kwargs):
    """Retry from identical actor/moments/order with half the learning rate.

    Accept only if the existing full-collection KL and surrogate checks pass.
    Rejection cannot accumulate hidden updates. Restore the caller's learning
    rates after the operation; return the actual accepted rate in metadata.
    """
    if type(retries) is not int or not 0 <= retries <= 4:
        raise ValueError('At most four bounded retries permitted')
    state, moments = deepcopy(actor.state_dict()), deepcopy(optimizer.state_dict())
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    rates = [g['lr'] for g in optimizer.param_groups]
    attempts = []
    accepted = False
    try:
        for retry in range(retries+1):
            actor.load_state_dict(state);optimizer.load_state_dict(moments)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:torch.cuda.set_rng_state_all(cuda_rng)
            factor = .5**retry
            for group, rate in zip(optimizer.param_groups, rates):group['lr'] = rate*factor
            history = update_actor(actor, optimizer, *data, **kwargs)
            accepted = any(row['accepted'] for row in history)
            attempts.append(dict(factor=factor, epochs=history, accepted=accepted))
            if accepted:break
    except BaseException:
        actor.load_state_dict(state);optimizer.load_state_dict(moments)
        raise
    finally:
        if not accepted:
            actor.load_state_dict(state);optimizer.load_state_dict(moments)
        for group, rate in zip(optimizer.param_groups, rates):group['lr'] = rate
    return attempts
