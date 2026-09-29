"""Rebuild inference memory from authentic public inputs under current weights.

This does not implement training-rollout integration.
Returned state must be recomputed after policy weights change.
"""
import numpy as np

def _prefix(prefix):
    observations=np.asarray(prefix,dtype=np.float32)
    if observations.ndim!=2 or observations.shape[1]!=119 or len(observations)==0 or not np.isfinite(observations).all():
        raise ValueError('Nonempty finite authentic119D prefix required')
    return observations

def rebuild_actor_memory(model,prefix):
    observations=_prefix(prefix)
    state=None
    start=np.ones(1,dtype=bool)
    for observation in observations:
        _,state=model.predict(observation,state=state,episode_start=start,deterministic=True)
        start[:]=False
    return state,start

def rebuild_actor_critic_memory(model, prefix):
    """Replay actual inputs before the current observation, without gradients.

    Discarded deterministic outputs never control the simulator. The recorded
    commands already embedded in the inputs remain unchanged. Recompute after
    every weight update; returned memory is a detached burn-in boundary.
    """
    import torch
    from sb3_contrib.common.recurrent.type_aliases import RNNStates

    observations = _prefix(prefix)
    policy = model.policy
    shape = policy.lstm_hidden_state_shape
    zero = lambda: torch.zeros(shape, device=policy.device)
    state = RNNStates((zero(), zero()), (zero(), zero()))
    start = torch.ones(1, device=policy.device)
    was_training = policy.training
    policy.set_training_mode(False)
    try:
        with torch.no_grad():
            for observation in observations:
                obs, _ = policy.obs_to_tensor(observation)
                _, _, _, state = policy.forward(obs, state, start, deterministic=True)
                start = torch.zeros_like(start)
    finally:
        policy.set_training_mode(was_training)
    return state, start
