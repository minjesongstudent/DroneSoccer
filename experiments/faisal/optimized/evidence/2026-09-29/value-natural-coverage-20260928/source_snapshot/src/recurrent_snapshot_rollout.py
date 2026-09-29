"""Single-trajectory PPO segments with current-weight authentic-prefix burn-in.

No autoresets: trajectory boundaries are explicit. This collector treats the
environment's finite mission timeout as terminal, while a collector length
limit bootstraps. Prefix replay is detached truncated BPTT, never a simulator
action source. Call evaluate_segment anew before every optimizer step.
"""
from dataclasses import dataclass
import numpy as np
import torch
from .recurrent_burnin import rebuild_actor_critic_memory


@dataclass
class Segment:
    prefix: np.ndarray
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    values: np.ndarray
    log_probs: np.ndarray
    bootstrap: float
    mission_ended: bool
    next_observation: np.ndarray
    next_prefix: np.ndarray
    final_info: dict
    natural_start: bool = False


def _initial_memory(model, prefix, natural_start):
    if not natural_start:
        return rebuild_actor_critic_memory(model, prefix)
    if np.asarray(prefix).shape != (0,119):
        raise ValueError('Natural reset must have an empty prefix')
    from sb3_contrib.common.recurrent.type_aliases import RNNStates
    zero=lambda: torch.zeros(model.policy.lstm_hidden_state_shape,device=model.device)
    return RNNStates((zero(),zero()),(zero(),zero())),torch.ones(1,device=model.device)


def collect_natural_segment(model, environment, seed, length=64):
    observation,_=environment.reset(seed=seed)
    return collect_segment(model,environment,observation,np.empty((0,119),np.float32),length,natural_start=True)


def collect_segment(model, environment, observation, prefix, length=64, *, natural_start=False):
    if length < 1:
        raise ValueError('Positive segment length required')
    policy = model.policy
    policy.set_training_mode(False)
    state, start = _initial_memory(model, prefix, natural_start)
    observations, actions, rewards, values, logs = [], [], [], [], []
    ended = False
    with torch.no_grad():
        for _ in range(length):
            observations.append(np.asarray(observation, dtype=np.float32).copy())
            tensor, _ = policy.obs_to_tensor(observation)
            action, value, log_prob, state = policy.forward(tensor, state, start)
            start = torch.zeros_like(start)
            raw = action[0].cpu().numpy()
            # PPO likelihood is for the raw Gaussian sample, as in SB3.
            # The environment receives its bounded version only.
            issued = np.clip(raw, environment.action_space.low, environment.action_space.high)
            observation, reward, terminated, truncated, info = environment.step(issued)
            actions.append(raw.copy()); rewards.append(float(reward))
            values.append(float(value.item())); logs.append(float(log_prob.item()))
            ended = bool(terminated or truncated)
            if ended:
                break
        bootstrap = 0.0
        if not ended:
            tensor, _ = policy.obs_to_tensor(observation)
            bootstrap = float(policy.predict_values(tensor, state.vf, start).item())
    observations = np.asarray(observations, dtype=np.float32)
    return Segment(np.asarray(prefix, dtype=np.float32).copy(), observations,
                   np.asarray(actions), np.asarray(rewards), np.asarray(values),
                   np.asarray(logs), bootstrap, ended, observation.copy(),
                   np.concatenate([prefix, observations]), dict(info), natural_start)


def advantages(segment, gamma=.99, lam=.95):
    """GAE for one uninterrupted segment; never propagates into another reset."""
    result = np.zeros(len(segment.rewards), dtype=np.float32)
    carry = 0.0
    next_value = segment.bootstrap
    for i in range(len(result)-1, -1, -1):
        delta = segment.rewards[i] + gamma * next_value - segment.values[i]
        carry = delta + gamma * lam * carry
        result[i] = carry
        next_value = segment.values[i]
    return result, result + segment.values


def evaluate_segment(model, segment):
    # Rebuilt on every call, including after the previous minibatch update.
    # cuDNN recurrent backward requires a training-mode forward.
    model.policy.set_training_mode(True)
    natural_start=getattr(segment,'natural_start',False)
    state, start = _initial_memory(model, segment.prefix, natural_start)
    policy = model.policy
    observations = torch.as_tensor(segment.observations, device=policy.device)
    actions = torch.as_tensor(segment.actions, device=policy.device)
    values, logs, entropies = [], [], []
    # Match the collector's one-step recurrent execution. Long batched cuDNN
    # sequences can drift numerically from stepwise rollout likelihoods.
    # Keep state transitions in the graph for within-segment BPTT.
    for observation, action in zip(observations, actions):
        value, log, entropy = policy.evaluate_actions(observation[None], action[None], state, start)
        _, _, _, state = policy.forward(observation[None], state, start, deterministic=True)
        start = torch.zeros_like(start)
        values.append(value); logs.append(log); entropies.append(entropy)
    return torch.cat(values), torch.cat(logs), torch.cat(entropies)
