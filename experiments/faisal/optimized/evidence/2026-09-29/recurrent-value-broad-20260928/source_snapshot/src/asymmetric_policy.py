"""Anchored SAC whose critic alone receives a separate simulator context.

Actor observations, prediction and retention masks remain exactly 100 values.
Privileged context travels through replay info, never through the actor's
observation space. This experimental critic is not a claim of Markov state
sufficiency or evidence of improved control; calibrate and evaluate it first.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import NamedTuple
import zipfile

from gymnasium import spaces
import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3 import SAC
from stable_baselines3.common.policies import ContinuousCritic
from stable_baselines3.common.torch_layers import FlattenExtractor
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.sac.policies import SACPolicy

from src import unified_policy
from src.anchored_policy import AnchoredSAC, TaskBalancedReplayBuffer, protected_observations
from src.critic_context import CONTEXT_DIM, VERSION as CONTEXT_VERSION


VERSION = 'separate-context-asymmetric-critic-v1'
PUBLIC_OBSERVATION_DIM = 100


def critic_inputs(observations, context):
    """Require an explicit matching context; never guess, cache or zero-fill it."""
    if (observations.ndim != 2 or observations.shape[1] != PUBLIC_OBSERVATION_DIM
            or context is None or context.ndim != 2
            or context.shape != (observations.shape[0], CONTEXT_DIM)):
        raise ValueError('Critic requires aligned 100-value observations and explicit simulator context.')
    return torch.cat((observations, context), dim=1)


class AsymmetricReplaySamples(NamedTuple):
    observations: torch.Tensor
    actions: torch.Tensor
    next_observations: torch.Tensor
    dones: torch.Tensor
    rewards: torch.Tensor
    discounts: torch.Tensor | None
    critic_context: torch.Tensor
    next_critic_context: torch.Tensor


class AsymmetricReplayBuffer(TaskBalancedReplayBuffer):
    """Keep pre-action and true post-action context aligned with each slot.

    SB3 replaces autoreset observations with terminal_observation before add;
    info's next context already describes that terminal state. Timeout masks
    remain the standard replay implementation's responsibility.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.context_version, self.context_dim = CONTEXT_VERSION, CONTEXT_DIM
        shape = (self.buffer_size, self.n_envs, CONTEXT_DIM)
        self.critic_contexts = np.zeros(shape, np.float32)
        self.next_critic_contexts = np.zeros(shape, np.float32)

    def add(self, obs, next_obs, action, reward, done, infos):
        contexts = []
        if len(infos) != self.n_envs:
            raise ValueError('Every replay transition needs its own critic context info.')
        for name in ('critic_context', 'next_critic_context'):
            if any(name not in info for info in infos):
                raise ValueError(f'Missing {name}; wrap the environment with CriticContextWrapper.')
            value = np.asarray([info[name] for info in infos], dtype=np.float32)
            if value.shape != (self.n_envs, CONTEXT_DIM) or not np.isfinite(value).all():
                raise ValueError(f'{name} has incompatible shape or nonfinite values.')
            contexts.append(value.copy())
        slot = self.pos
        super().add(obs, next_obs, action, reward, done, infos)
        self.critic_contexts[slot], self.next_critic_contexts[slot] = contexts

    def _get_samples(self, batch_inds, env=None):
        data = super()._get_samples(batch_inds, env=env)
        # TaskBalancedReplayBuffer explicitly restricts n_envs to one.
        return AsymmetricReplaySamples(*data,
            self.to_torch(self.critic_contexts[batch_inds, 0]),
            self.to_torch(self.next_critic_contexts[batch_inds, 0]))


class AsymmetricPolicy(SACPolicy):
    """Standard public-input actor; independently constructed larger critics."""
    def __init__(self, *args, **kwargs):
        if kwargs.get('share_features_extractor', False):
            raise ValueError('Asymmetric critics cannot share actor feature extraction.')
        super().__init__(*args, **kwargs)

    def make_critic(self, features_extractor=None):
        if self.observation_space.shape != (PUBLIC_OBSERVATION_DIM,):
            raise ValueError('The asymmetric policy preserves the 100-value actor contract.')
        if features_extractor is not None:
            raise ValueError('Privileged critic features must remain separate from the actor.')
        space = spaces.Box(-np.inf, np.inf, (PUBLIC_OBSERVATION_DIM+CONTEXT_DIM,), np.float32)
        kwargs = dict(self.critic_kwargs)
        kwargs.update(observation_space=space, features_extractor=FlattenExtractor(space),
                      features_dim=PUBLIC_OBSERVATION_DIM+CONTEXT_DIM,
                      share_features_extractor=False)
        return ContinuousCritic(**kwargs).to(self.device)


class AsymmetricSAC(AnchoredSAC):
    def __init__(self, policy, *args, **kwargs):
        self.asymmetric_version = VERSION
        self.context_version, self.context_dim = CONTEXT_VERSION, CONTEXT_DIM
        if policy == 'MlpPolicy':
            policy = AsymmetricPolicy
        if policy is not AsymmetricPolicy:
            raise ValueError('AsymmetricSAC requires its separate public actor/private critic policy.')
        kwargs.setdefault('replay_buffer_class', AsymmetricReplayBuffer)
        super().__init__(policy, *args, **kwargs)

    def _setup_model(self):
        if (self.asymmetric_version != VERSION or self.context_version != CONTEXT_VERSION
                or self.context_dim != CONTEXT_DIM):
            raise ValueError('Saved asymmetric context contract differs; use a fresh transfer.')
        super()._setup_model()
        if not isinstance(self.replay_buffer, AsymmetricReplayBuffer):
            raise ValueError('Asymmetric learning requires context-bearing replay.')

    @classmethod
    def load(cls, path, *args, **kwargs):
        source = path
        position = path.tell() if hasattr(path, 'tell') else None
        if isinstance(path, (str, Path)):
            source = Path(path)
            if not source.exists():
                source = Path(str(source)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                metadata = json.loads(archive.read('data'))
                if (metadata.get('asymmetric_version') != VERSION
                        or metadata.get('context_version') != CONTEXT_VERSION
                        or metadata.get('context_dim') != CONTEXT_DIM):
                    raise ValueError('Asymmetric resume requires its saved context contract; use actor transfer for other models.')
        finally:
            if position is not None:
                path.seek(position)
        return super().load(path, *args, **kwargs)

    def load_replay_buffer(self, *args, **kwargs):
        previous = self.replay_buffer
        try:
            super().load_replay_buffer(*args, **kwargs)
            replay = self.replay_buffer
            if (not isinstance(replay, AsymmetricReplayBuffer)
                    or replay.context_version != CONTEXT_VERSION or replay.context_dim != CONTEXT_DIM
                    or replay.critic_contexts.shape != (replay.buffer_size, replay.n_envs, CONTEXT_DIM)
                    or replay.next_critic_contexts.shape != replay.critic_contexts.shape):
                raise ValueError('Replay lacks the saved asymmetric critic context contract.')
        except Exception:
            self.replay_buffer = previous
            raise

    def _critic_loss(self, data):
        discount = data.discounts if data.discounts is not None else self.gamma
        with torch.no_grad():
            actions, log_prob = self.actor.action_log_prob(data.next_observations)
            next_q = torch.cat(self.critic_target(
                critic_inputs(data.next_observations, data.next_critic_context), actions), dim=1)
            target = data.rewards+(1-data.dones)*discount*(
                next_q.min(dim=1, keepdim=True).values-self.ent_coef_tensor*log_prob.reshape(-1, 1))
        q = self.critic(critic_inputs(data.observations, data.critic_context), data.actions)
        return .5*sum(F.mse_loss(value, target) for value in q)

    def _calibrate_critics(self, gradient_steps, batch_size):
        """Fit value estimates only; actor/reference/actor-Adam remain untouched."""
        if self.ent_coef_optimizer is not None:
            raise ValueError('Asymmetric calibration retains the fixed source entropy coefficient.')
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.critic.optimizer])
        losses = []
        for _ in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss = self._critic_loss(data)
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
        self.logger.record('train/ent_coef', self.ent_coef_tensor.item())
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')

    def _actor_step(self, observations, coefficient, context=None):
        private_observations = critic_inputs(observations, context)
        mask = protected_observations(observations)
        reference = self._reference_parameters(observations[mask]) if mask.any() else None
        actions, log_prob = self.actor.action_log_prob(observations)
        q = torch.cat(self.critic(private_observations, actions), dim=1).min(dim=1).values
        rl_loss = (coefficient*log_prob-q).mean()
        penalty = self._protected_kl(observations, mask, reference)
        loss = rl_loss+self.anchor_coefficient*penalty
        parameters = {key: value.detach().clone() for key, value in self.actor.state_dict().items()}
        optimizer_state = deepcopy(self.actor.optimizer.state_dict())
        self.actor.optimizer.zero_grad()
        loss.backward()
        self.actor.optimizer.step()
        with torch.no_grad():
            after = self._protected_kl(observations, mask, reference)
            valid = all(torch.isfinite(p).all().item() for p in self.actor.parameters())
            accepted = bool(valid and torch.isfinite(after).item() and after.item() <= self.anchor_max_kl)
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
        return {'loss':loss.item(),'rl_loss':rl_loss.item(),'kl':self.last_anchor_kl,
                'protected_fraction':mask.float().mean().item(),'accepted':accepted}

    def _train_anchored(self, gradient_steps, batch_size):
        if self.ent_coef_optimizer is not None:
            raise ValueError('Asymmetric SAC retains fixed source entropy.')
        self.policy.set_training_mode(True)
        self._update_learning_rate([self.actor.optimizer, self.critic.optimizer])
        losses, actor_stats = [], []
        for step in range(gradient_steps):
            data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)
            loss = self._critic_loss(data)
            self.critic.optimizer.zero_grad()
            loss.backward()
            self.critic.optimizer.step()
            losses.append(loss.item())
            actor_stats.append(self._actor_step(data.observations, self.ent_coef_tensor, data.critic_context))
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/critic_loss', np.mean(losses))
        self.logger.record('train/ent_coef', self.ent_coef_tensor.item())
        for key in ('loss','rl_loss','kl','protected_fraction','accepted'):
            self.logger.record('train/anchor_'+key, np.mean([row[key] for row in actor_stats]))
        self.logger.record('train/anchor_rejected_total', self.anchor_rejected)


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    """Preserve the current public actor/original reference; reset all learning."""
    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(path) as archive:
        saved = json.loads(archive.read('data'))
        has_reference = 'reference_actor.pth' in archive.namelist()
    source_class = AsymmetricSAC if 'asymmetric_version' in saved else (AnchoredSAC if has_reference else SAC)
    old = source_class.load(path, device='cpu')
    if old.observation_space.shape != (100,) or env.observation_space.shape != (100,):
        raise ValueError('Asymmetric transfer requires the unchanged current 100-value actor contract.')
    if old.use_sde or old.action_space != env.action_space:
        raise ValueError('Asymmetric transfer requires identical Gaussian action semantics.')
    if buffer_size < unified_policy.REFILL_TRANSITIONS:
        raise ValueError('Replay capacity must cover the transfer refill period.')
    coefficient = float(old.log_ent_coef.detach().exp().item()
                        if old.log_ent_coef is not None else old.ent_coef_tensor.item())
    model = AsymmetricSAC('MlpPolicy',env,device=device,seed=seed,
        learning_rate=unified_policy.CRITIC_LEARNING_RATE,buffer_size=buffer_size,
        learning_starts=0,batch_size=old.batch_size,tau=old.tau,gamma=old.gamma,
        train_freq=(1,'step'),gradient_steps=1,ent_coef=coefficient,
        target_update_interval=old.target_update_interval,tensorboard_log=str(tensorboard_log),
        policy_kwargs={'net_arch':old.actor.net_arch,'activation_fn':old.actor.activation_fn})
    model.actor.load_state_dict(old.actor.state_dict())
    model.reference_actor.load_state_dict((old.reference_actor if has_reference else old.actor).state_dict())
    for group in model.actor.optimizer.param_groups:
        group['lr'] = model.actor_learning_rate
    metadata = {'steps':old.num_timesteps,'updates':old._n_updates,
        'imitation_updates':getattr(old,'imitation_updates',0),'transfer':'actor_only',
        'fresh_critics':True,'fresh_optimizers':True,'entropy_coefficient':coefficient,
        'entropy_mode':'fixed_at_source_value','source_observation_dim':100,'observation_dim':100,
        'source_has_frozen_reference':has_reference,'source_sha256':source_hash,
        'replay_refill_transitions':model.refill_transitions,'critic_calibration_updates':model.calibration_updates,
        'actor_learning_rate':model.actor_learning_rate,'critic_learning_rate':model.critic_learning_rate,
        'policy_update_rule':VERSION,'critic_context_version':CONTEXT_VERSION,'critic_context_dim':CONTEXT_DIM,
        'reference':'original_frozen_reference_from_source_checkpoint' if has_reference else 'frozen_inherited_actor',
        'anchor_coefficient':model.anchor_coefficient,'anchor_max_batch_mean_kl':model.anchor_max_kl,
        'anchor_mask':'public single-pass task, front visible, or front seen within 1.8 seconds',
        'replay_sampling':'50% single-pass and 50% lap transitions when both available',
        'actor_receives_privileged_context':False,'new_demonstrations':0}
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Source checkpoint changed during transfer.')
    return model, metadata
