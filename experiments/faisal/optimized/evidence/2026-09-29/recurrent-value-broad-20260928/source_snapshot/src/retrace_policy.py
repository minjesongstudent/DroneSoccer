"""Experimental anchored SAC with short, importance-corrected soft traces.

The actor and critics both use the unchanged 100 public inputs. Replay stores
the exact collection density and pre-tanh sample only to correct old actions
during critic learning. This is an empirical neural/partial-observation
adaptation of Retrace, without a convergence or flight-completion guarantee.
"""
import json
from pathlib import Path
import zipfile

import numpy as np
import torch
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update

from src.anchored_policy import AnchoredSAC, start_from_checkpoint as transfer_actor
from src.retrace_replay import RetraceReplayBuffer, VERSION as REPLAY_VERSION, BEHAVIOR_DENSITY_VERSION
from src.retrace_targets import soft_retrace_targets, VERSION as TARGET_VERSION

VERSION = 'public-soft-retrace-anchored-sac-v1'
TRACE_LENGTH = 8
TRACE_LAMBDA = .95


class RetraceSAC(AnchoredSAC):
    observation_dim = 100

    def __init__(self, *args, **kwargs):
        if kwargs.get('use_sde', False):
            raise ValueError('Retrace requires Gaussian sampling without gSDE.')
        self._trace_defaults()
        kwargs.setdefault('replay_buffer_class', RetraceReplayBuffer)
        kwargs.setdefault('replay_buffer_kwargs', {'sequence_length': TRACE_LENGTH})
        super().__init__(*args, **kwargs)

    def _trace_defaults(self):
        self.retrace_version = VERSION
        self.retrace_target_version = TARGET_VERSION
        self.retrace_replay_version = REPLAY_VERSION
        self.behavior_density_version = BEHAVIOR_DENSITY_VERSION
        self.trace_length, self.trace_lambda = TRACE_LENGTH, TRACE_LAMBDA
        self._pending_behavior = None

    def _validate_collection(self):
        if (self.n_envs != 1 or self.learning_starts != 0 or self.use_sde
                or self.action_noise is not None or self._vec_normalize_env is not None
                or self.observation_space.shape != (self.observation_dim,)
                or self.action_space.shape != (4,)
                or not np.all(self.action_space.low == -1)
                or not np.all(self.action_space.high == 1)):
            raise ValueError('Retrace requires one unnormalized 100D environment, normalized four actions, '
                             'policy-only refill, no action noise and no gSDE.')

    def _setup_model(self):
        if (self.retrace_version != VERSION or self.retrace_target_version != TARGET_VERSION
                or self.retrace_replay_version != REPLAY_VERSION
                or self.behavior_density_version != BEHAVIOR_DENSITY_VERSION
                or self.trace_length != TRACE_LENGTH or self.trace_lambda != TRACE_LAMBDA):
            raise ValueError('Retrace contract differs; use a new experiment.')
        super()._setup_model()
        self._validate_collection()
        if (not isinstance(self.replay_buffer, RetraceReplayBuffer)
                or self.replay_buffer.sequence_length != self.trace_length):
            raise ValueError('Retrace requires sequence replay with behavior densities.')

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['_pending_behavior']

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
                saved = json.loads(archive.read('data'))
                if (saved.get('retrace_version') != VERSION
                        or saved.get('retrace_target_version') != TARGET_VERSION
                        or saved.get('retrace_replay_version') != REPLAY_VERSION
                        or saved.get('behavior_density_version') != BEHAVIOR_DENSITY_VERSION
                        or saved.get('trace_length') != TRACE_LENGTH
                        or saved.get('trace_lambda') != TRACE_LAMBDA):
                    raise ValueError('Retrace resume requires its complete saved target/replay contract.')
        finally:
            if position is not None:
                path.seek(position)
        return super().load(path, *args, **kwargs)

    def load_replay_buffer(self, *args, **kwargs):
        previous = self.replay_buffer
        try:
            super().load_replay_buffer(*args, **kwargs)
            replay = self.replay_buffer
            if (not isinstance(replay, RetraceReplayBuffer)
                    or replay.sequence_length != self.trace_length
                    or replay.replay_version != REPLAY_VERSION
                    or replay.behavior_density_version != BEHAVIOR_DENSITY_VERSION
                    or replay.behavior_log_probs.shape != (replay.buffer_size, 1)
                    or replay.behavior_gaussian_actions.shape != (replay.buffer_size, 1, 4)):
                raise ValueError('Retrace cannot resume replay without its behavior-density sequence contract.')
            replay.break_stream()
            self._pending_behavior = None
            self._last_obs = None
        except Exception:
            self.replay_buffer = previous
            raise

    def _setup_learn(self, total_timesteps, callback=None, reset_num_timesteps=True,
                     tb_log_name='run', progress_bar=False):
        self._validate_collection()
        if reset_num_timesteps or self._last_obs is None:
            self.replay_buffer.break_stream()
            self._pending_behavior = None
        return super()._setup_learn(total_timesteps, callback, reset_num_timesteps,
                                   tb_log_name, progress_bar)

    def _sample_action(self, learning_starts, action_noise=None, n_envs=1):
        self._validate_collection()
        if learning_starts != 0 or action_noise is not None or n_envs != 1:
            raise ValueError('Unsupported behavior distribution for Retrace replay.')
        if self._pending_behavior is not None or self._last_obs is None:
            raise RuntimeError('Behavior metadata must be consumed once per valid observation.')
        observation = np.asarray(self._last_obs, dtype=np.float32)
        if observation.shape != (1, self.observation_dim) or not np.isfinite(observation).all():
            raise ValueError('Invalid public actor observation during collection.')
        with torch.no_grad():
            action, log_prob = self.actor.action_log_prob(torch.as_tensor(observation, device=self.device))
            gaussian = self.actor.action_dist.gaussian_actions
            if gaussian is None:
                raise RuntimeError('Collector did not retain its pre-tanh sample.')
            action_array = action.cpu().numpy().copy()
            gaussian_array = gaussian.cpu().numpy().copy()
            log_array = log_prob.cpu().numpy().copy()
        if not all(np.isfinite(x).all() for x in (action_array, gaussian_array, log_array)):
            raise ValueError('Nonfinite behavior density or sample.')
        self._pending_behavior = (observation.copy(), action_array.copy(), gaussian_array, log_array)
        # Exact normalized Box: avoid an unscale/scale round-trip that would
        # change rounded tanh samples and their density relationship.
        return action_array, action_array.copy()

    def _store_transition(self, replay_buffer, buffer_action, new_obs, reward, dones, infos):
        record = self._pending_behavior
        if record is None:
            raise RuntimeError('Cannot store a transition without its original behavior sample.')
        observation, action, gaussian, log_prob = record
        if not np.array_equal(observation, self._last_obs) or not np.array_equal(action, buffer_action):
            raise ValueError('Behavior metadata does not match the stored observation/action.')
        enriched = [dict(info, behavior_log_prob=float(log_prob[index]),
                         behavior_gaussian_actions=gaussian[index].copy())
                    for index, info in enumerate(infos)]
        super()._store_transition(replay_buffer, buffer_action, new_obs, reward, dones, enriched)
        self._pending_behavior = None

    def collect_rollouts(self, *args, **kwargs):
        try:
            return super().collect_rollouts(*args, **kwargs)
        finally:
            if self._pending_behavior is not None:
                # SB3 callbacks can stop after env.step but before storage.
                # The cached previous observation is then stale. A new stream
                # preserves old bootstrap semantics without inventing a done.
                self._pending_behavior = None
                self._last_obs = None
                self._last_original_obs = None
                self.replay_buffer.break_stream()

    def _critic_loss(self, data):
        shape = data.seq_valid.shape
        with torch.no_grad():
            observation = data.seq_observations.reshape(-1, self.observation_dim)
            next_observation = data.seq_next_observations.reshape(-1, self.observation_dim)
            action = data.seq_actions.reshape(-1, 4)
            target_q = torch.cat(self.critic_target(observation, action), 1).min(1).values.reshape(shape)
            next_action, next_log_prob = self.actor.action_log_prob(next_observation)
            next_q = torch.cat(self.critic_target(next_observation, next_action), 1).min(1).values
            soft_value = (next_q-self.ent_coef_tensor*next_log_prob).reshape(shape)
            mean, log_std, _ = self.actor.get_action_dist_params(observation)
            distribution = self.actor.action_dist.proba_distribution(mean, log_std)
            log_prob = distribution.log_prob(action, data.seq_behavior_gaussian_actions.reshape(-1, 4)).reshape(shape)
            terminated = data.seq_dones.squeeze(-1).bool()
            truncated = data.seq_episode_ends.squeeze(-1).bool() & ~terminated
            target = soft_retrace_targets(data.seq_rewards.squeeze(-1), terminated, truncated,
                data.seq_valid.bool(), target_q, soft_value, data.seq_behavior_log_prob.squeeze(-1), log_prob,
                gamma=self.gamma, trace_lambda=self.trace_lambda).reshape(-1, 1)
        q = self.critic(data.observations, data.actions)
        return .5*sum(F.mse_loss(value, target) for value in q)

    def _calibrate_critics(self, gradient_steps, batch_size):
        if self.ent_coef_optimizer is not None:
            raise ValueError('Retrace retains fixed source entropy.')
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

    def _train_anchored(self, gradient_steps, batch_size):
        if self.ent_coef_optimizer is not None:
            raise ValueError('Retrace retains fixed source entropy.')
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
            actor_stats.append(self._actor_step(data.observations, self.ent_coef_tensor))
            if step % self.target_update_interval == 0:
                polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.)
        self._n_updates += gradient_steps
        self.logger.record('train/n_updates', self._n_updates, exclude='tensorboard')
        self.logger.record('train/critic_loss', np.mean(losses))
        self.logger.record('train/ent_coef', self.ent_coef_tensor.item())
        for key in ('loss', 'rl_loss', 'kl', 'protected_fraction', 'accepted'):
            self.logger.record('train/anchor_'+key, np.mean([row[key] for row in actor_stats]))
        self.logger.record('train/anchor_rejected_total', self.anchor_rejected)


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    model, metadata = transfer_actor(path, env, device=device, seed=seed,
                                    buffer_size=buffer_size, tensorboard_log=tensorboard_log)
    if model.num_timesteps or model._n_updates or model.replay_buffer.size():
        raise ValueError('Retrace transfer requires fresh critics/optimizers/replay.')
    model.__class__ = RetraceSAC
    model._trace_defaults()
    model._validate_collection()
    model.replay_buffer_class = RetraceReplayBuffer
    model.replay_buffer_kwargs = {'sequence_length': TRACE_LENGTH}
    model.replay_buffer = RetraceReplayBuffer(buffer_size, model.observation_space, model.action_space,
        device=model.device, n_envs=1, optimize_memory_usage=False, sequence_length=TRACE_LENGTH)
    metadata.update(policy_update_rule=VERSION, trace_length=TRACE_LENGTH, trace_lambda=TRACE_LAMBDA,
                    retrace_target_version=TARGET_VERSION, retrace_replay_version=REPLAY_VERSION,
                    behavior_density_version=BEHAVIOR_DENSITY_VERSION,
                    behavior_density='exact collected pre-tanh sample and squashed joint log density',
                    critic_observation_dim=100, actor_receives_privileged_context=False)
    return model, metadata
