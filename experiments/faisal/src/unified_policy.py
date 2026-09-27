"""Continue a saved policy with ordinary SAC and fresh online accounting."""
from stable_baselines3 import SAC


VERSION = 'one-policy-mixed-starts-v1'
REFILL_TRANSITIONS = 1000


class UnifiedSAC(SAC):
    """Use the student's policy to refill replay before ordinary SAC updates.

    No demonstrations or imitation loss are used. Keeping learning_starts=0
    prevents SB3 from replacing the inherited policy with random warmup actions.
    """

    def train(self, gradient_steps, batch_size=64):
        if self.replay_buffer.size() < REFILL_TRANSITIONS:
            return
        return super().train(gradient_steps, batch_size)


def start_from_checkpoint(path, env, *, device, seed, buffer_size, tensorboard_log):
    model = UnifiedSAC.load(path, env=env, device=device,
                            learning_starts=0, buffer_size=buffer_size,
                            tensorboard_log=str(tensorboard_log))
    source = {'steps': model.num_timesteps, 'updates': model._n_updates,
              'imitation_updates': getattr(model, 'imitation_updates', 0)}
    # SAC.load restores weights and optimizers but creates a new replay buffer.
    # Old transitions and counters must not be presented as this run's results.
    model.replay_buffer.reset()
    for name in ('num_timesteps', '_n_updates', '_episode_num',
                 '_num_timesteps_at_start', '_total_timesteps'):
        setattr(model, name, 0)
    for name in ('_last_obs', '_last_original_obs', '_last_episode_starts',
                 'ep_info_buffer', 'ep_success_buffer'):
        setattr(model, name, None)
    for name in ('demo_obs', 'demo_actions', 'imitation_updates'):
        model.__dict__.pop(name, None)
    model.seed = seed
    model.set_random_seed(seed)
    return model, source
