"""Trainable public-observation bypass; recurrent actor remains the action source."""
import torch
from torch import nn
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy

class PublicBypassPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, **kwargs):
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        if self.features_dim != 119 or self.shared_lstm:
            raise ValueError('Requires119 public features and separate actor/critic recurrence')
        projection=nn.Linear(self.features_dim,self.lstm_actor.hidden_size,bias=False)
        nn.init.zeros_(projection.weight)
        self.lstm_actor.add_module('public_skip',projection)
        # Base construction preceded this module; ensure its parameters are optimized.
        self.optimizer=self.optimizer_class(self.parameters(),lr=lr_schedule(1),**self.optimizer_kwargs)

    @staticmethod
    def _process_sequence(features,lstm_states,episode_starts,lstm):
        output,state=RecurrentActorCriticPolicy._process_sequence(features,lstm_states,episode_starts,lstm)
        if hasattr(lstm,'public_skip'):
            output=output+lstm.public_skip(features)
        return output,state

def add_public_bypass(model):
    """Explicit architecture migration; preserve old tensors, initialize only skip to0."""
    if isinstance(model.policy,PublicBypassPolicy):raise ValueError('Bypass already installed')
    policy=PublicBypassPolicy(model.observation_space,model.action_space,model.lr_schedule,**model.policy_kwargs).to(model.device)
    result=policy.load_state_dict(model.policy.state_dict(),strict=False)
    if result.missing_keys!=['lstm_actor.public_skip.weight'] or result.unexpected_keys:
        raise ValueError(f'Unexpected migration mismatch: {result}')
    model.policy=policy
    model.policy_class=PublicBypassPolicy
    return model
