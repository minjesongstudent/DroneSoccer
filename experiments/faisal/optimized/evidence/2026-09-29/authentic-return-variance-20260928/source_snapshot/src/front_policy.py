"""Separate115D learned policy contract with zero-column source transfer."""
import json
from pathlib import Path
import zipfile

import torch
from torch import nn
from torch.nn import functional as F
from stable_baselines3.sac.policies import SACPolicy

from src.anchored_policy import AnchoredSAC, protected_observations
from src.recent_retrace import RecentRetraceSAC, RecentLapReplay
from src.unified_policy import CRITIC_LEARNING_RATE, REFILL_TRANSITIONS

VERSION = 'front-memory-recent-retrace-v1'


class SplitInputLinear(nn.Linear):
    """Keep the original100-column arithmetic while learning the added15."""
    def forward(self, inputs):
        return (F.linear(inputs[..., :100].contiguous(), self.weight[:, :100].contiguous(), self.bias)
                + F.linear(inputs[..., 100:].contiguous(), self.weight[:, 100:].contiguous()))


class FrontActorPolicy(SACPolicy):
    def make_actor(self, features_extractor=None):
        actor = super().make_actor(features_extractor)
        layer = actor.latent_pi[0]
        if not isinstance(layer, nn.Linear) or layer.in_features != 115:
            raise ValueError('Front actor requires115D linear input.')
        replacement = SplitInputLinear(115, layer.out_features, device=layer.weight.device)
        replacement.load_state_dict(layer.state_dict())
        actor.latent_pi[0] = replacement
        return actor


class FrontLapReplay(RecentLapReplay):
    observation_dim = 115


class FrontMemorySAC(RecentRetraceSAC):
    observation_dim = 115

    def __init__(self, *args, **kwargs):
        self.front_policy_version = VERSION
        kwargs.setdefault('replay_buffer_class', FrontLapReplay)
        super().__init__(*args, **kwargs)

    @staticmethod
    def _protection_mask(observations):
        if observations.ndim!=2 or observations.shape[1]!=115:
            raise ValueError('Front-memory policy requires115D observations.')
        return protected_observations(observations[:,:100])

    def _setup_model(self):
        super()._setup_model()
        if self.front_policy_version!=VERSION or not isinstance(self.replay_buffer,FrontLapReplay):
            raise ValueError('Front policy/replay contract differs.')

    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():
            source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                data=json.loads(archive.read('data'))
                if cls is FrontMemorySAC and data.get('repair_version') is not None:
                    raise ValueError('Repair checkpoint requires its explicit policy class')
                if data.get('front_policy_version')!=VERSION:
                    raise ValueError('Front policy requires its saved115D version.')
        finally:
            if position is not None: path.seek(position)
        return super().load(path,*args,**kwargs)

    def load_replay_buffer(self,*args,**kwargs):
        previous=self.replay_buffer
        try:
            super().load_replay_buffer(*args,**kwargs)
            if not isinstance(self.replay_buffer,FrontLapReplay) or self.replay_buffer.observation_dim!=115:
                raise ValueError('Front policy requires115D replay.')
        except Exception:
            self.replay_buffer=previous
            raise


def expanded_actor_state(destination, source):
    result=destination.state_dict()
    for name,value in source.state_dict().items():
        if name=='latent_pi.0.weight':
            if value.shape[1]!=100 or result[name].shape[1]!=115 or value.shape[0]!=result[name].shape[0]:
                raise ValueError('Only explicit100-to115 actor expansion is supported.')
            result[name].zero_()
            result[name][:,:100].copy_(value)
        else:
            if result[name].shape!=value.shape:
                raise ValueError('Source actor architecture differs.')
            result[name].copy_(value)
    return result


def start_from_checkpoint(path,env,*,device,seed,buffer_size,tensorboard_log):
    from src.return_practice import _sha
    path=Path(path)
    source_hash=_sha(path.read_bytes())
    old=AnchoredSAC.load(path,device='cpu')
    if old.observation_space.shape!=(100,) or env.observation_space.shape!=(115,):
        raise ValueError('Front transfer requires preserved100D source and genuine115D environment.')
    if old.use_sde or old.action_space!=env.action_space or buffer_size<REFILL_TRANSITIONS:
        raise ValueError('Unsupported action contract or insufficient refill capacity.')
    coefficient=float(old.ent_coef_tensor.item())
    model=FrontMemorySAC(FrontActorPolicy,env,device=device,seed=seed,
        learning_rate=CRITIC_LEARNING_RATE,buffer_size=buffer_size,learning_starts=0,
        batch_size=old.batch_size,tau=old.tau,gamma=old.gamma,train_freq=(1,'step'),gradient_steps=1,
        ent_coef=coefficient,target_update_interval=old.target_update_interval,
        tensorboard_log=str(tensorboard_log),
        policy_kwargs={'net_arch':old.actor.net_arch,'activation_fn':old.actor.activation_fn})
    model.actor.load_state_dict(expanded_actor_state(model.actor,old.actor))
    model.reference_actor.load_state_dict(expanded_actor_state(model.reference_actor,old.reference_actor))
    for group in model.actor.optimizer.param_groups: group['lr']=model.actor_learning_rate
    if _sha(path.read_bytes())!=source_hash:
        raise RuntimeError('Source changed during transfer.')
    return model,dict(steps=old.num_timesteps,updates=old._n_updates,
        imitation_updates=getattr(old,'imitation_updates',0),transfer='actor_only_zero_column_append',
        source_sha256=source_hash,source_observation_dim=100,observation_dim=115,
        source_has_frozen_reference=True,reference='original_frozen_reference_from_source_checkpoint',
        fresh_critics=True,fresh_optimizers=True,new_demonstrations=0,
        entropy_coefficient=coefficient,entropy_mode='fixed_at_source_value',
        replay_refill_transitions=model.refill_transitions,critic_calibration_updates=model.calibration_updates,
        actor_learning_rate=model.actor_learning_rate,critic_learning_rate=model.critic_learning_rate,
        policy_update_rule=VERSION,anchor_coefficient=model.anchor_coefficient,
        anchor_max_batch_mean_kl=model.anchor_max_kl,
        actor_receives_privileged_context=False,critic_observation_dim=115,
        replay_sampling='50% all single-pass;25% all lap;25% latest2048 lap insertions')
