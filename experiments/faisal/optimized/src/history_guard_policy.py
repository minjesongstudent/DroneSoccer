"""Experimental fixed-history drift gate for reward-driven actor proposals.

Saved public observations constrain updates; they never provide flight actions.
The bank is FIT-only and the frozen reference is the trial's starting actor.
This limits mean KL on that bank, not unseen-state drift or flight failures.
"""
from copy import deepcopy
import torch
from src.geometric_policy import GeometricSAC
from src.anchored_policy import gaussian_kl

class HistoryGuardSAC(GeometricSAC):
    def __init__(self,*args,**kwargs):
        self.history_max_kl=.01
        self.history_rejected=0
        self.last_history_kl=0.
        self.history_observations=torch.empty((0,119))
        super().__init__(*args,**kwargs)

    def set_history_bank(self,observations):
        bank=torch.as_tensor(observations,dtype=torch.float32,device=self.device)
        if bank.ndim!=2 or bank.shape[1]!=119 or not len(bank) or not torch.isfinite(bank).all():
            raise ValueError('Finite nonempty119D public observations required')
        self.history_observations=bank.detach().clone()

    def _excluded_save_params(self):
        return super()._excluded_save_params()+['history_observations']

    def _get_torch_save_params(self):
        states,variables=super()._get_torch_save_params()
        return states,variables+['history_observations']

    @torch.no_grad()
    def history_kl(self):
        if not len(self.history_observations):raise ValueError('Configure history bank before actor updates')
        total=self.history_observations.new_zeros(())
        for observations in self.history_observations.split(1024):
            mean,std,_=self.actor.get_action_dist_params(observations)
            ref,rstd,_=self.reference_actor.get_action_dist_params(observations)
            total+=gaussian_kl(mean,std,ref,rstd).sum()
        return float(total/len(self.history_observations))

    def _actor_step(self,observations,coefficient):
        if not len(self.history_observations):raise ValueError('Missing history bank')
        before={k:v.detach().clone() for k,v in self.actor.state_dict().items()}
        optimizer=deepcopy(self.actor.optimizer.state_dict())
        stats=super()._actor_step(observations,coefficient)
        if stats['accepted']:
            drift=self.history_kl()
            if not torch.isfinite(torch.tensor(drift)) or drift>self.history_max_kl:
                self.actor.load_state_dict(before)
                self.actor.optimizer.load_state_dict(optimizer)
                self.actor.optimizer.zero_grad(set_to_none=True)
                self.actor_updates-=1;self.anchor_accepted-=1;self.anchor_rejected+=1
                self.history_rejected+=1
                stats['accepted']=False
                mask=self._protection_mask(observations)
                ref=self._reference_parameters(observations[mask]) if mask.any() else None
                with torch.no_grad():self.last_anchor_kl=float(self._protected_kl(observations,mask,ref))
                stats['kl']=self.last_anchor_kl
            self.last_history_kl=self.history_kl()
        return stats
