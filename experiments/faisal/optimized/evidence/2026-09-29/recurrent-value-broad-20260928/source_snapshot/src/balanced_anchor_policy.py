"""Experimental gradient balance; original reference and hard KL gate remain."""
from copy import deepcopy
import json
from pathlib import Path
import zipfile
import torch
from src.geometric_policy import GeometricSAC, start_from_checkpoint as transfer

VERSION = 'anchor-gradient-norm-at-most-rl-v1'


def balanced_gradients(rl, anchor):
    """Bound weighted anchor gradient norm by RL norm, without changing either."""
    rn = torch.stack([g.square().sum() for g in rl]).sum().sqrt()
    an = torch.stack([g.square().sum() for g in anchor]).sum().sqrt()
    scale = torch.minimum(torch.ones_like(rn), rn / an.clamp_min(1e-30))
    return [r + scale * a for r, a in zip(rl, anchor)], scale


class BalancedAnchorSAC(GeometricSAC):
    def __init__(self, *args, **kwargs):
        self.balanced_anchor_version = VERSION
        super().__init__(*args, **kwargs)

    @classmethod
    def load(cls, path, *args, **kwargs):
        position = path.tell() if hasattr(path, 'tell') else None
        source = path
        if isinstance(path, (str, Path)) and not Path(path).exists():
            source = Path(str(path) + '.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                data=json.loads(archive.read('data'))
                if cls is BalancedAnchorSAC and data.get('source_anchor_version') is not None:
                    raise ValueError('Source anchor archive requires its explicit policy class')
                if cls is BalancedAnchorSAC and data.get('balanced_repair_version') is not None:
                    raise ValueError('Balanced repair archive requires its explicit policy class')
                if data.get('balanced_anchor_version') != VERSION:
                    raise ValueError('Balanced anchor version missing or different')
        finally:
            if position is not None:
                path.seek(position)
        return super().load(path, *args, **kwargs)

    def _actor_step(self, observations, coefficient):
        mask = self._protection_mask(observations)
        reference = self._reference_parameters(observations[mask]) if mask.any() else None
        actions, log_prob = self.actor.action_log_prob(observations)
        q = torch.cat(self.critic(observations, actions), dim=1).min(dim=1).values
        rl_loss = (coefficient * log_prob - q).mean()
        penalty = self._protected_kl(observations, mask, reference)
        parameters = list(self.actor.parameters())
        rg = torch.autograd.grad(rl_loss, parameters, retain_graph=True, allow_unused=True)
        ag = (torch.autograd.grad(self.anchor_coefficient * penalty, parameters, allow_unused=True)
              if mask.any() else [None] * len(parameters))
        rg = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, rg)]
        ag = [torch.zeros_like(p) if g is None else g for p, g in zip(parameters, ag)]
        gradients, scale = balanced_gradients(rg, ag)
        before = deepcopy(self.actor.state_dict())
        optimizer = deepcopy(self.actor.optimizer.state_dict())
        self.actor.optimizer.zero_grad(set_to_none=True)
        for p, g in zip(parameters, gradients):
            p.grad = g.detach()
        self.actor.optimizer.step()
        with torch.no_grad():
            after = self._protected_kl(observations, mask, reference)
            valid = all(torch.isfinite(p).all().item() for p in parameters)
            accepted = bool(valid and torch.isfinite(after).item()
                            and self._accept_anchor_proposal(penalty.item(), after.item()))
        self.actor_attempts += 1
        if accepted:
            self.actor_updates += 1
            self.anchor_accepted += 1
            self.last_anchor_kl = after.item()
        else:
            self.actor.load_state_dict(before)
            self.actor.optimizer.load_state_dict(optimizer)
            self.actor.optimizer.zero_grad(set_to_none=True)
            self.anchor_rejected += 1
            self.last_anchor_kl = penalty.item()
        self.last_anchor_gradient_scale = scale.item()
        return dict(loss=(rl_loss + self.anchor_coefficient * scale * penalty).item(),
                    rl_loss=rl_loss.item(), kl=self.last_anchor_kl,
                    protected_fraction=mask.float().mean().item(), accepted=accepted,
                    before_kl=penalty.item(), anchor_gradient_scale=scale.item())


def start_from_checkpoint(*args, **kwargs):
    model, metadata = transfer(*args, **kwargs)
    model.__class__ = BalancedAnchorSAC
    model.balanced_anchor_version = VERSION
    metadata.update(balanced_anchor_version=VERSION)
    return model, metadata
