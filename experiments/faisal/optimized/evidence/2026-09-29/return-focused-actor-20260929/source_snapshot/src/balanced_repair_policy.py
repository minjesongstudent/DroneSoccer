"""Controlled combination of gradient balancing and monotonic reference repair."""
import json,zipfile
from pathlib import Path
from src.balanced_anchor_policy import BalancedAnchorSAC,start_from_checkpoint as transfer
from src.repair_anchor_policy import accepts
VERSION='balanced-monotonic-reference-kl-repair-v1'
class BalancedRepairSAC(BalancedAnchorSAC):
    def __init__(self,*args,**kwargs):
        self.balanced_repair_version=VERSION;self.anchor_repairs=0
        super().__init__(*args,**kwargs)
    def _accept_anchor_proposal(self,before,after):
        return accepts(before,after,self.anchor_max_kl)
    def _actor_step(self,observations,coefficient):
        stats=super()._actor_step(observations,coefficient)
        if stats['accepted'] and stats['kl']>self.anchor_max_kl:self.anchor_repairs+=1
        return stats
    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                if json.loads(archive.read('data')).get('balanced_repair_version')!=VERSION:
                    raise ValueError('Balanced repair version missing or different')
        finally:
            if position is not None:path.seek(position)
        return super().load(path,*args,**kwargs)
def start_from_checkpoint(*args,**kwargs):
    model,metadata=transfer(*args,**kwargs)
    model.__class__=BalancedRepairSAC;model.balanced_repair_version=VERSION;model.anchor_repairs=0
    metadata.update(balanced_repair_version=VERSION,anchor_repair_minimum_fraction=.01)
    return model,metadata
