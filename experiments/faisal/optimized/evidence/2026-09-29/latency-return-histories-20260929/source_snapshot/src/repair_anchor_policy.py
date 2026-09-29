"""Versioned monotonic repair of an already-infeasible batch KL constraint."""
import json,zipfile,math
from pathlib import Path
from src.front_policy import FrontMemorySAC, start_from_checkpoint as front_transfer

VERSION='monotonic-reference-kl-repair-v1'


def accepts(before,after,limit=.1):
    if not all(math.isfinite(x) and x>=0 for x in (before,after,limit)):
        return False
    return after<=limit or (before>limit and after<=.99*before)


class RepairFrontSAC(FrontMemorySAC):
    def __init__(self,*args,**kwargs):
        self.repair_version=VERSION;self.anchor_repairs=0
        super().__init__(*args,**kwargs)

    def _accept_anchor_proposal(self,before,after):
        return accepts(before,after,self.anchor_max_kl)

    def _actor_step(self,observations,coefficient):
        stats=super()._actor_step(observations,coefficient)
        if stats['accepted'] and stats['kl']>self.anchor_max_kl:
            self.anchor_repairs+=1
        return stats

    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                if json.loads(archive.read('data')).get('repair_version')!=VERSION:
                    raise ValueError('Repair-anchor contract missing or different')
        finally:
            if position is not None:path.seek(position)
        return super().load(path,*args,**kwargs)


def start_from_checkpoint(*args,**kwargs):
    model,metadata=front_transfer(*args,**kwargs)
    model.__class__=RepairFrontSAC
    model.repair_version=VERSION;model.anchor_repairs=0
    metadata.update(repair_version=VERSION,anchor_repair_minimum_fraction=.01,
                    anchor_original_reference_preserved=True)
    return model,metadata
