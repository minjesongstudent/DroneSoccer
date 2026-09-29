"""Verified source reference with standard weighted KL gradient and strict gate."""
import json,zipfile
from pathlib import Path
from src.source_anchor_policy import SourceAnchorSAC,start_from_checkpoint as transfer
from src.anchored_policy import AnchoredSAC
VERSION='source-reference-standard-weighted-gradient-v1'
class StandardSourceSAC(SourceAnchorSAC):
    _actor_step=AnchoredSAC._actor_step
    def __init__(self,*args,**kwargs):
        self.standard_source_version=VERSION
        self.anchor_gradient_mode='standard'
        super().__init__(*args,**kwargs)
    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                data=json.loads(archive.read('data'))
                if data.get('standard_source_version')!=VERSION or data.get('anchor_gradient_mode')!='standard':
                    raise ValueError('Standard source gradient contract missing or different')
        finally:
            if position is not None:path.seek(position)
        return super().load(path,*args,**kwargs)
def start_from_checkpoint(*args,**kwargs):
    model,metadata=transfer(*args,**kwargs)
    model.__class__=StandardSourceSAC;model.standard_source_version=VERSION
    model.anchor_gradient_mode='standard'
    metadata.update(standard_source_version=VERSION,anchor_gradient_mode='standard',gradient_balancing_active=False,policy_update_rule=VERSION)
    return model,metadata
