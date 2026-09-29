"""Explicit reward contract for the unchanged119D actor architecture."""
import json,zipfile
from pathlib import Path
from src.metric_policy import MetricRingSAC,start_from_checkpoint as metric_transfer
from src.return_geometry import VERSION
class GeometricSAC(MetricRingSAC):
    def __init__(self,*args,**kwargs):
        self.geometric_reward_version=VERSION
        super().__init__(*args,**kwargs)
    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                data=json.loads(archive.read('data'))
                if cls is GeometricSAC and data.get('balanced_anchor_version') is not None:
                    raise ValueError('Balanced checkpoint requires its explicit policy class')
                if data.get('geometric_reward_version')!=VERSION:
                    raise ValueError('Geometric reward contract missing or different')
        finally:
            if position is not None:path.seek(position)
        return super().load(path,*args,**kwargs)
def start_from_checkpoint(*args,**kwargs):
    model,metadata=metric_transfer(*args,**kwargs)
    model.__class__=GeometricSAC;model.geometric_reward_version=VERSION
    metadata.update(geometric_reward_version=VERSION)
    return model,metadata
