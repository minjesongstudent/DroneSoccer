"""Anchor to verified source actor while archiving the older original reference."""
from copy import deepcopy
import json,zipfile
from pathlib import Path
import torch
from src.balanced_anchor_policy import BalancedAnchorSAC,start_from_checkpoint as transfer
VERSION='verified-source-actor-retention-v1'
class SourceAnchorSAC(BalancedAnchorSAC):
    def __init__(self,*args,**kwargs):
        self.source_anchor_version=VERSION
        super().__init__(*args,**kwargs)
    def _make_source_anchor(self):
        self.source_anchor_actor=deepcopy(self.actor)
        self.source_anchor_actor.set_training_mode(False)
        self.source_anchor_actor.requires_grad_(False)
    def _setup_model(self):
        super()._setup_model()
        self._make_source_anchor()
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['source_anchor_actor']
    def _get_torch_save_params(self):
        states,variables=super()._get_torch_save_params()
        return states+['source_anchor_actor'],variables
    def _reference_parameters(self,observations):
        with torch.no_grad():
            mean,log_std,_=self.source_anchor_actor.get_action_dist_params(observations)
        return mean,log_std
    @classmethod
    def load(cls,path,*args,**kwargs):
        position=path.tell() if hasattr(path,'tell') else None
        source=path
        if isinstance(path,(str,Path)) and not Path(path).exists():source=Path(str(path)+'.zip')
        try:
            with zipfile.ZipFile(source) as archive:
                data=json.loads(archive.read('data'))
                if cls is SourceAnchorSAC and data.get('standard_source_version') is not None:
                    raise ValueError('Standard source archive requires its explicit policy class')
                if (data.get('source_anchor_version')!=VERSION
                    or 'source_anchor_actor.pth' not in archive.namelist()
                    or 'reference_actor.pth' not in archive.namelist()):
                    raise ValueError('Source anchor version and BOTH frozen actors required')
        finally:
            if position is not None:path.seek(position)
        return super().load(path,*args,**kwargs)
def start_from_checkpoint(*args,**kwargs):
    model,metadata=transfer(*args,**kwargs)
    model.__class__=SourceAnchorSAC;model.source_anchor_version=VERSION
    model._make_source_anchor()
    metadata.update(source_anchor_version=VERSION,active_retention_reference='verified source actor',
                    original_reference_preserved=True,original_reference_used_for_penalty=False)
    return model,metadata
