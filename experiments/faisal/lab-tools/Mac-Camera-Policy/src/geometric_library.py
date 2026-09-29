"""Adapt authentic snapshots only where no outside-crossing history is missing."""
from src.metric_return_library import MetricReturnLibrary
from src.return_geometry import GeometricReturnProgress,VERSION
class GeometricReturnLibrary(MetricReturnLibrary):
    reward_version=VERSION
    def restore(self,identifier):
        result=super().restore(identifier);base=result['environment'].unwrapped
        if (base.pos-base.hoop_centre)@base.hoop_normal<=0:
            raise ValueError('Geometric adaptation requires pre-crossing back-side snapshots')
        result['environment']=GeometricReturnProgress(result['environment'])
        return result
