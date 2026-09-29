"""Public stateless adaptation of unchanged authentic115D snapshots."""
from src.front_return_library import FrontReturnLibrary
from src.ring_estimate import MetricRingObservation, VERSION as ESTIMATE_VERSION
from src.front_return_library import OBSERVATION_VERSION as FRONT_VERSION

OBSERVATION_VERSION=FRONT_VERSION+'+'+ESTIMATE_VERSION

class MetricReturnLibrary(FrontReturnLibrary):
    def __init__(self,directory):
        # Validate the original115D payload and its original source hashes first.
        super().__init__(directory)
        self.source_observation_version=self.observation_version
        self.observation_version=OBSERVATION_VERSION
        self.observation_dim=119

    def restore(self,identifier):
        result=super().restore(identifier)
        result['environment']=MetricRingObservation(result['environment'])
        result['observation']=result['environment'].observation(result['observation'])
        return result
