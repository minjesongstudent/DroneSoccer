from pathlib import Path
import numpy as np
import pytest
import torch
from stable_baselines3.common.logger import configure
from src import unified_policy
from src.anchored_policy import AnchoredSAC, protected_observations
from src.metric_policy import MetricRingSAC, MetricLapReplay, start_from_checkpoint
from src.front_memory import FrontMemoryObservation
from src.ring_estimate import MetricRingObservation
from src.mission_observation import MissionObservation
from src.guided_curriculum import GuidedLessonEnv
from src.recent_retrace import RecentRetraceSAC

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA transfer/lifecycle')
def test_cuda_zero_column_actor_reference_transfer_and_learning_resume(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    source=ROOT/'runs/return-route-40k-20260927/best_model.zip'
    if not source.exists(): pytest.skip('Preserved source required')
    monkeypatch.setattr(unified_policy,'REFILL_TRANSITIONS',8)
    monkeypatch.setattr(unified_policy,'CALIBRATION_UPDATES',4)
    env=MetricRingObservation(FrontMemoryObservation(MissionObservation(GuidedLessonEnv(stage=4),single_pass=False)))
    old=AnchoredSAC.load(source,device='cuda')
    model,metadata=start_from_checkpoint(source,env,device='cuda',seed=7,buffer_size=5000,tensorboard_log=tmp_path)
    assert metadata['source_observation_dim']==100 and metadata['observation_dim']==119
    assert model.replay_buffer.size()==0 and model._n_updates==0
    assert not model.actor.optimizer.state and not model.critic.optimizer.state
    for actor,previous in [(model.actor,old.actor),(model.reference_actor,old.reference_actor)]:
        assert torch.count_nonzero(actor.latent_pi[0].weight[:,100:])==0
        torch.testing.assert_close(actor.latent_pi[0].weight[:,:100],previous.latent_pi[0].weight,rtol=0,atol=0)
        observations=torch.randn(32,119,device='cuda')
        with torch.no_grad():
            mean,std,_=actor.get_action_dist_params(observations)
            expected_mean,expected_std,_=previous.get_action_dist_params(observations[:,:100].contiguous())
        torch.testing.assert_close(mean,expected_mean,rtol=0,atol=0)
        torch.testing.assert_close(std,expected_std,rtol=0,atol=0)
    # The task flag remains index99; appended cosine must never label a task.
    observations=torch.zeros(4,119,device='cuda')
    observations[0,99]=1
    observations[1,114]=1
    assert model._protection_mask(observations).tolist()==[True,False,False,False]
    frozen={k:v.clone() for k,v in model.reference_actor.state_dict().items()}
    model.set_logger(configure(folder=None,format_strings=[]))
    model.learn(24)
    assert model.critic_calibration_updates==4 and model.actor_attempts==13
    assert model.replay_buffer.observations.shape[-1]==119
    assert isinstance(model.replay_buffer,MetricLapReplay)
    for key,value in frozen.items(): torch.testing.assert_close(model.reference_actor.state_dict()[key],value,rtol=0,atol=0)
    model.save(tmp_path/'model')
    model.save_replay_buffer(tmp_path/'replay.pkl')
    restored=MetricRingSAC.load(tmp_path/'model',env=env,device='cuda')
    restored.load_replay_buffer(tmp_path/'replay.pkl')
    assert restored.replay_buffer.size()==24 and restored.actor_attempts==13
    obs,_=env.reset(seed=101)
    np.testing.assert_array_equal(restored.predict(obs,deterministic=True)[0],model.predict(obs,deterministic=True)[0])
    restored.set_logger(configure(folder=None,format_strings=[]))
    restored.learn(4,reset_num_timesteps=False)
    assert restored.num_timesteps==28
    with pytest.raises(ValueError):
        RecentRetraceSAC.load(tmp_path/'model',device='cuda')
    with pytest.raises(ValueError,match='saved119D'):
        MetricRingSAC.load(source,device='cuda')
    env.close()
