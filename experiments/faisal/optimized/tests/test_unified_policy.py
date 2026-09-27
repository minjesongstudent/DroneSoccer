import json
import threading
import time
from dataclasses import asdict

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor

from src.bootstrap import GuidedSAC
from src.curriculum import REWARD_VERSION
from src.unified_curriculum import MixedLessonEnv
from src.unified_policy import UnifiedSAC, start_from_checkpoint
from src import unified_policy, live


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    env = Monitor(MixedLessonEnv())
    model = GuidedSAC('MlpPolicy', env, device='cpu', seed=0,
                      buffer_size=1200, learning_starts=0, policy_kwargs={'net_arch': [16, 16]})
    model.learn(2)
    model.num_timesteps = 520_000
    model.imitation_updates = 49_999
    model.save(tmp_path/'source.zip')
    config = {'reward_version': REWARD_VERSION,
              'geometry': {k: asdict(getattr(env.unwrapped, k))
                           for k in ('cage', 'drone', 'hoop', 'camera', 'pad')}}
    (tmp_path/'source.json').write_text(json.dumps(config))
    yield model, tmp_path/'source.zip', tmp_path/'source.json'
    env.close()


def test_checkpoint_keeps_policy_but_resets_accounting_and_uses_no_imitation(source, tmp_path, monkeypatch):
    old, checkpoint, _ = source
    env = Monitor(MixedLessonEnv())
    obs, _ = env.reset(seed=17)
    expected = old.predict(obs, deterministic=True)[0]
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 16)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 4)
    model, history = start_from_checkpoint(checkpoint, env, device='cpu', seed=7,
                                           buffer_size=1200, tensorboard_log=tmp_path/'logs')
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], expected)
    assert {k: history[k] for k in ('steps', 'updates', 'imitation_updates')} == {
        'steps': 520_000, 'updates': 2, 'imitation_updates': 49_999}
    assert history['transfer'] == 'actor_only' and history['fresh_critics']
    assert model.num_timesteps == model._n_updates == model.replay_buffer.size() == 0
    assert not isinstance(model, GuidedSAC)
    assert not hasattr(model, 'demo_obs') and not hasattr(model, 'imitation_updates')
    assert not model.actor.optimizer.state_dict()['state']
    model.learn(15, reset_num_timesteps=False)
    assert model._n_updates == 0
    np.testing.assert_array_equal(model.predict(obs, deterministic=True)[0], expected)
    model.learn(5, reset_num_timesteps=False)
    assert model._n_updates == 5 and model.num_timesteps == 20
    assert model.critic_calibration_updates == 4 and model.actor_updates == 1
    assert not hasattr(model, 'imitation_updates')
    env.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA integration check')
def test_live_save_and_disk_resume_keep_one_cuda_policy(source, tmp_path, monkeypatch):
    _, checkpoint, config = source
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    # The live runner now preserves a source snapshot for each new experiment.
    for directory in ('src', 'tests'):
        (tmp_path/directory).mkdir()
        (tmp_path/directory/'fixture.py').write_text('# snapshot fixture\n')
    # Smaller refill for this lifecycle test; the separate test checks its semantics.
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 16)
    monkeypatch.setattr(live, 'REFILL_TRANSITIONS', 16)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 4)
    session = live.Session(24, 'smoke', source_checkpoint=checkpoint,
                           source_config=config, buffer_size=1200, eval_episodes=1)
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    deadline = time.monotonic()+60
    try:
        while session.state['mode'] not in ('Replay', 'Error') and time.monotonic() < deadline:
            time.sleep(.05)
        assert session.state['mode'] == 'Replay', session.state.get('error', session.state)
    finally:
        session.closed.set()
        session.wake.set()
        thread.join(5)
    assert not thread.is_alive()
    assert session.state['source_checkpoint_steps'] == 520_000
    assert session.state['steps'] == 24
    assert (session.out/'evaluations'/'0.json').exists()
    assert (session.out/'evaluations'/'24.json').exists()
    assert (session.out/'source_snapshot'/'src'/'fixture.py').read_text() == '# snapshot fixture\n'
    assert not (session.out/'demonstrations.npz').exists()
    restored = live.Session(32, 'smoke', resume=True)
    env = Monitor(live.MissionObservation(live.LessonEnv()))
    saved = restored.load(env)
    assert saved['status']['steps'] == 24
    assert restored.target() == 32 and restored.model.num_timesteps == 24
    assert restored.model.replay_buffer.size() == 24
    assert restored.model._n_updates == session.model._n_updates == 9
    assert restored.model.critic_calibration_updates == 4
    assert restored.model.actor_updates == 5
    assert type(restored.model) is live.UnifiedSAC
    assert next(restored.model.policy.parameters()).device.type == 'cuda'
    assert restored.curriculum.stage == 4 and env.unwrapped.next_focus_stage == 4
    obs, _ = env.reset(seed=19)
    np.testing.assert_array_equal(restored.model.predict(obs, deterministic=True)[0],
                                  session.model.predict(obs, deterministic=True)[0])
    assert abs(sum(x['probability'] for x in restored.state['training_mix'])-1) < 1e-8
    env.close()
