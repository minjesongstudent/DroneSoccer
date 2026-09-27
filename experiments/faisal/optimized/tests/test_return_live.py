"""Bounded live-run lifecycle checks for authentic return practice.

The CUDA smoke uses existing local source/bank artifacts when present. It
creates only a temporary 24-transition run, and reduces refill/calibration and
development evaluation counts only inside this test process.
"""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from stable_baselines3.common.monitor import Monitor

from src import live, unified_policy, return_evaluation
from src.return_mixture import ReturnMixtureEnv
from src.return_practice import ReturnSnapshotLibrary
from src.return_selection import ReturnSelectionConfig, ReturnSkillSelector


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def artifacts():
    paths = {'checkpoint': ROOT/'runs/return-route-40k-20260927/best_model.zip',
             'config': ROOT/'runs/return-route-40k-20260927/config.json',
             'train': ROOT/'return-libraries/train-v1',
             'development': ROOT/'return-libraries/development-v1'}
    required = [paths['checkpoint'], paths['config'], paths['train']/'manifest.json',
                paths['development']/'manifest.json']
    if not all(path.exists() for path in required):
        pytest.skip('Local completed policy and frozen return banks are required for this integration smoke.')
    return paths


def assert_state_dict_equal(left, right):
    assert left.keys() == right.keys()
    for key in left:
        a, b = left[key], right[key]
        if torch.is_tensor(a):
            # PyTorch may relocate Adam's scalar step on CUDA restore.
            assert torch.equal(a.detach().cpu(), b.detach().cpu()), key
        elif isinstance(a, dict):
            assert_state_dict_equal(a, b)
        elif isinstance(a, list):
            assert len(a) == len(b)
            for aa, bb in zip(a, b):
                if isinstance(aa, dict):
                    assert_state_dict_equal(aa, bb)
                else:
                    assert aa == bb
        else:
            assert a == b


def selector_report(step, *, episodes=50, bank='a'*64):
    return {'step': step, 'suites': {
        'original_centered': {'success_rate': .96}, 'varied_approach': {'success_rate': .92},
        'return_practice': {'episodes': episodes, 'successful_episodes': 0,
                            'safe_front_return_episodes': 0, 'crash_rate': .2}},
        'protocols': {'return_practice': {'id': 'fixed-test-return', 'bank_sha256': bank,
                                         'seeds': list(range(8_500_000, 8_500_000+episodes))}}}


def stopped_selector():
    selector = ReturnSkillSelector()
    selector.observe(selector_report(0))
    selector.observe(selector_report(20_000))
    assert selector.should_stop and selector.stop_reason == 'return_no_sustained_progress'
    return selector


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA live integration')
@pytest.mark.parametrize('asymmetric,retrace,recent,front,resets,repair,metric', [(False,False,False,False,False,False,False),(True,False,False,False,False,False,False),(False,True,False,False,False,False,False),(False,True,True,False,False,False,False),(False,True,True,True,False,False,False),(False,True,True,True,True,False,False),(False,True,True,True,True,True,False),(False,True,True,True,True,False,True)])
def test_cuda_return_run_preserves_histories_and_resumes_actor_reference_replay_selector(
        artifacts, tmp_path, monkeypatch, asymmetric, retrace, recent, front, resets, repair, metric):
    torch.set_num_threads(1)
    if resets:
        from src import return_reset_curriculum
        monkeypatch.setattr(return_reset_curriculum, 'EPISODES', 2)
    library_class = ReturnSnapshotLibrary
    if front:
        from src.front_return_library import FrontReturnLibrary
        library_class = FrontReturnLibrary
        artifacts['train'] = Path('D:/DroneSoccerData/return-libraries/front-memory-train-v1')
        artifacts['development'] = Path('D:/DroneSoccerData/return-libraries/front-memory-development-v1')
    if metric:
        from src.metric_return_library import MetricReturnLibrary
        library_class=MetricReturnLibrary
    bank = library_class(artifacts['train'])
    assert bank.manifest['split'] == 'train'
    assert set(bank.phases) == {'return_side', 'return_front'}
    immutable = [artifacts['checkpoint'], artifacts['train']/'manifest.json',
                 artifacts['development']/'manifest.json']
    hashes = {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable}
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    for directory in ('src', 'tests'):
        (tmp_path/directory).mkdir()
        (tmp_path/directory/'fixture.py').write_text('# live smoke fixture\n')
    monkeypatch.setattr(unified_policy, 'REFILL_TRANSITIONS', 8)
    monkeypatch.setattr(live, 'REFILL_TRANSITIONS', 8)
    from src import front_policy
    monkeypatch.setattr(front_policy, 'REFILL_TRANSITIONS', 8)
    from src import metric_policy
    monkeypatch.setattr(metric_policy, 'REFILL_TRANSITIONS', 8)
    monkeypatch.setattr(unified_policy, 'CALIBRATION_UPDATES', 4)
    original_evaluate = return_evaluation.evaluate_return_practice

    def bounded_return_evaluation(model, library, progress=None):
        return original_evaluate(model, library, episodes=2, progress=progress)

    monkeypatch.setattr(return_evaluation, 'evaluate_return_practice', bounded_return_evaluation)
    transitions = []
    original_callback = live.LiveCallback._on_step

    def record_transition(callback):
        transitions.append((deepcopy(callback.locals['infos'][0]), bool(callback.locals['dones'][0])))
        return original_callback(callback)

    monkeypatch.setattr(live.LiveCallback, '_on_step', record_transition)
    runs_dir = tmp_path/'separate-run-storage'
    session = live.Session(24, 'return-smoke', source_checkpoint=artifacts['checkpoint'],
                           source_config=artifacts['config'], buffer_size=128, eval_episodes=1,
                           return_library=artifacts['train'], return_validation=artifacts['development'],
                           runs_dir=runs_dir, asymmetric_critic=asymmetric, retrace_critic=retrace,
                           recent_retrace=recent, front_memory=front, reset_lessons=resets, repair_anchor=repair, metric_ring=metric)
    assert session.out == runs_dir/'return-smoke'
    session.return_selector = ReturnSkillSelector(ReturnSelectionConfig(episodes=2, improvement_episodes=1))
    thread = threading.Thread(target=session.run, daemon=True)
    thread.start()
    deadline = time.monotonic()+90
    try:
        while session.state['mode'] not in ('Replay', 'Error') and time.monotonic() < deadline:
            time.sleep(.05)
        assert session.state['mode'] == 'Replay', session.state.get('error', session.state)
    finally:
        session.closed.set()
        session.wake.set()
        thread.join(10)
    assert not thread.is_alive()
    assert len(transitions) == session.model.num_timesteps == 24
    assert session.model._n_updates == 17 and session.model.critic_calibration_updates == 4
    assert session.model.actor_attempts == 13
    assert session.model.actor_updates+session.model.anchor_rejected == 13
    assert session.model.observation_space.shape == ((119,) if metric else (115,) if front else (100,))
    assert next(session.model.actor.parameters()).device.type == 'cuda'
    practice = [info for info, _ in transitions if info.get('return_practice')]
    if not resets:
        assert practice and all(i['historical_passes'] >= 1 and not i['full_mission_success'] for i in practice)
        assert practice[0]['passes'] == 0  # Captured historical passage is not credited.
    previous, credited = 0, 0
    for info, done in transitions:
        credited += max(0, info['passes']-previous)
        previous = 0 if done else info['passes']
    assert session.state['total_passes'] == credited
    config_path = session.out/'config.json'
    config = json.loads(config_path.read_text())
    assert config['asymmetric_critic'] is asymmetric
    assert config['retrace_critic'] is retrace
    if asymmetric or retrace:
        assert config['actor_uses_simulator_context'] is False
    assert config['return_library_sha256'] == bank.bank_sha256
    assert (session.out/'best_return_model.zip').exists()
    assert session.return_selector.to_dict()['history'][-1]['step'] == 24
    restored = live.Session(32, 'return-smoke', resume=True, buffer_size=128, runs_dir=runs_dir)
    if resets:
        from src.return_reset_curriculum import ResetMixture
        env=Monitor(ResetMixture(restored.reset_scheduler,metric_ring=metric))
    else:
        env = Monitor(ReturnMixtureEnv(bank, observation_wrapper=restored.observation_wrapper))
    if asymmetric:
        from src.critic_context import CriticContextWrapper
        env = CriticContextWrapper(env)
    saved = restored.load(env)
    assert restored.asymmetric_critic is asymmetric
    assert restored.retrace_critic is retrace
    if resets:
        assert restored.reset_scheduler.state()==session.reset_scheduler.state()
        assert env.unwrapped is not None
        assert restored.reset_lessons
    assert restored.metric_ring is metric
    assert restored.repair_anchor is repair
    if repair:
        from src.front_policy import FrontMemorySAC
        with pytest.raises(ValueError,match='explicit policy class'):
            FrontMemorySAC.load(session.out/'model.zip',device='cuda')
        assert restored.model.anchor_repairs==session.model.anchor_repairs
        with pytest.raises(ValueError,match='repair contract'):
            live.Session(32,'return-smoke',resume=True,buffer_size=128,runs_dir=runs_dir,repair_anchor=False)
    assert restored.front_memory is front
    assert restored.recent_retrace is recent
    if recent:
        from src.recent_retrace import RecentLapReplay, VERSION as recent_version
        assert isinstance(restored.model.replay_buffer, RecentLapReplay)
        assert config['recent_sampling_version'] == recent_version
        assert config['recent_lap_count'] == 2048
        with pytest.raises(ValueError, match='sampling cannot change'):
            live.Session(32, 'return-smoke', resume=True, buffer_size=128,
                         runs_dir=runs_dir, recent_retrace=False)
    assert saved['status']['steps'] == restored.model.num_timesteps == 24
    assert restored.target() == 32 and restored.last_evaluated == 24
    assert restored.return_selector.to_dict() == session.return_selector.to_dict()
    assert_state_dict_equal(restored.model.actor.state_dict(), session.model.actor.state_dict())
    assert_state_dict_equal(restored.model.reference_actor.state_dict(), session.model.reference_actor.state_dict())
    assert_state_dict_equal(restored.model.actor.optimizer.state_dict(), session.model.actor.optimizer.state_dict())
    assert_state_dict_equal(restored.model.critic.optimizer.state_dict(), session.model.critic.optimizer.state_dict())
    for name in ('observations', 'next_observations', 'actions', 'rewards', 'dones', 'timeouts', '_single_pass_slots'):
        np.testing.assert_array_equal(getattr(restored.model.replay_buffer, name),
                                      getattr(session.model.replay_buffer, name))
    assert restored.model.replay_buffer.size() == 24
    if asymmetric:
        for name in ('critic_contexts', 'next_critic_contexts'):
            np.testing.assert_array_equal(getattr(restored.model.replay_buffer, name),
                                          getattr(session.model.replay_buffer, name))
        with pytest.raises(ValueError, match='Critic architecture cannot change'):
            live.Session(32, 'return-smoke', resume=True, buffer_size=128,
                         runs_dir=runs_dir, asymmetric_critic=False)
    assert restored.model.actor_attempts == session.model.actor_attempts
    observation, info = env.reset(seed=7)
    if not resets:assert info['return_practice'] and info['passes'] == 0 and info['historical_passes'] >= 1
    np.testing.assert_array_equal(restored.model.predict(observation, deterministic=True)[0],
                                  session.model.predict(observation, deterministic=True)[0])
    # Resume rejects bank/config drift before loading any model or replay.
    for field, message in [('return_library_sha256', 'Return bank changed'),
                           ('return_validation_sha256', 'Return bank changed'),
                           ('reward_version', 'Reward version differs'),
                           ('observation_version', 'Observation version differs')]:
        changed = deepcopy(config)
        changed[field] = '0'*64 if field.endswith('sha256') else 'incompatible'
        config_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match=message):
            live.Session(32, 'return-smoke', resume=True, buffer_size=128, runs_dir=runs_dir).load(env)
    config_path.write_text(json.dumps(config))
    if asymmetric:
        changed = deepcopy(config)
        changed['critic_context_version'] = 'incompatible'
        config_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError, match='Critic context contract differs'):
            live.Session(32, 'return-smoke', resume=True, buffer_size=128, runs_dir=runs_dir).load(env)
        config_path.write_text(json.dumps(config))
    assert {path: hashlib.sha256(path.read_bytes()).hexdigest() for path in immutable} == hashes
    env.close()


@pytest.mark.parametrize('command', ['resume', 'extend'])
def test_return_selector_stop_latches_live_controls(tmp_path, monkeypatch, command):
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    session = live.Session(20_000, 'return-stop', return_library=tmp_path/'train',
                           return_validation=tmp_path/'development')
    session.return_selector = stopped_selector()
    session.stop.set()
    session.update(mode='Replay', steps=20_000)
    before = deepcopy(session.state)
    assert not session.guard.should_stop  # Return selector independently blocks resume.
    with pytest.raises(ValueError, match='Evaluation stopped this experiment'):
        session.control(command)
    assert session.state == before and session.stop.is_set() and not session.wake.is_set()


@pytest.mark.parametrize('field,value,match', [
    ('observation_version', 'incompatible-camera-inputs', 'Source observation meaning differs'),
    ('reward_version', 'unknown-reward', 'Source policy geometry or reward version is incompatible')])
def test_new_return_session_rejects_incompatible_source_config(artifacts, tmp_path, monkeypatch,
                                                               field, value, match):
    monkeypatch.setattr(live, 'ROOT', tmp_path)
    config = json.loads(artifacts['config'].read_text())
    config[field] = value
    source_config = tmp_path/'source.json'
    source_config.write_text(json.dumps(config))
    session = live.Session(1, 'invalid-source', source_checkpoint=artifacts['checkpoint'],
                           source_config=source_config, return_library=artifacts['train'],
                           return_validation=artifacts['development'])
    session.run()
    assert session.state['mode'] == 'Error' and match in session.state['error']
    assert session.model is None and not (session.out/'initial_model.zip').exists()
