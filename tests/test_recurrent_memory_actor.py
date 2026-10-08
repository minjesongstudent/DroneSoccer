from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from stable_baselines3 import SAC

import src.recurrent_memory_actor as module
import src.recurrent_ring_position as motion_module
from src.recurrent_memory_actor import (RecurrentMemoryActor, RecurrentMemoryPolicy,
                                        build_features, checkpoint_manifest)
from src.recurrent_ring_position import RingPositionEstimate, RingPositionSpec


SOURCE = Path(__file__).resolve().parents[1]/'trainedModels/seenflag-s1-level1.zip'


@pytest.fixture(scope='module')
def source():
    return SAC.load(SOURCE, device='cpu')


def feature_batch(count=12):
    rng = np.random.default_rng(800)
    public = rng.uniform(-1, 1, (count, 85)).astype(np.float32)
    estimates = rng.normal(size=(count, 64)).astype(np.float32)
    features = np.concatenate((public, estimates, np.ones((count, 1), np.float32)), axis=1)
    return torch.from_numpy(public), torch.from_numpy(features)


def test_hidden_feature_builder_masks_startup_and_excludes_output_predictions():
    public = feature_batch(1)[0][0].numpy()
    hidden = np.linspace(-.8,.8,64,dtype=np.float32).reshape(1,1,64)
    for count in (1,2,3,19):
        snapshot = RingPositionEstimate(public, np.ones(3,np.float32), np.ones(3,np.float32),count,count>=3)
        combined = build_features(snapshot,hidden)
        np.testing.assert_array_equal(combined[:85],public)
        np.testing.assert_array_equal(combined[85:149],hidden.reshape(64) if count>=3 else np.zeros(64))
        assert combined[149] == (count>=3)
        changed = replace(snapshot,position_mm=np.full(3,999,np.float32),standardized_position=np.full(3,-100,np.float32))
        np.testing.assert_array_equal(build_features(changed,hidden),combined)
    with pytest.raises(ValueError): build_features(snapshot,hidden.astype(np.float64))
    with pytest.raises(ValueError): build_features(snapshot,np.zeros((1,1,3),np.float32))
    with pytest.raises(ValueError): build_features(replace(snapshot,valid_history=False),hidden)
    from src.estimated_ring_position_actor import checkpoint_manifest as position_manifest
    assert checkpoint_manifest()!=position_manifest()
    assert checkpoint_manifest()['input_width']==150


def test_initial_distribution_deterministic_and_seeded_stochastic_action_parity(source):
    actor = RecurrentMemoryActor(source.actor)
    public, features = feature_batch()
    for old, new in zip(source.actor.get_action_dist_params(public)[:2],
                        actor.get_action_dist_params(features)[:2]):
        assert torch.equal(old, new)
    for deterministic in (True, False):
        torch.manual_seed(871)
        expected = source.actor(public, deterministic=deterministic)
        torch.manual_seed(871)
        actual = actor(features, deterministic=deterministic)
        assert torch.equal(actual, expected)


def test_nonzero_linear_memory_head_has_exact_zero_memory_parity(source):
    actor = RecurrentMemoryActor(source.actor)
    public,features = feature_batch()
    with torch.no_grad():
        actor.residual_weight[1,0]=.3
        actor.residual_weight[2,63]=-.2
    original_mean,original_std,_=source.actor.get_action_dist_params(public)
    mean,std,_=actor.get_action_dist_params(features)
    expected=torch.zeros_like(mean)
    expected[:,1]=.3*features[:,85]
    expected[:,2]=-.2*features[:,148]
    torch.testing.assert_close(mean,original_mean+expected,rtol=0,atol=1e-7)
    assert torch.equal(std,original_std)
    for validity in (0.,1.):
        masked=features.clone(); masked[:,85:149]=0; masked[:,149]=validity
        assert torch.equal(actor(masked,deterministic=True),source.actor(public,deterministic=True))


def test_only_residual_gets_gradient_and_base_stays_frozen_in_train_mode(source):
    actor = RecurrentMemoryActor(source.actor).train()
    _, features = feature_batch()
    assert not actor.base_actor.training
    trainable = [(name, parameter) for name, parameter in actor.named_parameters() if parameter.requires_grad]
    assert [name for name, _ in trainable] == ['residual_weight']
    assert trainable[0][1].numel() == 256
    mean, _, _ = actor.get_action_dist_params(features)
    mean.sum().backward()
    assert actor.residual_weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in actor.base_actor.parameters())
    optimizer = torch.optim.Adam([actor.residual_weight], lr=1e-5)
    optimizer.step()
    actor.assert_source_unchanged()
    with torch.no_grad():
        next(actor.base_actor.parameters()).add_(.01)
    with pytest.raises(ValueError, match='Frozen source'):
        actor.assert_source_unchanged()


def test_log_probability_interface_matches_existing_onpolicy_helper(source):
    from tools.run_onpolicy_trial import action_log_prob
    actor = RecurrentMemoryActor(source.actor)
    _, features = feature_batch()
    actions, log_prob = actor.action_log_prob(features)
    assert actions.shape == (12, 4) and log_prob.shape == (12,)
    torch.testing.assert_close(actor.log_prob(features, actions), action_log_prob(actor, features, actions),
                               rtol=0, atol=0)


@pytest.mark.parametrize('bad', ['width85', 'float64', 'nan', 'valid_half', 'unmasked_startup'])
def test_malformed_feature_contract_rejected(source, bad):
    actor = RecurrentMemoryActor(source.actor)
    _, features = feature_batch()
    if bad == 'width85': features = features[:, :85]
    elif bad == 'float64': features = features.double()
    elif bad == 'nan': features[0, 86] = float('nan')
    elif bad == 'valid_half': features[0, 149] = .5
    elif bad == 'unmasked_startup': features[0, 149] = 0
    with pytest.raises(ValueError): actor(features)


@pytest.mark.parametrize('cudnn,matmul', [(True, False), (False, True), (True, True), (False, False)])
def test_explicit_cuda_precision_contract_checks_flags_without_mutating_them(monkeypatch, cudnn, matmul):
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', cudnn)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', matmul)
    module._require_precision(torch.device('cpu'))
    if cudnn or matmul:
        with pytest.raises(ValueError, match='strict_float32'):
            module._require_precision(torch.device('cuda:0'))
    else:
        module._require_precision(torch.device('cuda:0'))
    assert torch.backends.cudnn.allow_tf32 == cudnn
    assert torch.backends.cuda.matmul.allow_tf32 == matmul
    assert checkpoint_manifest()['inference_contract'] == 'strict_float32'


@pytest.fixture
def estimator_checkpoint(tmp_path, monkeypatch):
    # A tiny frozen test artifact, explicitly registered only inside this test.
    model = motion_module._FrozenGRU64()
    path = tmp_path/'estimator.pt'
    torch.save(dict(model=model.state_dict(), normalizers=dict(
        x_mean=np.zeros(85, np.float32), x_scale=np.ones(85, np.float32),
        y_mean=np.array([1, 2, 3], np.float32), y_scale=np.array([2, 3, 4], np.float32))), path)
    spec = RingPositionSpec(module.ESTIMATOR_PROFILE, hashlib.sha256(path.read_bytes()).hexdigest())
    registry = {module.ESTIMATOR_PROFILE: spec}
    monkeypatch.setattr(motion_module, 'RING_POSITION_SPECS', registry)
    monkeypatch.setattr(module, 'RING_POSITION_SPECS', registry)
    return path


def test_policy_recurrence_advances_only_observe_and_reset_isolates_episode(estimator_checkpoint):
    policy = RecurrentMemoryPolicy(SOURCE, estimator_checkpoint, device='cpu')
    observations = feature_batch(5)[0].numpy()
    first = policy.observe(observations[0]); assert policy.ring_position.frame_count == 1
    assert np.array_equal(first[85:], np.zeros(65))
    for _ in range(3): policy.predict(first)
    assert policy.ring_position.frame_count == 1
    second = policy.observe(observations[1]); assert second[149] == 0
    third = policy.observe(observations[2]); assert third[149] == 1
    np.testing.assert_array_equal(third[85:149], policy.ring_position.hidden_state.reshape(64))
    cached = third.copy()
    policy.observe(observations[3]); np.testing.assert_array_equal(third, cached)
    policy.reset()
    np.testing.assert_array_equal(policy.observe(observations[0]), first)
    np.testing.assert_array_equal(policy.observe(observations[1]), second)
    np.testing.assert_array_equal(policy.observe(observations[2]), third)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required for streaming comparison')
def test_real_frozen_hidden_stream_cpu_cuda_and_reset(monkeypatch):
    """No simulator or commands: replay public inputs to verify the new memory contract."""
    checkpoint = Path('D:/DroneSoccerData/runs/ring-position-estimator-20261008/gru64.pt')
    baseline = SOURCE.parents[2]/'lab-data/policy-validation-20261006/seenflag-s1-level1/aligned/9410000.npz'
    monkeypatch.setattr(torch.backends.cudnn, 'allow_tf32', False)
    monkeypatch.setattr(torch.backends.cuda.matmul, 'allow_tf32', False)
    cpu = RecurrentMemoryPolicy(SOURCE,checkpoint,device='cpu')
    gpu = RecurrentMemoryPolicy(SOURCE,checkpoint,device='cuda:0')
    with np.load(baseline,allow_pickle=False) as data:
        public = data['actor_observations'].copy()
    assert len(public)==600
    cpu_rows=[]; gpu_rows=[]
    for row in public:
        cpu_rows.append(cpu.observe(row)); gpu_rows.append(gpu.observe(row))
    left,right=np.asarray(cpu_rows),np.asarray(gpu_rows)
    np.testing.assert_array_equal(left[:,:85],public)
    np.testing.assert_array_equal(right[:,:85],public)
    np.testing.assert_allclose(left[:,85:149],right[:,85:149],rtol=0,atol=1e-6)
    np.testing.assert_array_equal(left[:,149],right[:,149])
    assert max(np.abs(left[:,85:149]).max(),np.abs(right[:,85:149]).max())<=1
    for policy,expected in ((cpu,left),(gpu,right)):
        policy.reset()
        for row,want in zip(public[:10],expected[:10]):
            np.testing.assert_array_equal(policy.observe(row),want)
        assert policy.ring_position.frame_count==10


def test_wrapper_predict_matches_actual_sb3_predict_conversion_and_rng(source, estimator_checkpoint):
    policy = RecurrentMemoryPolicy(SOURCE, estimator_checkpoint, device='cpu')
    public, features = feature_batch()
    for single in (True, False):
        old = public.numpy()[0] if single else public.numpy()
        new = features.numpy()[0] if single else features.numpy()
        for deterministic in (True, False):
            torch.manual_seed(865)
            expected, _ = source.predict(old, deterministic=deterministic)
            torch.manual_seed(865)
            actual, _ = policy.predict(new, deterministic=deterministic)
            np.testing.assert_array_equal(actual, expected)


def test_checkpoint_restores_head_identity_and_actions_without_restoring_episode_state(tmp_path, estimator_checkpoint):
    policy = RecurrentMemoryPolicy(SOURCE, estimator_checkpoint, device='cpu')
    with torch.no_grad(): policy.actor.residual_weight[0, 0] = .1
    features = feature_batch()[1].numpy()
    expected, _ = policy.predict(features)
    path = tmp_path/'candidate.npz'; digest = policy.save(path)
    loaded = RecurrentMemoryPolicy.load(path, SOURCE, estimator_checkpoint, device='cpu', sha256=digest)
    assert loaded.ring_position.frame_count == 0
    np.testing.assert_array_equal(loaded.predict(features)[0], expected)
    loaded.actor.assert_source_unchanged()
    assert checkpoint_manifest()['profile'] != 'four-action'
    with pytest.raises(FileExistsError): policy.save(path)
    with pytest.raises(ValueError, match='checksum'):
        RecurrentMemoryPolicy.load(path, SOURCE, estimator_checkpoint, device='cpu', sha256='0'*64)


@pytest.mark.parametrize('bad', ['four_action_identity', 'source_hash', 'nan_weight', 'object_weight'])
def test_safe_load_rejects_incompatible_identity_or_invalid_weights(tmp_path, estimator_checkpoint, bad):
    manifest = checkpoint_manifest()
    weights = np.zeros((4, 64), np.float32)
    if bad == 'four_action_identity': manifest['profile'] = 'four-action-150'
    elif bad == 'source_hash': manifest['source_sha256'] = '0'*64
    elif bad == 'nan_weight': weights[0, 0] = np.nan
    elif bad == 'object_weight': weights = np.array([dict(unexpected=True)], dtype=object)
    path = tmp_path/'bad.npz'
    np.savez(path, manifest=np.frombuffer(json.dumps(manifest).encode(), np.uint8), residual_weight=weights)
    with pytest.raises(ValueError):
        RecurrentMemoryPolicy.load(path, SOURCE, estimator_checkpoint, device='cpu',
                                   sha256=hashlib.sha256(path.read_bytes()).hexdigest())
