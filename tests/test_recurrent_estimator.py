import copy
import json
import time

import numpy as np
import pytest
import torch

from src.camera_history import CameraHistory
from tools.run_recurrent_estimator import (
    PROFILE, Episode, BoundedStop, FeedForwardEstimator, RecurrentEstimator,
    episode_arrays, load_episodes, fit_normalizers, episode_schedule, make_batch,
    masked_mse, prediction_metrics, gate_estimator, ensure_budget, evaluate_estimator,
)


def saved_trace(n=12):
    rng = np.random.default_rng(99)
    actions = rng.uniform(-.5, .5, (n, 4)).astype(np.float32)
    detections = rng.normal(0, .1, (n, 8)).astype(np.float32)
    detections[:, 6] = np.arange(n) % 2
    detections[:, 7] = 0
    history = CameraHistory(PROFILE.lags)
    observations = np.array([history.update(detections[t], np.zeros(4, np.float32) if t == 0 else actions[t-1])
                             for t in range(n)])
    t = np.arange(n)
    positions = np.column_stack((t*t*.001, t*.002, t*.003))
    trace = dict(actor_observations=observations, actions=actions, positions=positions,
                 headings=t*.03, terminated=np.r_[np.zeros(n-1, bool), True], truncated=np.zeros(n, bool))
    return trace


def as_episode(trace=None, seed=11000000, index=0):
    trace = saved_trace() if trace is None else trace
    x, y, mask, visible = episode_arrays(trace)
    return Episode(x, y, mask, visible, seed, index, 'test-fixture')


def test_preaction_target_alignment_and_body_axes():
    trace = saved_trace(4)
    trace['positions'] = np.array([[0., 0., 0.], [.001, .002, .003], [.005, .006, .007], [9., 9., 9.]])
    trace['headings'] = np.array([9., 0., np.pi/2, 9.])
    x, y, mask, visible = episode_arrays(trace)
    np.testing.assert_array_equal(x, trace['actor_observations'])
    np.testing.assert_array_equal(mask, [False, False, True, True])
    np.testing.assert_array_equal(y[:2], np.zeros((2, 3)))
    np.testing.assert_allclose(y[2:], [[1., -2., 3.], [4., 4., 4.]], atol=1e-6)
    np.testing.assert_array_equal(visible, x[:, 70] > .5)


def test_current_and_future_truth_do_not_change_current_input_or_target():
    original = saved_trace()
    changed = copy.deepcopy(original)
    changed['positions'][7:] += 100
    changed['headings'][7:] += 2
    x, y, _, _ = episode_arrays(original)
    altered_x, altered_y, _, _ = episode_arrays(changed)
    np.testing.assert_array_equal(x, altered_x)
    np.testing.assert_array_equal(y[:8], altered_y[:8])
    assert not np.array_equal(y[8:], altered_y[8:])


def test_current_action_and_later_actions_are_not_current_estimator_inputs():
    original = saved_trace()
    changed = copy.deepcopy(original)
    changed['actions'][7:] *= -1
    history = CameraHistory(PROFILE.lags)
    for t in range(len(changed['actions'])):
        changed['actor_observations'][t] = history.update(original['actor_observations'][t, 64:72],
            np.zeros(4, np.float32) if t == 0 else changed['actions'][t-1])
    x, y, _, _ = episode_arrays(original)
    altered_x, altered_y, _, _ = episode_arrays(changed)
    np.testing.assert_array_equal(x[:8], altered_x[:8])
    np.testing.assert_array_equal(y, altered_y)
    assert not np.array_equal(x[8:], altered_x[8:])


def test_history_verifier_rejects_unissued_command_and_cross_reset_carry():
    trace = saved_trace()
    trace['actor_observations'][3, 80] += .01
    with pytest.raises(ValueError, match='history mismatch'):
        episode_arrays(trace)
    trace = saved_trace()
    trace['actor_observations'][0, 72] = .5
    with pytest.raises(ValueError, match='history mismatch'):
        episode_arrays(trace)


def test_recurrent_prefix_causality_and_automatic_episode_reset():
    torch.manual_seed(17)
    model = RecurrentEstimator().eval()
    x = torch.randn(1, 12, 85)
    changed = x.clone(); changed[:, 7:] += 10
    with torch.no_grad():
        original = model(x, torch.tensor([12]))
        later_changed = model(changed, torch.tensor([12]))
        prefix = model(x[:, :7], torch.tensor([7]))
        model(torch.randn(1, 20, 85)*100, torch.tensor([20]))
        reset_again = model(x, torch.tensor([12]))
        streaming, state = model.stream(x[:, :5])
        continued, _ = model.stream(x[:, 5:], state)
    torch.testing.assert_close(original[:, :7], later_changed[:, :7], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original[:, :7], prefix, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(original, reset_again, atol=0, rtol=0)
    torch.testing.assert_close(original, torch.cat([streaming, continued], 1), atol=1e-6, rtol=1e-6)
    assert not model.gru.bidirectional


def test_masked_first_rows_still_influence_recurrent_state():
    torch.manual_seed(11)
    model = RecurrentEstimator()
    x = torch.randn(1, 7, 85, requires_grad=True)
    y = torch.zeros(1, 7, 3)
    mask = torch.tensor([[False, False, True, True, True, True, True]])
    loss = masked_mse(model(x, torch.tensor([7])), y, mask)
    loss.backward()
    assert x.grad[0, :2].abs().sum() > 0


@pytest.mark.parametrize('model_type', [FeedForwardEstimator, RecurrentEstimator])
def test_padding_cannot_change_real_predictions_or_masked_loss(model_type):
    torch.manual_seed(10)
    episodes = [as_episode(saved_trace(5)), as_episode(saved_trace(12), index=1)]
    normalizers = fit_normalizers(episodes)
    x, y, mask, lengths = make_batch(episodes, [0, 1], normalizers)
    changed_x, changed_y = x.clone(), y.clone()
    changed_x[0, 5:] = 999
    changed_y[~mask] = 999
    model = model_type().eval()
    predicted = model(x, lengths)
    changed = model(changed_x, lengths)
    torch.testing.assert_close(predicted[mask], changed[mask], atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(masked_mse(predicted, y, mask), masked_mse(changed, changed_y, mask))
    single_x, _, _, single_lengths = make_batch(episodes, [0], normalizers)
    torch.testing.assert_close(predicted[0, :5], model(single_x, single_lengths)[0], atol=1e-6, rtol=1e-6)


def test_fit_normalizers_exclude_heldout_padding_and_masked_targets():
    episodes = [as_episode(saved_trace(5)), as_episode(saved_trace(12), index=1)]
    expected = fit_normalizers(episodes[:1])
    episodes[1].observations[:] = 100000
    episodes[1].targets[:] = 100000
    episodes[0].targets[:2] = -100000
    actual = fit_normalizers(episodes[:1])
    for key in expected:
        np.testing.assert_array_equal(actual[key], expected[key])
    np.testing.assert_allclose(actual['x_mean'], episodes[0].observations.mean(0), atol=1e-7)
    np.testing.assert_allclose(actual['y_mean'], episodes[0].targets[2:].mean(0), atol=1e-6)


def test_shared_episode_schedule_is_fixed_and_fit_only():
    fit = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    schedule = episode_schedule(fit, 30)
    np.random.seed(222); np.random.randn(100)
    np.testing.assert_array_equal(schedule, episode_schedule(fit, 30))
    assert schedule.shape == (30, 8)
    assert set(schedule.ravel()).issubset(set(fit))


def test_evaluation_excludes_padding_and_first_two_targets(tmp_path):
    episodes = [as_episode(saved_trace(5)), as_episode(saved_trace(9), index=1)]
    normalizers = fit_normalizers(episodes)
    model = FeedForwardEstimator()
    destination = tmp_path/'predictions.npz'
    metrics = evaluate_estimator(model, episodes, np.array([0, 1]), normalizers, destination, 'cpu')
    assert metrics['episodes'] == 2 and metrics['all']['samples'] == 10
    with np.load(destination) as saved:
        assert len(saved['predictions']) == 14  # no padded rows are written
        assert saved['mask'].sum() == 10
        manual = prediction_metrics(saved['predictions'], saved['targets'], saved['mask'], saved['visible'])
        assert metrics['all'] == manual['all']
        altered = saved['predictions'].copy()
        altered[~saved['mask']] = 100000
        assert prediction_metrics(altered, saved['targets'], saved['mask'], saved['visible']) == manual


def _gate_metrics(ratio=.85):
    metrics = {}
    for variant, factor in [('feedforward', 1.), ('gru64', ratio)]:
        metrics[variant] = {}
        for cohort in ('fit', 'heldout', 'gae_separate'):
            metrics[variant][cohort] = {group: dict(samples=10, rmse_mm_per_axis=10.*factor)
                                      for group in ('all', 'blind', 'visible')}
    return metrics


def test_gate_requires_both_cohorts_and_all_fixed_conditions():
    results = _gate_metrics()
    decision = gate_estimator(results, complete=True, source_unchanged=True)
    assert decision['passed'] and not decision['actor_integration_approved']
    assert not decision['acceptance'] and not decision['policy_promoted']
    results['gru64']['gae_separate']['blind']['rmse_mm_per_axis'] = 8.6
    assert not gate_estimator(results, complete=True, source_unchanged=True)['passed']
    results = _gate_metrics()
    results['gru64']['heldout']['visible']['rmse_mm_per_axis'] = 10.51
    assert not gate_estimator(results, complete=True, source_unchanged=True)['passed']


def test_gate_rejects_smoke_incomplete_source_change_and_empty_stratum():
    results = _gate_metrics()
    assert not gate_estimator(results, complete=True, source_unchanged=True, smoke=True)['passed']
    assert not gate_estimator({}, complete=False, source_unchanged=True)['passed']
    assert not gate_estimator(results, complete=True, source_unchanged=False)['passed']
    results['gru64']['heldout']['blind']['samples'] = 0
    assert not gate_estimator(results, complete=True, source_unchanged=True)['passed']


def test_stop_file_and_total_deadline_are_enforced(tmp_path):
    ensure_budget(time.monotonic()+10, tmp_path/'STOP')
    with pytest.raises(BoundedStop, match='wall-clock'):
        ensure_budget(time.monotonic()-1, tmp_path/'STOP')
    (tmp_path/'STOP').write_text('stop')
    with pytest.raises(BoundedStop, match='STOP file'):
        ensure_budget(time.monotonic()+10, tmp_path/'STOP')


def test_loader_keeps_only_complete_episodes_before_warmup_boundary(tmp_path):
    rows = []
    for index, n in enumerate((4, 5)):
        name = f'{index}.npz'
        np.savez(tmp_path/name, **saved_trace(n))
        rows.append(dict(seed=11000000+index, trace=name, steps=n, status='complete'))
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    episodes, records, hashes = load_episodes(tmp_path, warmup_limit=7)
    assert len(episodes) == 1 and records[0]['targets'] == 2
    assert str(tmp_path/'0.npz') in hashes and str(tmp_path/'1.npz') not in hashes
    altered = saved_trace(4); altered['terminated'][:] = False
    np.savez(tmp_path/'0.npz', **altered)
    with pytest.raises(ValueError, match='complete episode'):
        load_episodes(tmp_path, warmup_limit=7)
