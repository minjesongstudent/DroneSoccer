from copy import deepcopy
import json

import pytest

from src.evaluation_guard import EvaluationGuard, GuardConfig, SUITES


def report(step, centered=.96, varied=.88, full=.04, delayed=0., crash=.8, lesson=0.):
    rates = (centered, varied, full, delayed, lesson)
    return {'step': step, 'episodes_per_suite': 25, 'seed_start': 5_000_000,
            'policy_mode': 'deterministic', 'suites': {
                name: {'success_rate': rate, 'crash_rate': crash}
                for name, rate in zip(SUITES, rates)}}


def initialized(config=None, **rates):
    guard = EvaluationGuard(config)
    guard.observe(report(0, **rates))
    return guard


def test_baseline_is_retained_even_if_it_fails_absolute_floors():
    guard = initialized(centered=.5, varied=.4)
    decision = guard.observe(report(10_000, centered=.69, varied=.60, full=1))
    assert not decision['eligible']
    assert decision['baseline_step'] == decision['best_step'] == 0
    assert guard.history[0]['selected']
    assert not guard.history[0]['eligible']


def test_retention_exact_boundaries_and_absolute_floors():
    guard = initialized()
    decision = guard.observe(report(10_000, centered=.84, varied=.76, full=.08))
    assert decision['eligible'] and decision['selected']
    assert guard.best_step == 10_000
    assert not guard.observe(report(20_000, centered=.8399, varied=.76, full=.9))['eligible']
    assert not guard.observe(report(30_000, centered=.84, varied=.7599, full=.9))['eligible']
    low = initialized(centered=.75, varied=.65)
    decision = low.observe(report(10_000, centered=.70, varied=.60, full=.08))
    assert decision['eligible'] and decision['selected']
    assert not low.observe(report(20_000, centered=.6999, varied=.60, full=.9))['eligible']
    assert not low.observe(report(30_000, centered=.70, varied=.5999, full=.9))['eligible']


def test_ranking_is_lexicographic_and_ties_keep_previous_checkpoint():
    guard = initialized()
    assert guard.observe(report(10_000, full=.08, delayed=0, crash=.99))['selected']
    assert guard.observe(report(20_000, full=.08, delayed=.04, crash=1))['selected']
    assert guard.observe(report(30_000, full=.08, delayed=.04, crash=.8))['selected']
    assert not guard.observe(report(40_000, full=.08, delayed=.04, crash=.8, lesson=1))['selected']
    assert not guard.observe(report(50_000, full=.04, delayed=1, crash=0))['selected']
    assert not guard.observe(report(60_000, full=.08, delayed=0, crash=0))['selected']
    assert guard.best_step == 30_000


def test_severe_regression_starts_at_warmup_and_is_strictly_greater_than_drop():
    guard = initialized()
    assert guard.observe(report(10_000, centered=.1))['regression_streak'] == 0
    assert guard.observe(report(20_000, centered=.76, varied=.68))['regression_streak'] == 0
    assert guard.observe(report(30_000, centered=.759))['regression_streak'] == 1
    assert guard.observe(report(40_000, varied=.679))['regression_streak'] == 2
    decision = guard.observe(report(50_000, centered=.759))
    assert decision['regression_streak'] == 3
    assert decision['should_stop'] and decision['stop_reason'] == 'severe_regression'


def test_regression_must_be_consecutive_and_stop_latches():
    guard = initialized()
    guard.observe(report(20_000, centered=0))
    guard.observe(report(30_000, centered=0))
    assert guard.observe(report(40_000))['regression_streak'] == 0
    for step in (50_000, 60_000, 70_000):
        decision = guard.observe(report(step, varied=0))
    assert decision['should_stop']
    recovered = guard.observe(report(80_000, full=.5))
    assert recovered['should_stop'] and recovered['stop_reason'] == 'severe_regression'


def test_stagnation_boundary_uses_full_success_not_delay_or_crash_improvement():
    guard = initialized()
    assert not guard.observe(report(99_999, delayed=.5, crash=.1))['should_stop']
    decision = guard.observe(report(100_000, delayed=.5, crash=.1))
    assert decision['should_stop'] and decision['stop_reason'] == 'stagnation'
    improving = initialized()
    improving.observe(report(20_000, full=.08))
    assert not improving.observe(report(100_000, full=.08))['should_stop']


def test_ineligible_full_success_does_not_disable_stagnation():
    guard = initialized()
    decision = guard.observe(report(100_000, centered=.8, full=.5))
    assert not decision['eligible']
    assert not decision['full_task_improved']
    assert decision['stop_reason'] == 'stagnation'


def test_candidate_requires_three_consecutive_checks_and_only_requests_independent_test():
    guard = initialized(centered=.9, varied=.9)
    good = dict(centered=.85, varied=.80, full=.80, delayed=.80)
    assert guard.observe(report(10_000, **good))['candidate_streak'] == 1
    assert guard.observe(report(20_000, **good))['candidate_streak'] == 2
    assert guard.observe(report(30_000, **(good | {'delayed': .7999})))['candidate_streak'] == 0
    for step in (40_000, 50_000, 60_000):
        decision = guard.observe(report(step, **good))
    assert decision['candidate_streak'] == 3
    assert decision['independent_test_needed'] and decision['should_stop']
    assert decision['stop_reason'] == 'independent_test_required'
    assert decision['solved'] is False


@pytest.mark.parametrize('changed', [
    {'centered': .8499}, {'varied': .7999}, {'full': .7999}, {'delayed': .7999}])
def test_each_candidate_threshold_is_required(changed):
    guard = initialized(centered=.9, varied=.9)
    good = dict(centered=.85, varied=.80, full=.80, delayed=.80)
    decision = guard.observe(report(10_000, **(good | changed)))
    assert not decision['candidate_thresholds_met']


def test_configuration_controls_gates_and_patience():
    guard = initialized(GuardConfig(max_retention_drop=.3, min_centered_success=.4,
                                   min_varied_success=.3, severe_regression_drop=.4,
                                   severe_regression_patience=1, regression_warmup_steps=5,
                                   stagnation_steps=10))
    assert guard.observe(report(4, centered=.66, varied=.58, full=.08))['selected']
    decision = guard.observe(report(5, centered=.5))
    assert decision['should_stop'] and decision['stop_reason'] == 'severe_regression'


def test_json_round_trip_preserves_history_counters_best_and_next_decision():
    guard = initialized()
    guard.observe(report(10_000, full=.08))
    guard.observe(report(20_000, centered=.2))
    guard.observe(report(30_000, varied=.2))
    state = json.loads(json.dumps(guard.to_dict(), allow_nan=False))
    restored = EvaluationGuard.from_dict(state)
    assert restored.to_dict() == guard.to_dict()
    assert restored.history == guard.history
    assert restored.best_step == 10_000
    assert restored.observe(report(40_000, varied=.2)) == guard.observe(report(40_000, varied=.2))
    stopped = EvaluationGuard.from_dict(json.loads(json.dumps(restored.to_dict())))
    assert stopped.should_stop and stopped.stop_reason == 'severe_regression'


def test_duplicate_last_report_is_idempotent_and_external_mutation_is_isolated():
    guard = initialized()
    raw = report(20_000, centered=0)
    decision = guard.observe(raw)
    assert guard.observe(raw) == decision
    assert len(guard.history) == 2
    raw['suites']['original_centered']['success_rate'] = 1
    decision['best_step'] = -10
    state = guard.to_dict()
    state['history'].clear()
    history = guard.history
    history[0]['best_step'] = -20
    assert guard.history[-1]['regression_streak'] == 1
    assert guard.best_step == 0 and guard.history[0]['best_step'] == 0
    assert len(guard.to_dict()['history']) == 2


def test_invalid_or_out_of_order_reports_leave_state_unchanged():
    guard = EvaluationGuard()
    with pytest.raises(ValueError, match='step-zero'):
        guard.observe(report(1))
    guard.observe(report(0))
    snapshot = guard.to_dict()
    malformed = report(10_000)
    malformed['suites']['full_task']['success_rate'] = float('nan')
    for invalid in (malformed, report(-1), report(0, centered=.2), report(1.5)):
        with pytest.raises(ValueError):
            guard.observe(invalid)
    missing = report(10_000)
    del missing['suites']['lesson']
    with pytest.raises(ValueError, match='Missing'):
        guard.observe(missing)
    assert guard.to_dict() == snapshot


@pytest.mark.parametrize('key,value', [('episodes_per_suite', 50), ('seed_start', 42),
                                     ('policy_mode', 'stochastic')])
def test_incomparable_validation_reports_are_rejected(key, value):
    guard = initialized()
    changed = report(10_000)
    changed[key] = value
    with pytest.raises(ValueError, match='changed from baseline'):
        guard.observe(changed)


def test_fixed_protocols_are_pinned_but_lesson_protocol_can_change():
    baseline = report(0)
    baseline['protocols'] = {name: {'id': name, 'stage': 0} for name in SUITES}
    guard = EvaluationGuard()
    guard.observe(baseline)
    changed_lesson = deepcopy(baseline)
    changed_lesson['step'] = 10_000
    changed_lesson['protocols']['lesson']['stage'] = 4
    guard.observe(changed_lesson)
    changed_full = deepcopy(changed_lesson)
    changed_full['step'] = 20_000
    changed_full['protocols']['full_task']['id'] = 'easier-test'
    with pytest.raises(ValueError, match='protocols changed'):
        guard.observe(changed_full)
    assert EvaluationGuard.from_dict(guard.to_dict()).history == guard.history


@pytest.mark.parametrize('arguments', [
    {'max_retention_drop': -1}, {'candidate_full_success': 1.1},
    {'min_centered_success': float('inf')}, {'severe_regression_patience': 0},
    {'candidate_patience': 1.5}, {'regression_warmup_steps': -1},
    {'stagnation_steps': True}])
def test_invalid_configuration_is_rejected(arguments):
    with pytest.raises(ValueError):
        GuardConfig(**arguments)
