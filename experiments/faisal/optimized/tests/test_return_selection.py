from copy import deepcopy
import json

import pytest

from src.return_selection import ReturnSelectionConfig, ReturnSkillSelector


def report(step, successes=0, front=0, crash=.8, centered=.96, varied=.92):
    return {'step': step, 'suites': {
        'original_centered': {'success_rate': centered},
        'varied_approach': {'success_rate': varied},
        'return_practice': {'episodes': 50, 'successful_episodes': successes,
                            'safe_front_return_episodes': front, 'crash_rate': crash}},
        'protocols': {'return_practice': {
            'id': 'return-practice-200-step-v1', 'bank_sha256': 'a'*64,
            'seeds': list(range(8_000_000, 8_000_050)),
            'success': 'at least one new fully cleared pass and survive 200 remaining steps'}}}


def initialized(**rates):
    selector = ReturnSkillSelector()
    selector.observe(report(0, **rates))
    return selector


def test_zero_success_baseline_is_preserved_and_is_not_rejected():
    selector = initialized()
    assert selector.best_return_step == 0
    assert selector.history[0]['selected']
    assert not selector.should_stop
    assert not selector.extension_allowed
    assert selector.history[0]['mission_acceptance'] is False
    assert selector.history[0]['independent_test_needed'] is False


def test_original_retention_reference_does_not_degrade_with_inherited_baseline():
    selector = initialized(centered=.70, varied=.60)
    baseline = selector.history[0]
    assert baseline['selected'] and not baseline['eligible']
    assert baseline['retention_thresholds'] == pytest.approx({'original_centered': .84, 'varied_approach': .80})
    assert not selector.observe(report(5_000, successes=10, centered=.8399))['eligible']
    assert not selector.observe(report(10_000, successes=10, varied=.7999))['eligible']
    decision = selector.observe(report(15_000, successes=1, centered=.84, varied=.80))
    assert decision['eligible'] and decision['selected']
    restored = ReturnSkillSelector.from_dict(json.loads(json.dumps(selector.to_dict())))
    assert restored.retention_thresholds == pytest.approx({'original_centered': .84, 'varied_approach': .80})


def test_rank_prefers_new_passes_then_front_returns_then_fewer_crashes_and_keeps_ties():
    selector = initialized()
    assert selector.observe(report(1_000, front=2))['selected']
    assert selector.observe(report(2_000, front=2, crash=.7))['selected']
    assert selector.observe(report(3_000, successes=1, front=1, crash=.9))['selected']
    assert not selector.observe(report(4_000, successes=1, front=1, crash=.9))['selected']
    assert not selector.observe(report(5_000, successes=0, front=50, crash=0))['selected']
    assert not selector.observe(report(6_000, successes=50, front=50, crash=0, centered=.8))['selected']
    assert selector.best_return_step == 3_000


@pytest.mark.parametrize('metric', ['successes', 'front'])
def test_five_episode_gain_for_two_consecutive_checks_permits_extension(metric):
    selector = initialized(successes=2, front=3)
    values = {'successes': 2, 'front': 3}
    values[metric] += 5
    first = selector.observe(report(15_000, **values))
    assert first['improvement_streak'] == 1 and not first['extension_allowed']
    second = selector.observe(report(20_000, **values))
    assert second['improvement_streak'] == 2 and second['extension_allowed']
    assert not second['should_stop']
    assert not second['mission_acceptance'] and not second['independent_test_needed']


def test_four_episode_gain_or_ineligible_gain_does_not_count_and_streak_resets():
    selector = initialized()
    assert selector.observe(report(5_000, front=4))['improvement_streak'] == 0
    assert selector.observe(report(10_000, front=5))['improvement_streak'] == 1
    assert selector.observe(report(15_000, front=5, varied=.79))['improvement_streak'] == 0
    decision = selector.observe(report(20_000, front=5))
    assert decision['improvement_streak'] == 1
    assert decision['should_stop'] and decision['stop_reason'] == 'return_no_sustained_progress'


def test_extension_gate_exact_boundary_and_stopping_is_latched():
    selector = initialized()
    assert not selector.observe(report(19_999))['should_stop']
    assert selector.observe(report(20_000))['stop_reason'] == 'return_no_sustained_progress'
    selector.observe(report(25_000, successes=10, front=10))
    after = selector.observe(report(30_000, successes=10, front=10))
    assert after['extension_allowed']
    assert after['should_stop'] and after['stop_reason'] == 'return_no_sustained_progress'


def test_front_only_progress_cannot_extend_past_forty_thousand_without_a_new_pass_gain():
    selector = initialized(successes=2, front=3)
    selector.observe(report(10_000, successes=2, front=8))
    assert selector.observe(report(20_000, successes=2, front=8))['extension_allowed']
    # Permission is earned once, rather than being lost on a single noisy check.
    assert selector.observe(report(30_000, successes=2, front=7))['extension_allowed']
    assert not selector.observe(report(39_999, successes=2, front=8))['should_stop']
    final = selector.observe(report(40_000, successes=2, front=8))
    assert final['should_stop'] and final['stop_reason'] == 'return_no_new_pass_gain'
    assert not final['genuine_new_pass_gain']


def test_genuine_eligible_new_pass_gain_avoids_no_pass_cap_but_is_not_mission_acceptance():
    selector = initialized(successes=2, front=3)
    selector.observe(report(10_000, successes=2, front=8))
    selector.observe(report(20_000, successes=2, front=8))
    final = selector.observe(report(40_000, successes=3, front=8))
    assert final['genuine_new_pass_gain'] and not final['should_stop']
    assert not final['mission_acceptance'] and not final['independent_test_needed']


def test_ineligible_new_pass_gain_cannot_avoid_hard_cap():
    selector = initialized()
    selector.observe(report(10_000, front=5))
    selector.observe(report(20_000, front=5))
    final = selector.observe(report(40_000, successes=25, front=25, centered=.5))
    assert not final['eligible'] and not final['genuine_new_pass_gain']
    assert final['stop_reason'] == 'return_no_new_pass_gain'


def test_protocol_bank_and_seed_set_are_immutable():
    selector = initialized()
    for key, value in [('id', 'easier-return'), ('bank_sha256', 'b'*64),
                       ('seeds', list(range(8_100_000, 8_100_050))),
                       ('success', 'any centre-plane crossing')]:
        changed = report(5_000)
        changed['protocols']['return_practice'][key] = value
        with pytest.raises(ValueError, match='changed from baseline'):
            selector.observe(changed)
    assert len(selector.history) == 1


def test_json_resume_reconstructs_best_streak_permission_and_stop():
    selector = initialized()
    selector.observe(report(15_000, front=5))
    restored = ReturnSkillSelector.from_dict(json.loads(json.dumps(selector.to_dict(), allow_nan=False)))
    assert restored.history == selector.history
    assert restored.observe(report(20_000, front=5)) == selector.observe(report(20_000, front=5))
    assert restored.extension_allowed and restored.best_return_step == 15_000
    selector.observe(report(40_000, front=5))
    stopped = ReturnSkillSelector.from_dict(json.loads(json.dumps(selector.to_dict())))
    assert stopped.history == selector.history and stopped.should_stop
    assert stopped.stop_reason == 'return_no_new_pass_gain'


def test_duplicate_last_report_is_idempotent_and_returned_state_cannot_mutate_selector():
    selector = initialized()
    raw = report(10_000, front=5)
    decision = selector.observe(raw)
    assert selector.observe(raw) == decision
    assert len(selector.history) == 2 and not selector.extension_allowed
    raw['protocols']['return_practice']['seeds'][0] = 999
    decision['best_return_step'] = -1
    snapshot = selector.to_dict()
    snapshot['history'].clear()
    assert selector.best_return_step == 10_000
    assert len(selector.to_dict()['history']) == 2
    assert selector.history[-1]['improvement_streak'] == 1


@pytest.mark.parametrize('mutate', [
    lambda r: r.update(step=-1),
    lambda r: r.update(step=1.2),
    lambda r: r['suites']['return_practice'].update(episodes=25),
    lambda r: r['suites']['return_practice'].update(successful_episodes=51),
    lambda r: r['suites']['return_practice'].update(safe_front_return_episodes=-1),
    lambda r: r['suites']['return_practice'].update(crash_rate=float('nan')),
    lambda r: r['suites']['original_centered'].update(success_rate=1.1),
    lambda r: r['protocols']['return_practice'].update(bank_sha256='missing'),
    lambda r: r['protocols']['return_practice'].update(seeds=[8_000_000]*50),
    lambda r: r['protocols']['return_practice'].update(seeds=[True]+list(range(1, 50))),
    lambda r: r['protocols']['return_practice'].update(id=''),
])
def test_malformed_report_rejected_without_mutating_state(mutate):
    selector = initialized()
    before = deepcopy(selector.to_dict())
    malformed = report(10_000)
    mutate(malformed)
    with pytest.raises(ValueError):
        selector.observe(malformed)
    assert selector.to_dict() == before


def test_requires_zero_baseline_and_increasing_steps():
    selector = ReturnSkillSelector()
    with pytest.raises(ValueError, match='step-zero'):
        selector.observe(report(5_000))
    selector.observe(report(0))
    selector.observe(report(5_000))
    for invalid in (report(0), report(4_999), report(5_000, front=1)):
        with pytest.raises(ValueError, match='increase'):
            selector.observe(invalid)


@pytest.mark.parametrize('arguments', [
    {'episodes': 0}, {'improvement_episodes': 51}, {'improvement_checks': 0},
    {'reference_centered': float('nan')}, {'varied_floor': -.1},
    {'extension_gate_steps': True}, {'no_new_pass_cap_steps': 10_000},
])
def test_configuration_rejects_invalid_thresholds(arguments):
    with pytest.raises(ValueError):
        ReturnSelectionConfig(**arguments)
