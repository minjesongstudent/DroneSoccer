import numpy as np
import pytest
import torch
import json
from types import SimpleNamespace
import tools.run_onpolicy_trial as trial
from tools.run_onpolicy_trial import (
    discounted_returns,clipped_loss,mission_returns,grouped_advantages,
    finite_horizon_gae,parse_args,
)


def test_returns_end_at_episode_boundary():
    np.testing.assert_allclose(discounted_returns([1,2,3],.5),[2.75,3.5,3])
    np.testing.assert_array_equal(discounted_returns([-30],.995),[-30])


def test_clipping_for_both_advantage_signs():
    logs=torch.log(torch.tensor([1.5,.5],requires_grad=True))
    value=clipped_loss(logs,torch.zeros(2),torch.tensor([1.,-1.]))
    torch.testing.assert_close(value,torch.tensor(-.1))


def test_mission_requires_full_survival_repeat_and_no_wall():
    assert mission_returns(600,2,False,0).sum()==600
    for length,passes,crash,wall in [(599,2,False,0),(600,1,False,0),(600,2,True,0),(600,2,False,1)]:
        assert mission_returns(length,passes,crash,wall).sum()==0


def test_group_baseline_excludes_self_and_handles_equal_outcomes():
    lengths=[2,1,3,2,1,1,1,1,1]
    outcome=[1,0,1,0,1,1,1,1,0]
    adv,mixed=grouped_advantages(np.repeat(outcome,lengths),lengths,4)
    np.testing.assert_allclose(adv,np.repeat([2/3,-2/3,2/3,-2/3,0,0,0,0,0],lengths))
    assert mixed==1


def test_gae_lambda_one_matches_mc_minus_frozen_value_for_complete_episodes():
    rewards=np.array([1.,2.,-30.,.1,40.,-.1])
    values=np.array([4.,-2.,17.,30.,6.,21.])
    actual=finite_horizon_gae(rewards,values,[3,3],lam=1.)
    expected=np.concatenate([discounted_returns(rewards[:3],.995),
                             discounted_returns(rewards[3:],.995)])-values
    np.testing.assert_allclose(actual,expected,rtol=1e-6,atol=3e-6)


def test_gae_does_not_leak_a_later_episode_into_an_earlier_crash():
    first=finite_horizon_gae([1.,-30.],[2.,15.],[2])
    joined=finite_horizon_gae([1.,-30.,1000.,1000.],[2.,15.,500.,500.],[2,2])
    np.testing.assert_array_equal(first,joined[:2])
    assert joined[1]==-45.


def test_gae_zeroes_crash_and_finite_timeout_tails():
    # Crash and timeout final values are both discarded after their last action.
    rewards=np.r_[-30.,np.full(600,-.1)]
    values=np.r_[80.,np.full(600,80.)]
    advantages=finite_horizon_gae(rewards,values,[1,600])
    assert advantages[0]==-110.
    np.testing.assert_allclose(advantages[-1],-80.1,atol=1e-5)
    one_step=finite_horizon_gae([1.,2.],[10.,20.],[2],gamma=.5,lam=0.)
    np.testing.assert_allclose(one_step,[1.,-18.])


@pytest.mark.parametrize('lengths',[[1],[1,0],[.5,1.5],[],[3]])
def test_gae_rejects_incomplete_or_invalid_episode_partitions(lengths):
    with pytest.raises(ValueError):finite_horizon_gae([1.,2.],[1.,2.],lengths)


def test_gae_rejects_nonfinite_values_and_invalid_lambda():
    with pytest.raises(ValueError):finite_horizon_gae([1.],[np.nan],[1])
    with pytest.raises(ValueError):finite_horizon_gae([1.],[0.],[1],lam=1.1)


def test_gae_option_requires_the_preregistered_paired_configuration():
    paired=['--out','unused-test-directory','--advantage-estimator','gae95',
            '--approach-curriculum','--backtracking','--actor-lr','1e-5',
            '--episodes-per-iteration','48','--actor-epochs','1']
    assert parse_args(paired).advantage_estimator=='gae95'
    for flag in ('--approach-curriculum','--backtracking'):
        with pytest.raises(SystemExit):parse_args([arg for arg in paired if arg!=flag])
    with pytest.raises(SystemExit):parse_args(paired+['--objective','mission'])
    with pytest.raises(SystemExit):parse_args(paired+['--actor-epochs','3'])
    assert parse_args(['--out','unused-test-directory']).advantage_estimator=='monte-carlo'


def _write_episode_fixture(path,seed,first,repeat,crashes,count=100):
    path.parent.mkdir(parents=True,exist_ok=True)
    rows=[]
    for index in range(count):
        crashed=index>=count-crashes;repeated=index<repeat;passed=index<first
        rows.append(dict(seed=seed+index,steps=10 if crashed else 600,
            outcome='hit' if crashed else 'timeout',wall_steps=0,
            full_body_passes=2 if repeated else int(passed),returns_to_front=int(passed),
            crashed=crashed,clean_first_pass=passed,clean_repeated_episode=repeated))
    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    return dict(episodes=count,first_pass=first,clean_repeat=repeat,crashes=crashes)


@pytest.fixture
def gae_screen_fixture(tmp_path,monkeypatch):
    source=tmp_path/'source.zip';source.write_bytes(b'test-original-policy')
    profile=SimpleNamespace(name='test-original',sha256=trial.sha(source),lags=(1,0))
    monkeypatch.setattr(trial,'PROFILE',profile)
    reference=tmp_path/'baseline';reference.mkdir()
    report={}
    # Deliberately differs from the real report to prove no hardcoded counts.
    counts={'aligned':(90,86,14),'nominal':(85,80,20),'randomised':(74,44,56)}
    for suite,seed in trial.SUITES.items():
        first,repeat,crashes=counts[suite]
        _write_episode_fixture(reference/profile.name/suite/'episodes.jsonl',seed,first,repeat,crashes)
        report[f'{profile.name}/{suite}']=dict(episodes=100,clean_first_pass=first,
                                              clean_repeated_episodes=repeat,crashes=crashes)
    trial.write_json(reference/'results.json',report)
    trial.write_json(reference/'protocol.json',dict(episodes=100,max_steps=600,suites=trial.SUITES,
        profiles={profile.name:dict(sha256=profile.sha256,lags=profile.lags)}))
    out=tmp_path/'run';out.mkdir()
    record=trial.prepare_gae_screen_reference(reference,out/'screen_reference',source)
    trial.write_json(out/'protocol.json',dict(final_screen_reference=record))
    checkpoint=out/'policy_05.zip';checkpoint.write_bytes(b'test-final-policy')
    final=out/'final_development';final.mkdir()
    trial.write_json(final/'model.json',dict(sha256=trial.sha(checkpoint),lags=profile.lags))
    candidate={}
    for suite,seed in trial.SUITES.items():
        first,repeat,crashes=counts[suite]
        if suite=='randomised':repeat+=5;crashes-=5
        candidate[suite]=_write_episode_fixture(final/suite/'episodes.jsonl',seed,first,repeat,crashes)
    trial.write_json(final/'summary.json',candidate)
    result=dict(status='bounded_trial_complete',source_unchanged=True,
                iterations=[dict(iteration=5,failures=[])],final_development=candidate)
    trial.write_json(out/'result.json',result)
    return out,result,checkpoint,source,record


def test_final_screen_counts_real_report_and_binds_evidence(gae_screen_fixture):
    out,result,checkpoint,source,record=gae_screen_fixture
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert decision['passed'] and decision['candidate']['randomised']['clean_repeat']==49
    assert decision['reference']['randomised']['clean_repeat']==44
    assert decision['decision']=='eligible_for_one_fresh_development_screen'
    assert not decision['acceptance'] and not decision['policy_promoted']
    for label in ('source_policy','final_candidate','baseline/results.json','baseline/protocol.json'):
        item=decision['evidence'][label]
        assert item['sha256']==trial.sha(item['path'])
    assert json.loads((out/'final_screen_decision.json').read_text())==decision


def test_final_screen_rejects_optimistic_summary_or_duplicate_seed(gae_screen_fixture):
    out,result,*_=gae_screen_fixture
    path=out/'final_development/randomised/episodes.jsonl'
    lines=path.read_text().splitlines();lines[-1]=lines[0]
    path.write_text('\n'.join(lines)+'\n')
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert not decision['passed']
    assert any('100 unique' in reason for reason in decision['reasons'])


def test_final_screen_rejects_a_summary_that_does_not_match_episode_counts(gae_screen_fixture):
    out,result,*_=gae_screen_fixture
    result['final_development']['randomised']['clean_repeat']+=1
    trial.write_json(out/'final_development/summary.json',result['final_development'])
    trial.write_json(out/'result.json',result)
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert not decision['passed']
    assert any('disagrees with saved episodes' in reason for reason in decision['reasons'])


def test_final_screen_rejects_baseline_tampering(gae_screen_fixture):
    out,*_=gae_screen_fixture
    (out/'screen_reference/results.json').write_text('{}')
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert not decision['passed']
    assert any('Frozen baseline results.json changed' in reason for reason in decision['reasons'])


def test_final_screen_rejects_source_change_or_retention_stop(gae_screen_fixture):
    out,result,checkpoint,source,record=gae_screen_fixture
    result['status']='retention_regression';source.write_bytes(b'changed-source')
    trial.write_json(out/'result.json',result)
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert not decision['passed']
    assert 'Original source policy changed' in decision['reasons']
    assert 'Training hit a retention-regression stop' in decision['reasons']


def test_final_screen_ignores_favorable_intermediate_and_requires_both_gains(gae_screen_fixture):
    out,result,*_=gae_screen_fixture
    result['iterations'].insert(0,dict(iteration=1,failures=[],metrics={'randomised':dict(clean_repeat=100,crashes=0)}))
    result['final_development']['randomised']=_write_episode_fixture(
        out/'final_development/randomised/episodes.jsonl',trial.SUITES['randomised'],74,48,52)
    trial.write_json(out/'final_development/summary.json',result['final_development'])
    trial.write_json(out/'result.json',result)
    decision=trial.write_gae_final_screen(*gae_screen_fixture)
    assert not decision['passed']
    assert 'randomised: clean-repeat gain is below 5/100' in decision['reasons']
    assert 'randomised: crash reduction is below 5/100' in decision['reasons']


def test_final_screen_smoke_is_never_a_pass(gae_screen_fixture):
    decision=trial.write_gae_final_screen(*gae_screen_fixture,smoke=True)
    assert not decision['passed'] and decision['decision']=='software_only'


@pytest.mark.parametrize('metric,delta', [('first_pass',-3),('clean_repeat',-3),('crashes',3)])
def test_final_screen_enforces_all_suite_retention(gae_screen_fixture,metric,delta):
    out,result,checkpoint,source,record=gae_screen_fixture
    reference=json.loads((out/'screen_reference/reference.json').read_text())['metrics']
    candidate=json.loads(json.dumps(result['final_development']))
    candidate['nominal'][metric]=reference['nominal'][metric]+delta
    assert trial.gae_screen_reasons(reference,candidate)
