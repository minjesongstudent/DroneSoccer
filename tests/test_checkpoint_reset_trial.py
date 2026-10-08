"""Software contract tests only: no CUDA, learned rollouts, fitting or evaluations."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from tools import run_checkpoint_reset_trial as trial


def test_seed_scan_catches_custom_unused_reservations_and_descendant_parents(tmp_path):
    old=tmp_path/'old';old.mkdir()
    config=trial.configuration(False)
    (old/'protocol.json').write_text(json.dumps(dict(configuration=dict(bank_seeds=config['bank_seeds']))))
    with pytest.raises(ValueError,match='reserved evidence'):
        trial.scan_allocation([tmp_path],config['training_seeds'])
    (old/'protocol.json').write_text(json.dumps(dict(arms=['old_control'],training_seeds=[11000000])))
    (old/'episodes.jsonl').write_text(json.dumps(dict(parent_seed=config['bank_seeds'][0]))+'\n')
    with pytest.raises(ValueError,match='saved or reserved'):
        trial.scan_allocation([tmp_path],config['training_seeds'])
    (old/'episodes.jsonl').write_text(json.dumps(dict(seed=11000000))+'\n')
    found=trial.scan_allocation([tmp_path,tmp_path],config['training_seeds'])
    assert found['episode_rows']==1 and found['conflicts']==[]
    calls=[]
    def stop():
        calls.append(1);raise trial.shared.TrialStopped('STOP')
    with pytest.raises(trial.shared.TrialStopped):trial.scan_allocation([tmp_path],config['training_seeds'],stop)
    assert len(calls)==1


def test_schedule_counts_balances_all_parents_and_never_selects_by_endpoint():
    config=trial.configuration(False)
    parents=[dict(seed=seed,irrelevant_future_crash=i%2==0) for i,seed in enumerate(config['bank_seeds'][:37])]
    schedule=trial.make_schedule(list(reversed(parents)),config)
    assert len(schedule)==240 and len({row['trajectory_id'] for row in schedule})==240
    normal=[row['seed'] for row in schedule if row['kind']=='normal']
    tails=[row['seed'] for row in schedule if row['kind']=='continuation']
    assert normal==config['normal_seeds'] and len(tails)==120
    counts=[tails.count(row['seed']) for row in parents]
    assert max(counts)-min(counts)==1 and min(counts)>0
    for row in parents:row['irrelevant_future_crash']=not row['irrelevant_future_crash']
    assert trial.make_schedule(parents,config)==schedule
    with pytest.raises(ValueError,match='Distinct'):trial.make_schedule(parents[:15],config)


def test_second_clearance_rule_reads_only_present_and_requires_full_current_clearance():
    class NoFuture(SimpleNamespace):
        @property
        def future_outcome(self):raise AssertionError('Future outcome cannot select bank')
    env=NoFuture(_active=True,audit=SimpleNamespace(count=2),
        unwrapped=SimpleNamespace(steps=450,wall_steps=0),_trace={'clearance_events':[0,1]})
    assert trial.eligible_checkpoint(env)
    env._trace['clearance_events'][-1]=0;assert not trial.eligible_checkpoint(env)
    env._trace['clearance_events'][-1]=1;env.audit.count=1;assert not trial.eligible_checkpoint(env)
    env.audit.count=2;env.unwrapped.steps=451;assert not trial.eligible_checkpoint(env)
    env.unwrapped.steps=400;env.unwrapped.wall_steps=1;assert not trial.eligible_checkpoint(env)
    env.unwrapped.wall_steps=0;env._active=False;assert not trial.eligible_checkpoint(env)


def endpoint(prefix=598,tail=2,terminated=False):
    return dict(prefix_steps=prefix,tail_length=tail,absolute_steps=prefix+tail,
        historical_full_body_passes=2,full_body_passes=2,new_full_body_passes=0,
        terminated=terminated,truncated=not terminated,wall_steps=0)


def test_mc_uses_suffix_only_without_inherited_pass_bonus_or_extra_horizon():
    np.testing.assert_array_equal(trial.tail_returns([3.,-5.],endpoint()),[-2.,-5.])
    first=trial.tail_returns([1.,2.],endpoint(prefix=123,terminated=True))
    second=trial.tail_returns([-7.],endpoint(prefix=599,tail=1))
    np.testing.assert_array_equal(first,[3.,2.]);np.testing.assert_array_equal(second,[-7.])
    changed=endpoint();changed['historical_full_body_passes']=99;changed['full_body_passes']=99
    np.testing.assert_array_equal(trial.tail_returns([3.,-5.],changed),[-2.,-5.])
    for bad in (dict(endpoint(),absolute_steps=601),dict(endpoint(),prefix_steps=0),
                dict(endpoint(),terminated=False,truncated=False),dict(endpoint(),new_full_body_passes=1)):
        with pytest.raises(ValueError):trial.tail_returns([3.,-5.],bad)


class CountingPolicy:
    def __init__(self):self.rows=[];self.commands=0
    def reset(self):self.rows=[]
    def observe(self,public):
        assert public.shape==(85,) and public.dtype==np.float32
        self.rows.append(public.copy())
        return np.r_[public,np.array([len(self.rows),0,0,len(self.rows)>=3],np.float32)].astype(np.float32)
    def predict(self,features,deterministic):
        self.commands+=1
        return np.zeros(4,np.float32),None


class SuffixFixture:
    """Two deterministic software transitions; not a drone/learned-policy simulation."""
    def __init__(self):
        self.unwrapped=SimpleNamespace(steps=598,pos=np.zeros(3),heading=0.,wall_steps=0,
            hoop_centre=np.zeros(3),hoop_normal=np.ones(3),_half_box=np.ones(3))
        self.audit=SimpleNamespace(count=2)
        self._trace={'actor_observations':[np.zeros(85,np.float32)]*598,'actions':[np.zeros(4,np.float32)]*598}
        self._last_info={'outcome':'none'}
    def step(self,action):
        state=self.unwrapped;state.steps+=1;state.pos[0]+=1
        public=np.full(85,state.steps,np.float32)
        self._trace['actor_observations'].append(public.copy());self._trace['actions'].append(action.copy())
        cut=state.steps==600;self._last_info['outcome']='timeout' if cut else 'none'
        return public,1.,False,cut,{}


def test_prefix_replayed_once_current_not_repeated_tail_saved_separately(tmp_path):
    policy=CountingPolicy();budget=trial.shared.Budget(tmp_path,60)
    prefix=np.arange(598*85,dtype=np.float32).reshape(598,85)
    trial.replay_prefix(policy,prefix,budget)
    assert policy.commands==0 and len(policy.rows)==598
    current=np.full(85,598,np.float32);env=SuffixFixture()
    arrays,end,path=trial.run_suffix(policy,env,current,tmp_path/'tails','repeat-unique',
        dict(trajectory_id='repeat-unique',seed=11245000,parent_seed=11245000),budget,deterministic=False)
    assert len(policy.rows)==600 and policy.commands==2
    np.testing.assert_array_equal(policy.rows[598],current)
    np.testing.assert_array_equal(arrays['features89'][:,85],[599,600])
    assert end['absolute_steps']==600 and end['prefix_steps']==598 and end['tail_length']==2
    assert end['new_full_body_passes']==0
    with np.load(path,allow_pickle=False) as saved:
        assert len(saved['features89'])==2 and len(saved['raw_actions'])==600
        assert json.loads(str(saved['metadata']))['parent_seed']==11245000
    with pytest.raises(FileExistsError):
        trial.run_suffix(policy,env,current,tmp_path/'tails','repeat-unique',{},budget,deterministic=False)


def test_stop_during_prefix_never_issues_a_command(tmp_path):
    policy=CountingPolicy();budget=trial.shared.Budget(tmp_path,60);(tmp_path/'STOP').touch()
    with pytest.raises(trial.shared.TrialStopped):trial.replay_prefix(policy,np.zeros((4,85),np.float32),budget)
    assert policy.commands==0 and policy.rows==[]


def test_exact_suffix_parity_includes_dtype_sign_bits_absolute_clock_and_terminal_flag():
    source={key:np.array([0.,1.,2.,3.],np.float32) for key in trial.PARITY_KEYS}
    suffix={key:value[2:].copy() for key,value in source.items()}
    assert trial.exact_suffix_parity(source,suffix,2)['passed']
    for key in trial.PARITY_KEYS:
        bad=copy.deepcopy(suffix);bad[key][0]+=1
        with pytest.raises(ValueError,match=key):trial.exact_suffix_parity(source,bad,2)
    bad=copy.deepcopy(suffix);bad['actions']=bad['actions'].astype(np.float64)
    with pytest.raises(ValueError,match='actions'):trial.exact_suffix_parity(source,bad,2)
    source['rewards'][2]=0.;suffix['rewards'][0]=-0.
    with pytest.raises(ValueError,match='rewards'):trial.exact_suffix_parity(source,suffix,2)
    source['rewards']=np.array([0.,0.,1.,3.],np.float64)
    suffix['rewards']=np.array([1.+1e-12,3.],np.float64)
    assert source['rewards'][2:].astype(np.float32).tobytes()==suffix['rewards'].astype(np.float32).tobytes()
    with pytest.raises(ValueError,match='rewards'):trial.exact_suffix_parity(source,suffix,2)


def test_pre_fit_barrier_rejects_missing_and_tampered_parity_without_work(tmp_path):
    bank=tmp_path/'bank';bank.mkdir();protocol=tmp_path/'protocol.json';protocol.write_text('{}')
    parents=[dict(seed=11245000)]
    trial.shared.write_json(bank/'parity.json',dict(passed=False,complete=True,coverage_passed=True,checks=[dict(passed=False)]))
    digest=trial.pre_fit_manifest(tmp_path,parents,[],trial.shared.sha(protocol))
    with pytest.raises(ValueError,match='Every authentic'):trial.require_pre_fit(tmp_path,digest,trial.shared.sha(protocol))
    trial.shared.write_json(bank/'parity.json',dict(passed=True,complete=True,coverage_passed=True,checks=[dict(passed=True)]))
    with pytest.raises(ValueError,match='evidence changed'):trial.require_pre_fit(tmp_path,digest,trial.shared.sha(protocol))


def completed_fixture(smoke=False):
    config=trial.configuration(smoke)
    parents=[dict(seed=s) for s in config['bank_seeds'][:config['minimum_parents']]]
    schedule=trial.make_schedule(parents,config)
    result=dict(status='complete',completed_training_rounds=config['rounds'],frozen_unchanged=True,
        completed_trajectory_ids=[r['trajectory_id'] for r in schedule],final_development={},feature_use={},
        iterations=[dict(iteration=i,fresh_starts=config['normal_per_round'],continuations=config['tails_per_round'],failures=[])
                    for i in range(1,config['rounds']+1)])
    return config,schedule,result


def test_budget_reports_descendants_truthfully_and_stopped_pipeline_never_passes(tmp_path,monkeypatch):
    config,schedule,result=completed_fixture()
    assert trial.training_complete(result,schedule,config)
    result['status']='stopped';result['error']='Retention regression';del result['final_development'];del result['feature_use']
    result['iterations'][-1]['failures']=['randomised crash rise exceeds2']
    monkeypatch.setattr(trial.shared,'verify_evaluation_identity',lambda *a,**k:pytest.fail('Skipped evaluation must not run'))
    (tmp_path/'retained.txt').write_text('partial evidence')
    decision=trial.final_screen(tmp_path,result,config,schedule,None,None,None)
    assert decision['training_budget_complete'] and not decision['final_pipeline_complete'] and not decision['passed']
    assert any('Retention regression' in r for r in decision['reasons'])
    assert not any('Evidence validation failed' in r for r in decision['reasons'])
    assert decision['evidence'][str(tmp_path/'retained.txt')]==trial.shared.sha(tmp_path/'retained.txt')
    assert decision['unique_training_parent_layouts']==120+16
    assert not decision['acceptance'] and not decision['policy_promoted'] and not decision['physical_flight']
    result['completed_trajectory_ids'].pop();assert not trial.training_complete(result,schedule,config)


def test_software_smoke_and_incomplete_rounds_cannot_qualify():
    config,schedule,result=completed_fixture(True)
    assert any('Software-only' in r for r in trial.decision_reasons(result,schedule,config))
    config,schedule,result=completed_fixture()
    result['iterations'][-1]['iteration']=4
    assert any('Every prescribed round' in r for r in trial.decision_reasons(result,schedule,config))
    result['frozen_unchanged']=False
    assert any('Frozen source' in r for r in trial.decision_reasons(result,schedule,config))


def test_source_changes_report_failure_even_after_retention_stop(tmp_path,monkeypatch):
    root=tmp_path/'repo';root.mkdir();out=tmp_path/'run';out.mkdir();(out/'source').mkdir()
    (root/'code.py').write_text('original');(out/'source/code.py').write_text('original')
    digest=trial.shared.sha(root/'code.py')
    trial.shared.write_json(out/'protocol.json',dict(sources={'code.py':digest},input_sha256={}))
    protocol=trial.shared.sha(out/'protocol.json');monkeypatch.setattr(trial,'ROOT',root)
    result=dict(status='stopped',error='Retention regression',frozen_unchanged=True)
    trial.record_source_integrity(out,result,protocol)
    assert result['frozen_sources_verified'] and result['status']=='stopped'
    (root/'code.py').write_text('changed');trial.record_source_integrity(out,result,protocol)
    assert not result['frozen_unchanged'] and not result['frozen_sources_verified'] and result['status']=='failed'


def test_unchanged_final_normal_start_gates_reject_threshold_and_any_suite_regression():
    original={s:dict(episodes=100,first_pass=a,clean_repeat=b,crashes=c) for s,a,b,c in
              [('aligned',84,84,16),('nominal',80,74,26),('randomised',73,48,52)]}
    candidate=copy.deepcopy(original);candidate['randomised'].update(clean_repeat=53,crashes=47)
    assert trial.shared.gae_screen_reasons(original,candidate)==[]
    candidate['nominal']['first_pass']=77
    assert any('nominal: first-pass' in r for r in trial.shared.gae_screen_reasons(original,candidate))
    candidate=copy.deepcopy(original)
    assert len(trial.shared.gae_screen_reasons(original,candidate))==2
