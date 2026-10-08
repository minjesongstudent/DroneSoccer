import copy
from contextlib import contextmanager
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.camera_history import CameraHistory
from tools import run_estimated_motion_trial as trial


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


def test_total_budget_includes_preparation_and_stop_is_immediate(tmp_path):
    now = [10.]
    budget = trial.Budget(tmp_path, 60, started=0., clock=lambda: now[0])
    budget.check()
    now[0] = 60.
    with pytest.raises(trial.TrialStopped, match='Total wall'): budget.check()
    now[0] = 11.
    (tmp_path/'STOP').touch()
    with pytest.raises(trial.TrialStopped, match='STOP'): budget.check()


def public_trace(length=8):
    rng = np.random.default_rng(71); history = CameraHistory(trial.PROFILE.lags)
    actions = rng.uniform(-1,1,(length,4)).astype(np.float32)
    observations = []
    for t in range(length):
        public = rng.normal(size=8).astype(np.float32)
        public[6] = float(t%2 == 0)
        observations.append(history.update(public, np.zeros(4,np.float32) if t == 0 else actions[t-1]))
    done = np.zeros(length,bool); done[-1] = True
    return dict(actor_observations=np.asarray(observations), actions=actions,
                terminated=done, truncated=np.zeros(length,bool))


def test_guard_inputs_do_not_read_truth_targets_or_current_terminal_action():
    class PublicOnly(dict):
        def __getitem__(self, key):
            if key not in ('actor_observations','actions','terminated','truncated'):
                raise AssertionError('Truth or estimator targets were read')
            return super().__getitem__(key)
    trace = PublicOnly(public_trace())
    expected = trial.public_episode_arrays(trace)
    trace['positions'] = object(); trace['headings'] = object(); trace['targets'] = object()
    trace['actions'][-1] *= -1  # Not in any pre-action observation in this episode.
    np.testing.assert_array_equal(trial.public_episode_arrays(trace), expected)
    trace['actor_observations'][3,72] += .5
    with pytest.raises(ValueError, match='history mismatch'): trial.public_episode_arrays(trace)


def test_public_episode_loader_rejects_partial_or_cross_reset_sequences():
    trace = public_trace(); trace['terminated'][-1] = False
    with pytest.raises(ValueError): trial.public_episode_arrays(trace)
    trace = public_trace(); trace['terminated'][2] = True
    with pytest.raises(ValueError): trial.public_episode_arrays(trace)


def test_seed_allocation_rejects_saved_episode_and_unexecuted_declared_budget(tmp_path):
    folder = tmp_path/'old'; folder.mkdir()
    (folder/'episodes.jsonl').write_text(json.dumps(dict(seed=11242000))+'\n')
    with pytest.raises(ValueError, match='overlaps'): trial.scan_allocation(tmp_path, trial.MAIN_SEEDS)
    (folder/'episodes.jsonl').write_text(json.dumps(dict(seed=11241000))+'\n')
    write(folder/'protocol.json', dict(training_seeds=list(trial.MAIN_SEEDS)))
    with pytest.raises(ValueError, match='overlaps'): trial.scan_allocation(tmp_path, trial.MAIN_SEEDS)
    write(folder/'protocol.json', dict(reset_seed_base=11241000,iterations=5,episodes_per_iteration=48))
    result = trial.scan_allocation(tmp_path, trial.MAIN_SEEDS)
    assert result['conflicts'] == [] and result['episode_rows'] == 1
    with pytest.raises(ValueError, match='FIT interval'): trial.scan_allocation(tmp_path, (11250000,))


def test_warmup_whole_episode_partition_is_every_fifth_and_public_only(tmp_path):
    records = []
    for index in range(11):
        path = tmp_path/f'{index}.npz'; np.savez(path, harmless=np.zeros(1))
        records.append(dict(seed=11000000+index, trace=path.name, status='complete', steps=10))
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    actual = trial.warmup_records(tmp_path, smoke=True)
    assert len(actual) == 10
    assert [r['index'] for r in actual if r['split']=='heldout'] == [4,9]
    assert set(r['seed'] for r in actual if r['split']=='fit').isdisjoint(
        r['seed'] for r in actual if r['split']=='heldout')
    records[0]['steps'] = 199998
    (tmp_path/'episodes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in records))
    assert len(trial.warmup_records(tmp_path, smoke=True)) == 1


def test_frozen_probe_draws_are_valid_and_shuffle_jointly_within_visibility():
    x = np.zeros((100,89),np.float32)
    x[:,88] = 1; x[:2,88] = 0; x[::2,70] = 1
    x[:,85:88] = np.arange(100)[:,None] + np.array([.1,.2,.3])
    selected, permutation = trial.probe_selection(x,64,11591000)
    again = trial.probe_selection(x,64,11591000)
    np.testing.assert_array_equal(selected, again[0]); np.testing.assert_array_equal(permutation, again[1])
    assert (selected >= 2).all() and len(set(selected)) == 64
    assert sorted(permutation) == list(range(64))
    np.testing.assert_array_equal(x[selected,70], x[selected[permutation],70])
    with pytest.raises(ValueError): trial.probe_selection(x,99,11591000)


def completed_result():
    return dict(status='complete', frozen_unchanged=True,
                final_development={}, feature_use={},
                completed_seeds=list(trial.MAIN_SEEDS),
                iterations=[dict(iteration=i,episodes=48,failures=[]) for i in range(1,6)])


@pytest.mark.parametrize('change', ['stopped','short_round','duplicate_seed','retention','changed_source'])
def test_incomplete_or_invalid_run_cannot_pass_even_with_good_flight_counts(change):
    result = completed_result()
    assert trial.completion_reasons(result,False) == []
    if change == 'stopped': result['status'] = 'stopped'
    elif change == 'short_round': result['iterations'][-1]['episodes'] = 47
    elif change == 'duplicate_seed': result['completed_seeds'][-1] = result['completed_seeds'][0]
    elif change == 'retention': result['iterations'][1]['failures'] = ['first-pass regression']
    elif change == 'changed_source': result['frozen_unchanged'] = False
    assert trial.completion_reasons(result,False)


def test_smoke_never_passes_but_its_exact_software_schedule_is_recognized():
    result = dict(status='complete',frozen_unchanged=True,completed_seeds=list(trial.SMOKE_SEEDS),
                  final_development={}, feature_use={},
                  iterations=[dict(iteration=1,episodes=2,failures=[])])
    assert trial.completion_reasons(result,True) == ['Software-only smoke can never pass']


def test_feature_gate_requires_dependence_and_exact_zero_parity():
    source = np.zeros((20,4),np.float32)
    actual = source+.02; shuffled = actual+.01
    assert trial.feature_use_metrics(source,actual,shuffled,source)['passed']
    assert not trial.feature_use_metrics(source,actual,actual+.001,source)['passed']
    assert not trial.feature_use_metrics(source,source,source,source)['passed']
    almost = source.copy(); almost[0,0] = 1e-8
    assert not trial.feature_use_metrics(source,actual,shuffled,almost)['passed']


def evaluation_fixture(tmp_path, count=2):
    checkpoint = tmp_path/'candidate.npz'; checkpoint.write_bytes(b'candidate identity fixture')
    folder = tmp_path/'evaluation'; folder.mkdir()
    manifest = trial.checkpoint_manifest()
    write(folder/'model.json', dict(sha256=trial.sha(checkpoint),manifest=manifest,
          episodes_per_suite=count,diagnostic_dynamics=None,precision='strict_float32'))
    evidence = {}; summary = {}
    for suite, seed in trial.SUITES.items():
        target = folder/suite; target.mkdir(); rows=[]
        for index in range(count):
            row=dict(seed=seed+index,steps=600,outcome='timeout',wall_steps=0,
                     full_body_passes=2,returns_to_front=1,crashed=False,
                     clean_first_pass=True,clean_repeated_episode=True)
            rows.append(row)
            path=target/f'{seed+index}.npz'; path.write_bytes(b'synthetic trace identity')
            evidence[f'{suite}/{path.name}']=trial.sha(path)
        log=target/'episodes.jsonl'; log.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        evidence[f'{suite}/episodes.jsonl']=trial.sha(log); summary[suite]=trial.summarize(rows)
    write(folder/'summary.json',summary)
    for name in ('model.json','summary.json'): evidence[name]=trial.sha(folder/name)
    write(folder/'evidence.json',evidence)
    return folder,checkpoint


def test_evaluation_identity_counts_exact_seeds_and_binds_all_trajectory_hashes(tmp_path):
    folder,checkpoint=evaluation_fixture(tmp_path)
    result=trial.verify_evaluation_identity(folder,checkpoint,2)
    assert all(row['episodes']==2 and row['clean_repeat']==2 for row in result.values())
    (folder/'aligned/9410000.npz').write_bytes(b'different trace')
    with pytest.raises(ValueError,match='changed'): trial.verify_evaluation_identity(folder,checkpoint,2)


@pytest.mark.parametrize('bad', ['candidate_hash','wrong_profile','different_physics','duplicate_seed','missing_trace'])
def test_evaluation_with_good_counts_cannot_use_wrong_candidate_or_incomplete_evidence(tmp_path,bad):
    folder,checkpoint=evaluation_fixture(tmp_path)
    identity=json.loads((folder/'model.json').read_text())
    evidence=json.loads((folder/'evidence.json').read_text())
    if bad == 'candidate_hash': identity['sha256']='0'*64
    elif bad == 'wrong_profile': identity['manifest']['profile']='four-action-89'
    elif bad == 'different_physics': identity['diagnostic_dynamics']={'tau':.55}
    elif bad == 'missing_trace': del evidence['aligned/9410000.npz']
    elif bad == 'duplicate_seed':
        log=folder/'aligned/episodes.jsonl'; rows=[json.loads(x) for x in log.read_text().splitlines()]
        rows[1]['seed']=rows[0]['seed']; log.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        evidence['aligned/episodes.jsonl']=trial.sha(log)
    write(folder/'model.json',identity); evidence['model.json']=trial.sha(folder/'model.json')
    write(folder/'evidence.json',evidence)
    with pytest.raises(ValueError): trial.verify_evaluation_identity(folder,checkpoint,2)


def test_fixed_flight_gate_rejects_small_gain_and_any_suite_retention_loss():
    reference={suite:dict(episodes=100,first_pass=80,clean_repeat=48,crashes=52) for suite in trial.SUITES}
    candidate=copy.deepcopy(reference)
    candidate['randomised'].update(clean_repeat=53,crashes=47)
    assert trial.gae_screen_reasons(reference,candidate)==[]
    candidate['randomised']['clean_repeat']=52
    assert trial.gae_screen_reasons(reference,candidate)
    candidate['randomised']['clean_repeat']=53
    candidate['nominal']['first_pass']=77
    assert trial.gae_screen_reasons(reference,candidate)


def test_final_screen_rejects_missing_evidence_and_stop_without_throwing(tmp_path):
    result=completed_result(); result['final_checkpoint']='policy_05.npz'
    decision=trial.final_screen(tmp_path,result,{'path':'missing','sha256':'0'*64},'0'*64,smoke=False)
    assert not decision['passed'] and decision['decision']=='reject_close_variant'
    def stop(): raise trial.TrialStopped('STOP requested')
    decision=trial.final_screen(tmp_path,result,{},None,smoke=False,check=stop)
    assert not decision['passed'] and 'STOP' in decision['reasons'][-1]


def stopped_training_complete():
    result = completed_result()
    result.update(status='stopped',error='Retention regression',completed_training_rounds=5)
    result.pop('final_development'); result.pop('feature_use')
    result['iterations'][-1]['failures'] = ['randomised: crashes increased by more than 3/25']
    return result


def test_retention_stop_reports_training_complete_but_final_pipeline_skipped(tmp_path, monkeypatch):
    result = stopped_training_complete()
    def never(*args, **kwargs): raise AssertionError('Skipped final validation/probe was attempted')
    monkeypatch.setattr(trial, 'verify_evaluation_identity', never)
    monkeypatch.setattr(trial, 'feature_use_metrics', never)
    decision = trial.final_screen(tmp_path,result,None,None,smoke=False,check=never)
    assert decision['training_budget_complete'] is True
    assert decision['final_pipeline_complete'] is False
    assert decision['final_screen_status'] == 'skipped_incomplete_pipeline'
    assert decision['final_development_completed'] is False
    assert decision['feature_use_completed'] is False
    assert any('Round 5' in reason and 'crashes' in reason for reason in decision['reasons'])
    assert not any('Evidence validation failed' in reason or 'No such file' in reason
                   or 'training budget was not completed' in reason for reason in decision['reasons'])
    assert not decision['passed'] and not decision['acceptance'] and not decision['policy_promoted']


def test_stopped_report_hashes_all_available_partial_artifacts_without_final_outputs(tmp_path):
    artifacts = ['protocol.json','result.json','policy_05.npz','bounded_partial.npz',
                 'bounded_partial-state.pt','guard_05/randomised/9430000.npz',
                 'guard_05/summary.json','estimated_trajectories/11242239.npz',
                 'source/tools/run_estimated_motion_trial.py']
    for name in artifacts:
        path = tmp_path/name; path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(('saved '+name).encode())
    decision = trial.final_screen(tmp_path,stopped_training_complete(),None,None,smoke=False)
    for name in artifacts:
        assert decision['evidence'][str(tmp_path/name)] == trial.sha(tmp_path/name)
    assert str(tmp_path/'decision.json') not in decision['evidence']
    assert not decision['passed']


def frozen_artifact_fixture(tmp_path, monkeypatch):
    source, estimator = tmp_path/'source.zip', tmp_path/'estimator.pt'
    source.write_bytes(b'original source'); estimator.write_bytes(b'frozen estimator')
    monkeypatch.setattr(trial,'SOURCE',source); monkeypatch.setattr(trial,'ESTIMATOR',estimator)
    monkeypatch.setattr(trial,'PROFILE',SimpleNamespace(sha256=trial.sha(source)))
    estimator_digest = trial.sha(estimator)
    monkeypatch.setattr(trial,'checkpoint_manifest',lambda:dict(estimator_sha256=estimator_digest))
    monkeypatch.setattr(trial,'assert_frozen',lambda *args: None)
    return source,estimator


def test_stopped_run_records_successful_frozen_verification(tmp_path,monkeypatch):
    source,estimator = frozen_artifact_fixture(tmp_path,monkeypatch)
    result = stopped_training_complete(); result['frozen_unchanged']=False
    assert trial.record_frozen_integrity(result,object(),'state-digest') is True
    assert result['status']=='stopped' and result['error']=='Retention regression'
    assert result['frozen_weights_verified'] is True
    assert all(row['unchanged'] for row in result['frozen_artifact_checks'].values())
    assert result['frozen_artifact_checks'][str(source)]['actual_sha256']==trial.sha(source)


@pytest.mark.parametrize('failure',['source','estimator','weights'])
def test_frozen_integrity_failure_cannot_be_reported_unchanged(tmp_path,monkeypatch,failure):
    source,estimator=frozen_artifact_fixture(tmp_path,monkeypatch)
    if failure=='source': source.write_bytes(b'changed')
    elif failure=='estimator': estimator.write_bytes(b'changed')
    else:
        def fail(*args): raise ValueError('Frozen weights changed')
        monkeypatch.setattr(trial,'assert_frozen',fail)
    result=stopped_training_complete(); result['frozen_unchanged']=True
    assert trial.record_frozen_integrity(result,object(),'digest') is False
    assert result['frozen_unchanged'] is False and result['status']=='failed'
    assert 'integrity_error' in result
    assert trial.completion_reasons(result,False)


def test_training_finished_before_interrupted_guard_is_reported_separately():
    result=stopped_training_complete(); result['iterations']=result['iterations'][:-1]
    assert trial.training_budget_complete(result,False)
    assert not trial.pipeline_complete(result)
    assert trial.completion_reasons(result,False)
    result['completed_training_rounds']=4
    assert not trial.training_budget_complete(result,False)


def endpoint(length, *, passes=2, terminated=False, wall_steps=0):
    return dict(steps=length, full_body_passes=passes, terminated=terminated,
                truncated=not terminated, wall_steps=wall_steps)


def test_credit_targets_keep_default_mc_and_change_only_declared_horizon():
    rewards = np.array([2., 0., -4.], np.float32)
    actual = trial.episode_targets(rewards, endpoint(3), trial.DEFAULT_TARGET)
    np.testing.assert_array_equal(actual, trial.discounted_returns(rewards, .995))
    np.testing.assert_allclose(actual, [2-4*.995**2, -4*.995, -4], rtol=1e-6)
    np.testing.assert_array_equal(trial.episode_targets(rewards, endpoint(3), 'full_horizon'), [-2,-4,-4])
    np.testing.assert_array_equal(trial.episode_targets(rewards, endpoint(3), 'local_credit'), actual)
    with pytest.raises(ValueError,match='Unknown'): trial.target_spec('arbitrary_gamma')


@pytest.mark.parametrize('length,passes,terminated,walls,success', [
    (600,2,False,0,True), (600,3,False,0,True), (599,2,False,0,False),
    (600,1,False,0,False), (600,2,True,0,False), (600,2,False,1,False)])
def test_mission_targets_require_full_clearance_and_whole_clean_horizon(length,passes,terminated,walls,success):
    final = endpoint(length,passes=passes,terminated=terminated,wall_steps=walls)
    actual = trial.episode_targets(np.arange(length,dtype=np.float32),final,'mission')
    np.testing.assert_array_equal(actual,np.full(length,float(success),np.float32))
    # Dense reward scale/sign cannot silently enter the binary mission target.
    np.testing.assert_array_equal(actual,trial.episode_targets(np.full(length,-1e6,np.float32),final,'mission'))


def test_partial_endpoint_cannot_supply_a_mission_or_dense_training_target():
    final = endpoint(3); final.update(terminated=False,truncated=False)
    for mode in (trial.DEFAULT_TARGET,*trial.COHORT_ARMS):
        with pytest.raises(ValueError,match='complete real episode'):
            trial.episode_targets(np.zeros(3),final,mode)


def test_local_credit_zero_tail_and_episode_isolation_match_manual_values():
    rewards = np.array([0.,5.,0.,-7.],np.float32)
    values = torch.tensor([2.,3.,4.,5.])
    targets = torch.tensor([.995*5,5.,-.995*7,-7.])
    raw, normalized = trial.actor_advantages(targets,values,rewards,[2,2],'local_credit')
    expected = torch.tensor([.995*3-2 + .995*.95*2,2., .995*5-4 + .995*.95*-12,-12.])
    torch.testing.assert_close(raw,expected)
    changed = rewards.copy(); changed[2:] = 1000
    changed_values = values.clone(); changed_values[2:] = -1000
    other,_ = trial.actor_advantages(targets,changed_values,changed,[2,2],'local_credit')
    torch.testing.assert_close(other[:2],raw[:2],rtol=0,atol=0)
    torch.testing.assert_close(normalized,(expected-expected.mean())/expected.std(unbiased=False))


def test_default_mc_advantage_arithmetic_remains_bitwise_identical():
    target=torch.tensor([1.234,-4.321,.111,31.0]); values=torch.tensor([.5,-.3,.2,5.2])
    expected=target-values
    old=(expected-expected.mean())/(expected.std(unbiased=False)+1e-8)
    raw,actual=trial.actor_advantages(target,values,np.zeros(4),[2,2],trial.DEFAULT_TARGET)
    assert torch.equal(raw,expected) and torch.equal(actual,old)


def test_complete_parallel_schedule_is_checked_against_authorized_config_not_old_main():
    seeds=list(range(11244000,11244240)); result=completed_result(); result['completed_seeds']=seeds
    assert not trial.training_budget_complete(result,False)
    assert trial.training_budget_complete(result,False,seeds)
    assert trial.completion_reasons(result,False,seeds)==[]
    assert trial.completion_reasons(result,False,seeds[:-1])
    result['completed_seeds']=list(trial.MAIN_SEEDS)
    assert not trial.training_budget_complete(result,False,seeds)


def test_parent_stop_or_lost_lease_stops_budget_before_more_work(tmp_path):
    checked=[]
    def lost_parent():
        checked.append(True)
        raise RuntimeError('parent exited')
    budget=trial.Budget(tmp_path,60,started=0,clock=lambda: 1.,parent_check=lost_parent)
    with pytest.raises(trial.TrialStopped,match='Cohort lease stopped: parent exited'): budget.check()
    assert checked==[True]


def cohort_fixture(tmp_path, *, smoke=False):
    manifest=tmp_path/'manifest.json'; write(manifest,dict(cohort='unit-only'))
    args=SimpleNamespace(cohort_manifest=manifest,arm='full_horizon',smoke=smoke)
    config=dict(training_seeds=list(range(11244900,11244902) if smoke else range(11244000,11244240)),
        algorithm_seed=11592000,memory_fraction=.15,cohort_id='fixture',target_mode='full_horizon',
        max_total_seconds=600 if smoke else 3600,cohort_manifest_sha256=trial.sha(manifest))
    lease=SimpleNamespace(config=config,check=lambda:None)
    return args,lease


def test_cli_requires_cohort_and_arm_together_and_default_config_preserved(tmp_path):
    required=['--out',str(tmp_path/'out'),'--initialization-report',str(tmp_path/'initial.json')]
    args=trial.parse_args(required)
    config=trial.execution_config(args,None)
    assert config['training_seeds']==list(trial.MAIN_SEEDS)
    assert config['algorithm_seed']==trial.ALGORITHM_SEED
    assert config['target_mode']==trial.DEFAULT_TARGET and config['memory_fraction'] is None
    with pytest.raises(SystemExit): trial.parse_args(required+['--arm','mission'])
    with pytest.raises(SystemExit): trial.parse_args(required+['--cohort-manifest','x.json'])


@pytest.mark.parametrize('change',['none','target','seeds','algorithm','memory','manifest'])
def test_authenticated_worker_configuration_is_bound_to_declared_three_arm_contract(tmp_path,change):
    args,lease=cohort_fixture(tmp_path)
    if change=='target': lease.config['target_mode']='mission'
    elif change=='seeds': lease.config['training_seeds']=list(trial.MAIN_SEEDS)
    elif change=='algorithm': lease.config['algorithm_seed']+=1
    elif change=='memory': lease.config['memory_fraction']=1.0
    elif change=='manifest': args.cohort_manifest.write_bytes(b'changed')
    if change=='none': assert trial.execution_config(args,lease)==lease.config
    else:
        with pytest.raises(ValueError,match='execution contract'): trial.execution_config(args,lease)


def test_worker_obtains_live_lease_and_sets_memory_limit_before_any_preparation(tmp_path,monkeypatch):
    args,lease=cohort_fixture(tmp_path,smoke=True); order=[]
    @contextmanager
    def scope(_args):
        order.append('lease')
        yield lease
    monkeypatch.setattr(trial,'execution_lease',scope)
    monkeypatch.setattr(torch.cuda,'is_available',lambda:order.append('cuda_available') or True)
    monkeypatch.setattr(torch.cuda,'set_per_process_memory_fraction',
                        lambda fraction,device:order.append(('memory',fraction,device)))
    monkeypatch.setattr(torch,'get_num_interop_threads',lambda:8)
    monkeypatch.setattr(torch,'set_num_interop_threads',lambda count:order.append(('interop',count)))
    def before_fitting(*_args):
        order.append('preparation')
        raise trial.TrialStopped('unit fixture stop before any model load')
    monkeypatch.setattr(trial,'verify_initialization',before_fitting)
    trial.main(['--out',str(tmp_path/'worker'),'--initialization-report',str(tmp_path/'initial.json'),
                '--cohort-manifest',str(args.cohort_manifest),'--arm','full_horizon','--smoke'])
    assert order==['lease','cuda_available',('memory',.15,0),('interop',1),'preparation']
    result=json.loads((tmp_path/'worker/result.json').read_text())
    decision=json.loads((tmp_path/'worker/decision.json').read_text())
    assert result['status']=='stopped' and result['completed_seeds']==[]
    assert decision['decision']=='software_only' and not decision['passed']


def test_peak_memory_reports_both_allocator_measures_without_initializing_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:False)
    def forbidden(*args): raise AssertionError('Do not initialize CUDA for a stopped preparation')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',forbidden)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',forbidden)
    assert trial.cuda_memory_peaks()==dict(initialized=False,peak_allocated_bytes=None,peak_reserved_bytes=None)
    monkeypatch.setattr(torch.cuda,'is_initialized',lambda:True)
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda device:1234)
    monkeypatch.setattr(torch.cuda,'max_memory_reserved',lambda device:4096)
    assert trial.cuda_memory_peaks()==dict(initialized=True,peak_allocated_bytes=1234,peak_reserved_bytes=4096)


@pytest.mark.parametrize('cancel',[False,True])
def test_first_rollout_parity_approval_precedes_any_value_optimizer_update(tmp_path,monkeypatch,cancel):
    out=tmp_path/'worker';out.mkdir()
    np.savez(out/'rollout_01.npz',features89=np.zeros((2,89),np.float32),
             actions=np.zeros((2,4),np.float32),episode_lengths=[2])
    write(out/'episode_endpoints_01.json',[endpoint(2)])
    approval=tmp_path/'first_rollout_parity.json'
    hashes={arm:trial.sha(out/'rollout_01.npz') for arm in trial.COHORT_ARMS}
    write(approval,dict(passed=True,matched=True,cohort_manifest_sha256='manifest-digest',rollout_sha256=hashes))
    order=[]
    def wait():
        assert (out/'rollout_01.npz').exists() and (out/'episode_endpoints_01.json').exists()
        order.append('parity')
        if cancel: raise RuntimeError('parent STOP before parity')
        return dict(path=str(approval),sha256=trial.sha(approval))
    lease=SimpleNamespace(wait_first_rollout=wait,config=dict(target_mode='full_horizon',
                          cohort_manifest_sha256='manifest-digest'))
    value=torch.nn.Linear(1,1);optimizer=torch.optim.Adam(value.parameters(),lr=.01)
    original={name:tensor.clone() for name,tensor in value.state_dict().items()}
    step=optimizer.step
    def tracked_step(*args,**kwargs):
        order.append('value_update')
        return step(*args,**kwargs)
    monkeypatch.setattr(optimizer,'step',tracked_step)
    result={};budget=SimpleNamespace(check=lambda:None)
    def run():
        trial.fit_value_baseline(value,optimizer,torch.ones((2,1)),torch.zeros(2),2,budget,
            lease=lease,iteration=1,out=out,result=result)
    if cancel:
        with pytest.raises(trial.TrialStopped,match='parity barrier stopped'):run()
        assert order==['parity'] and optimizer.state_dict()['state']=={}
        assert all(torch.equal(tensor,original[name]) for name,tensor in value.state_dict().items())
        assert 'first_rollout_parity' not in result
    else:
        run()
        assert order==['parity','value_update','value_update']
        assert result['first_rollout_parity']['sha256']==trial.sha(out/'first_rollout_parity.json')


def test_changed_or_mismatched_rollout_approval_cannot_authorize_training(tmp_path):
    rollout=tmp_path/'rollout_01.npz';rollout.write_bytes(b'raw rollout fixture')
    approval=tmp_path/'approval.json'
    content=dict(passed=True,matched=True,cohort_manifest_sha256='manifest-digest',
                 rollout_sha256={arm:trial.sha(rollout) for arm in trial.COHORT_ARMS})
    write(approval,content)
    record=dict(path=str(approval),sha256=trial.sha(approval))
    config=dict(target_mode='mission',cohort_manifest_sha256='manifest-digest')
    trial.verify_first_rollout_approval(record,tmp_path,config)
    rollout.write_bytes(b'changed rollout')
    with pytest.raises(ValueError,match='does not bind'):trial.verify_first_rollout_approval(record,tmp_path,config)
    content['passed']=False;write(approval,content)
    with pytest.raises(ValueError,match='approval changed'):trial.verify_first_rollout_approval(record,tmp_path,config)


def test_evaluation_loading_and_episode_sampling_preserve_next_training_rng(tmp_path,monkeypatch):
    from tools import check_estimated_motion_initialization as initialization
    checkpoint=tmp_path/'head.npz'; checkpoint.write_bytes(b'unit identity only')
    def consume_rng():
        return random.random(),np.random.random(),torch.rand(3)
    def load(*args,**kwargs):
        consume_rng()
        return object()
    monkeypatch.setattr(trial,'EstimatedMotionPolicy',SimpleNamespace(load=load))
    monkeypatch.setattr(initialization,'InitializationBridge',lambda *args,**kwargs:object())
    cuda_states=[torch.tensor([7],dtype=torch.uint8)]
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:cuda_states.copy())
    restored=[]
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda states:restored.append(states))
    def fixture_episode(bridge,suite,seed,path):
        consume_rng(); path.write_bytes(b'synthetic unit episode, no simulator')
        return dict(seed=seed,steps=600,outcome='timeout',wall_steps=0,full_body_passes=2,
            returns_to_front=1,crashed=False,clean_first_pass=True,clean_repeated_episode=True)
    monkeypatch.setattr(trial,'episode',fixture_episode)
    random.seed(314);np.random.seed(314);torch.manual_seed(314)
    state=trial.rng_state(); expected=consume_rng()
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch'])
    trial.evaluate_head(checkpoint,tmp_path/'eval',2,SimpleNamespace(check=lambda:None))
    actual=consume_rng()
    assert actual[:2]==expected[:2] and torch.equal(actual[2],expected[2])
    assert len(restored)==1 and torch.equal(restored[0][0],cuda_states[0])
