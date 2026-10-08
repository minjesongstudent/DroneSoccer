"""Admission continuation is allowed only after complete, verified software smoke."""
import json
import pytest
from tools.parallel_training import ARMS, digest
from tools.queue_parallel_credit import smoke_ready


def put(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data))


def fixture(path):
    put(path/'cohort_manifest.json',{'smoke':True})
    workers = {}
    for arm in ARMS:
        folder=path/'arms'/arm
        put(folder/'result.json',dict(status='complete',frozen_unchanged=True,
            training_budget_complete=True,final_pipeline_complete=True,
            completed_seeds=[11244900,11244901]))
        put(folder/'decision.json',dict(reasons=['Software-only smoke can never pass'],
            evidence={str(folder/'result.json'):digest(folder/'result.json')}))
        workers[arm]=dict(exit_code=0,decision='software_only',passed=False,
            decision_sha256=digest(folder/'decision.json'))
    decision=dict(status='complete',errors=[],smoke=True,selected=None,fairness={'matched':True},
        cohort_manifest_sha256=digest(path/'cohort_manifest.json'),workers=workers)
    put(path/'decision.json',decision)
    return decision


def test_complete_software_smoke_allows_next_bounded_stage(tmp_path):
    fixture(tmp_path)
    assert smoke_ready(tmp_path)==digest(tmp_path/'decision.json')


@pytest.mark.parametrize('change',['mismatch','worker_failure','changed_evidence','extra_validation_error'])
def test_no_main_after_incomplete_or_changed_smoke(tmp_path,change):
    data=fixture(tmp_path)
    if change=='mismatch': data['fairness']['matched']=False
    elif change=='worker_failure': data['workers']['mission']['exit_code']=1
    elif change=='changed_evidence':
        put(tmp_path/'arms/mission/result.json',{'status':'failed'})
    else:
        path=tmp_path/'arms/mission/decision.json'
        saved=json.loads(path.read_text());saved['reasons'].append('Evidence failed')
        put(path,saved);data['workers']['mission']['decision_sha256']=digest(path)
    put(tmp_path/'decision.json',data)
    with pytest.raises((ValueError,KeyError)): smoke_ready(tmp_path)
