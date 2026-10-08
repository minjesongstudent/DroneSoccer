from pathlib import Path
import pytest
import src.viewer as viewer

ROOT=Path(__file__).resolve().parents[1]


def test_selected_policy_viewer_runs_complete_episode(monkeypatch):
    states=[]
    class Complete(Exception):pass
    def sleep(_):
        if 'outcome' in viewer.STATE:
            states.append(dict(viewer.STATE))
            if viewer.STATE['outcome'] in ('hit','net','wrong_way','timeout'):
                raise Complete('test captured full episode')
    monkeypatch.setattr(viewer.time,'sleep',sleep)
    viewer.worker(ROOT/'trainedModels/seenflag-s1-level1.zip')
    assert len(states)>1 and states[-1]['policy_inputs']==85
    assert states[-1]['outcome'] in ('hit','net','wrong_way','timeout')
    assert states[-1]['passes']<=states[-1]['centre_passes']
    assert states[-1]['seconds']>0


def test_viewer_rejects_guessed_profile(tmp_path):
    with pytest.raises(ValueError,match='recorded observation history'):
        viewer.load_replay_policy(tmp_path/'unknown.zip')
