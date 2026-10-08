from types import SimpleNamespace
import numpy as np
import pytest
import torch
from tools import run_dynamics_identification as trial


@pytest.mark.parametrize('curriculum,seed',[(False,11000000),(True,11241000),(True,11241001)])
def test_label_reset_matches_first_public_frame_and_never_steps(tmp_path,curriculum,seed,monkeypatch):
    env=(trial.ApproachCurriculumEnv if curriculum else trial.HoopEnv)(level=1,randomise=True,hanger=True)
    env.reset(seed=seed)
    first=trial.CameraHistory(trial.core.PROFILE.lags).update(env._frames[-1],np.zeros(4,np.float32))
    expected=np.array([env.dyn.delay_steps,env.dyn.tau],np.float32)
    path=tmp_path/'trace.npz'
    np.savez(path,hoop_centre=env.hoop_centre,hoop_normal=env.hoop_normal,drone_half_box=env._half_box)
    ep=SimpleNamespace(seed=seed,trace=str(path),observations=first[None,:])
    monkeypatch.setattr(env,'step',lambda *_:pytest.fail('Reset-label collection must never move'))
    np.testing.assert_array_equal(trial.reset_label(ep,env),expected)
    ep.observations[0,0]+=1
    with pytest.raises(ValueError,match='public observation'):trial.reset_label(ep,env)
    env.close()


def test_burn_in_and_equal_episode_weight_ignore_padding():
    pred=torch.zeros((2,100,2),requires_grad=True)
    target=torch.tensor([[1.,1.],[3.,3.]])
    loss=trial.episode_loss(pred,target,torch.tensor([33,100]))
    assert loss.item()==5  # Equal episode weights despite 1 vs68 supervised rows.
    loss.backward()
    assert not pred.grad[:,:32].any() and not pred.grad[0,33:].any()
    with pytest.raises(ValueError,match='burn-in'):trial.episode_loss(pred,target,torch.tensor([32,30]))


def test_no_truth_or_seed_columns_enter_inputs():
    eps=[SimpleNamespace(observations=np.arange(34*85,dtype=np.float32).reshape(34,85),seed=object(),truth=object())]
    norm=dict(x_mean=np.zeros(85),x_scale=np.ones(85),y_mean=np.zeros(2),y_scale=np.ones(2))
    x,y,lengths=trial.batch(eps,np.array([[4.,.6]]),np.array([0]),norm,'cpu')
    np.testing.assert_array_equal(x[0],eps[0].observations)
    assert x.shape==(1,34,85) and y.shape==(1,2) and lengths.tolist()==[34]


def test_recurrent_prefix_predictions_cannot_read_future():
    torch.manual_seed(14);model=trial.core.RecurrentEstimator();model.output=torch.nn.Linear(64,2);model.eval()
    x=torch.randn(1,40,85);later=x.clone();later[:,20:]=100
    with torch.no_grad():
        a=model(x,torch.tensor([40]));b=model(later,torch.tensor([40]))
    torch.testing.assert_close(a[:,:20],b[:,:20],rtol=0,atol=0)


def test_metrics_use_equal_episode_weight_and_exclude_burn_in():
    p=np.zeros((100,2));t=np.ones((100,2));ids=np.r_[np.zeros(33),np.ones(67)]
    steps=np.r_[np.arange(33),np.arange(67)];t[33:]=3
    m=trial.metrics(p,t,ids,steps,np.zeros(2))
    np.testing.assert_allclose(m['rmse'],np.sqrt(5))
    assert m['rows']==36 and m['episodes']==2


@pytest.mark.parametrize('failure',['none','delay','tau','prior','missing','incomplete','smoke'])
def test_gate_requires_both_parameters_and_both_cohorts(failure):
    results={v:{c:dict(rmse=[1.,1.] if v=='feedforward' else [.7,.7],prior_rmse=[1.,1.])
                for c in ('heldout','gae_separate')} for v in trial.core.VARIANTS}
    if failure in ('delay','tau'):results['gru64']['gae_separate']['rmse'][failure=='tau']=.9
    if failure=='prior':results['gru64']['heldout']['prior_rmse']=[.5,.5]
    if failure=='missing':del results['gru64']['heldout']
    result=trial.gate(results,failure!='incomplete',failure=='smoke')
    assert result['passed']==(failure=='none')
