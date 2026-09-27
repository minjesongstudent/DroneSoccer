"""Analytical soft Retrace checks, independent of learned critic quality."""
import numpy as np
import pytest
import torch

from src.retrace_targets import soft_retrace_targets


def target(rewards,q,values,*,terminal=None,truncated=None,valid=None,behavior=None,policy=None,
           gamma=.5,lam=1.):
    r=torch.tensor(rewards,dtype=torch.float64).reshape(1,-1)
    n=r.shape[1]
    flag=lambda value:torch.tensor(value or [False]*n,dtype=torch.bool).reshape(1,-1)
    number=lambda value:torch.tensor(value,dtype=torch.float64).reshape(1,-1)
    return soft_retrace_targets(r,flag(terminal),flag(truncated),flag(valid or [True]*n),
        number(q),number(values),number(behavior or [0.]*n),number(policy or [0.]*n),
        gamma=gamma,trace_lambda=lam).item()


def test_one_step_and_zero_lambda_are_exact_sac_targets():
    assert target([2],[99],[10],lam=.95)==7
    assert target([2,500],[99,-200],[10,300],lam=0)==7
    assert target([2],[99],[10],terminal=[True])==2
    assert target([2],[99],[10],truncated=[True])==7


def test_on_policy_lambda_one_telescopes_to_soft_return_without_current_entropy():
    # Next values use the subsequent sampled behavior action for this analytical
    # telescoping example: V1=Q1-alpha*logp1 and V2=Q2-alpha*logp2.
    actual=target([1,2,3],[50,60,70],[60+.4,70+.8,999],terminal=[False,False,True],
                  behavior=[900,-2,-4],policy=[900,-2,-4])
    assert actual==pytest.approx(1+.5*(2+.4)+.25*(3+.8))
    assert actual==pytest.approx(target([1,2,3],[50,60,70],[60+.4,70+.8,999],
        terminal=[False,False,True],behavior=[-900,-2,-4],policy=[-900,-2,-4]))


def test_timeout_keeps_bootstrap_and_masks_padding_without_crossing_reset():
    actual=target([1,2,999],[10,20,888],[20+.4,30,777],
                  truncated=[False,True,False],valid=[True,True,False])
    assert actual==pytest.approx(1+.5*(2+.4)+.25*30)
    crash=target([1,2,999],[10,20,888],[20+.4,30,777],
                  terminal=[False,True,False],valid=[True,True,False])
    assert actual-crash==pytest.approx(.25*30)


def test_importance_clipping_uses_successor_ratio_not_start_ratio():
    # delta0=0, delta1=4, gamma=.5. c1=.8*.25; c0 must be ignored.
    actual=target([0,4],[0,0],[0,0],terminal=[False,True],lam=.8,
                  behavior=[0,0],policy=[-100,np.log(.25)])
    assert actual==pytest.approx(.4)
    assert target([0,4],[0,0],[0,0],terminal=[False,True],lam=.8,
                  behavior=[0,0],policy=[-100,1000])==pytest.approx(1.6)
    assert target([0,4],[0,0],[0,0],terminal=[False,True],lam=.8,
                  behavior=[0,0],policy=[0,-1000])==0


def test_matches_explicit_eight_step_residual_sum():
    rng=np.random.default_rng(2)
    r,q,v,mu,pi=rng.normal(size=(5,8))
    gamma,lam=.995,.95
    delta=r+gamma*v-q
    weights=[1.]
    for i in range(1,8):
        weights.append(weights[-1]*gamma*lam*min(1,np.exp(pi[i]-mu[i])))
    expected=q[0]+np.dot(weights,delta)
    assert target(r.tolist(),q.tolist(),v.tolist(),behavior=mu.tolist(),policy=pi.tolist(),
                  gamma=gamma,lam=lam)==pytest.approx(expected)


@pytest.mark.parametrize('terminal,truncated,valid',[
    ([True,False],[False,False],[True,True]),
    ([False,False],[True,False],[True,True]),
    ([False,False,False],[False,False,False],[True,False,True]),
    ([False,False],[False,False],[False,True])])
def test_invalid_boundary_or_padding_sequences_fail(terminal,truncated,valid):
    n=len(valid)
    with pytest.raises(ValueError):
        target([1]*n,[0]*n,[0]*n,terminal=terminal,truncated=truncated,valid=valid)


def test_target_is_detached_even_if_target_values_require_grad():
    values=torch.ones((2,8),requires_grad=True)
    mask=torch.zeros((2,8),dtype=torch.bool)
    result=soft_retrace_targets(values,mask,mask,~mask,values,values,values,values,
                                gamma=.995,trace_lambda=.95)
    assert not result.requires_grad and result.shape==(2,)


def test_zero_lambda_is_bit_exact_float32_one_step_despite_large_baselines():
    generator=torch.Generator().manual_seed(83)
    r=torch.randn((16,8),generator=generator)
    v=torch.randn((16,8),generator=generator)
    q=torch.full_like(r,1e7)
    terminal=torch.zeros_like(r,dtype=torch.bool)
    valid=torch.ones_like(terminal)
    logp=torch.zeros_like(r)
    got=soft_retrace_targets(r,terminal,terminal,valid,q,v,logp,logp,
                             gamma=.995,trace_lambda=0.)
    assert torch.equal(got,r[:,0]+.995*v[:,0])


def test_sequences_do_not_follow_flattened_train_rows_across_development_gaps():
    from diagnostics.retrace_critic_comparison import sequence_table
    obs=np.zeros((7,100),np.float32); obs[:,0]=np.arange(7)
    next_obs=obs.copy(); next_obs[:,0]+=1
    arrays={'rewards':np.zeros(7),'observations':obs,'next_observations':next_obs,
            'flight_id':np.array([0,0,1,1,1,2,2]),'train':np.array([1,1,0,0,0,1,1],bool),
            'terminated':np.array([0,1,0,0,0,0,0],bool),
            'truncated':np.array([0,0,0,0,1,0,1],bool)}
    flights=[{'flight_id':i,'start':start,'end':end,'split':split,
              'terminal':terminal,'truncated':not terminal}
             for i,start,end,split,terminal in [(0,0,2,'train',True),(1,2,5,'development',False),(2,5,7,'train',False)]]
    sequence,valid=sequence_table(arrays,flights)
    np.testing.assert_array_equal(sequence[1],np.repeat(1,8))
    np.testing.assert_array_equal(sequence[5],np.r_[5,np.repeat(6,7)])
    np.testing.assert_array_equal(valid[[1,5,6]].sum(1),[1,2,1])
    assert arrays['train'][sequence[[1,5,6]][valid[[1,5,6]]]].all()
    arrays['next_observations'][5,0]=999
    with pytest.raises(ValueError,match='chronologically'):
        sequence_table(arrays,flights)


def test_frozen_dataset_sequence_alignment_readonly():
    import json
    from pathlib import Path
    from diagnostics.retrace_critic_comparison import sequence_table
    directory=Path('D:/DroneSoccerData/diagnostics/asymmetric-comparison-v1')
    if not (directory/'dataset.npz').exists():
        pytest.skip('Local frozen diagnostic dataset unavailable')
    with np.load(directory/'dataset.npz',allow_pickle=False) as stored:
        arrays={key:stored[key] for key in ('rewards','observations','next_observations',
                                           'flight_id','train','terminated','truncated')}
    flights=json.loads((directory/'flights.json').read_text())
    sequence,valid=sequence_table(arrays,flights)
    train=np.flatnonzero(arrays['train'])
    assert arrays['train'][sequence[train][valid[train]]].all()
    np.testing.assert_array_equal(arrays['flight_id'][sequence],np.repeat(arrays['flight_id'][:,None],8,axis=1))
    assert np.all(valid.sum(1)>=1) and np.all(valid.sum(1)<=8)
    for flight in flights:
        assert valid[flight['end']-1].sum()==1
