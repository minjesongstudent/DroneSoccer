import numpy as np
import pytest
import torch

from tools.probe_parameter_signal import score_endpoint, scaled_direction, signal_decision, config


def endpoint(**changes):
    return dict(dict(absolute_steps=600,terminated=False,truncated=True,wall_steps=0,full_body_passes=4),**changes)


def test_complete_passes_do_not_hide_crash_or_wall():
    assert score_endpoint(endpoint())==dict(score=114,first=1,repeat=1,crash=0)
    for e in [endpoint(absolute_steps=599,terminated=True,truncated=False),endpoint(wall_steps=1)]:
        assert score_endpoint(e)==dict(score=-6,first=1,repeat=0,crash=1)
    assert score_endpoint(endpoint(full_body_passes=1))['repeat']==0
    assert score_endpoint(endpoint(full_body_passes=9))['score']==115


def test_incomplete_flight_rejected():
    with pytest.raises(ValueError):score_endpoint(endpoint(absolute_steps=599))
    with pytest.raises(ValueError):score_endpoint(endpoint(terminated=True))


def test_normalized_gaussian_directions_respect_both_limits():
    g=torch.Generator().manual_seed(32)
    direction=torch.randn((4,258),generator=g)
    features=torch.randn((100,258),generator=g)
    std=torch.randn((100,4),generator=g).clamp(-3,2)
    weight,scale=scaled_direction(direction,features,std)
    kl=(.5*((features@weight.T)/std.exp()).square()).sum(1)
    assert scale>0 and kl.mean()<.001 and kl.max()<.05
    assert torch.equal(weight,-(-weight))
    with pytest.raises(ValueError):scaled_direction(torch.zeros_like(direction),features,std)


def data(mode):
    rows=[]
    for h in ['zero']+[f'd{d:02d}{s}' for d in range(12) for s in ('p','m')]:
        for seed in range(4):
            good=h.endswith('p') and (mode=='consistent' or (mode=='reversed' and seed%2==0))
            if mode=='reversed' and h.endswith('m') and seed%2:good=True
            rows.append(dict(head=h,seed=seed,score=114 if good else -6,first=1,repeat=int(good),crash=int(not good)))
    return rows


def test_ties_cannot_manufacture_positive_signal():
    result=signal_decision(data('tied'),list(range(4)),12)
    assert not result['signal_passed'] and result['sign_agreement'] is None
    assert result['nonzero_both_halves']==0


def test_opposite_fit_halves_fail_and_consistent_improvements_pass():
    assert not signal_decision(data('reversed'),list(range(4)),12)['signal_passed']
    result=signal_decision(data('consistent'),list(range(4)),12)
    assert result['signal_passed'] and result['sign_agreement']==1
    assert len(result['useful_heads'])==12


def test_missing_duplicate_or_unknown_layout_is_rejected():
    rows=data('tied')
    for invalid in [rows[:-1],rows+[rows[0]],rows+[dict(rows[0],seed=8)]]:
        with pytest.raises(ValueError):signal_decision(invalid,list(range(4)),12)


def test_main_and_smoke_allocation_and_budget():
    main,smoke=config(False),config(True)
    assert set(main['training_seeds']).isdisjoint(smoke['training_seeds'])
    assert main['seconds']==1800 and smoke['seconds']==300
    assert (1+2*main['directions'])*len(main['training_seeds'])+2==802
    assert (1+2*smoke['directions'])*len(smoke['training_seeds'])+2==12
