import numpy as np
import pytest
import torch
from sb3_contrib import RecurrentPPO
from src.recurrent_baseline import CompleteFlightMixture,RecurrentPredictor
from src.unified_evaluation import evaluate

def test_episode_state_carry_and_reset():
    class Stub:
        num_timesteps=0
        def predict(self,obs,state,episode_start,deterministic):
            assert bool(episode_start[0]) == (state is None)
            return np.zeros(4), 1 if state is None else state+1
    p=RecurrentPredictor(Stub())
    assert p.predict(None)[1]==1 and p.predict(None)[1]==2
    p.reset_episode();assert p.predict(None)[1]==1

def test_cuda_training_save_load_and_memory(tmp_path):
    if not torch.cuda.is_available():pytest.skip('CUDA required')
    torch.set_num_threads(1)
    env=CompleteFlightMixture()
    model=RecurrentPPO('MlpLstmPolicy',env,device='cuda',n_steps=32,batch_size=16,n_epochs=1,
        seed=5,policy_kwargs=dict(lstm_hidden_size=16,net_arch=dict(pi=[16],vf=[16])))
    before={k:v.clone() for k,v in model.policy.state_dict().items()}
    model.learn(64)
    assert next(model.policy.parameters()).is_cuda
    assert any(not torch.equal(v,model.policy.state_dict()[k]) for k,v in before.items())
    predictor=RecurrentPredictor(model);obs,_=env.reset(seed=33)
    first,state=predictor.predict(obs)
    assert state is not None
    predictor.predict(obs);predictor.reset_episode()
    assert np.array_equal(first,predictor.predict(obs)[0])
    model.save(tmp_path/'model')
    loaded=RecurrentPPO.load(tmp_path/'model',device='cuda')
    assert np.array_equal(first,RecurrentPredictor(loaded).predict(obs)[0])
    calls=[]
    def reset():calls.append(1);predictor.reset_episode()
    report=evaluate(predictor,4,episodes=1,metric_ring=True,geometric_return=True,episode_reset=reset)
    assert len(calls)==5 and report['episodes_per_suite']==1
    env.close()
