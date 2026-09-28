import sys,json,hashlib
from pathlib import Path
OUT=Path(__file__).parent
sys.path.insert(0,str(OUT/'source_snapshot'))
import numpy as np,torch
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv,VecNormalize
from src.recurrent_baseline import CompleteFlightMixture,RecurrentPredictor,VERSION
from src.unified_evaluation import evaluate

def save(name,value):(OUT/name).write_text(json.dumps(value,indent=2))

def fingerprint(model):
    h=hashlib.sha256()
    for k,v in model.policy.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def main():
    assert not (OUT/'protocol.json').exists()
    torch.set_num_threads(1);assert torch.cuda.is_available()
    save('protocol.json',dict(version=VERSION,algorithm='sb3-contrib2.9.0 RecurrentPPO MlpLstmPolicy',
        device=torch.cuda.get_device_name(),seed=41,fresh_initialization=True,source_weights_transferred=False,
        max_steps=131072,checkpoints=[0,32768,65536,98304,131072],n_envs=8,n_steps=256,batch_size=256,n_epochs=5,
        learning_rate=.0003,gamma=.995,gae_lambda=.95,clip_range=.2,target_kl=.01,ent_coef=.01,
        lstm_hidden_size=128,actor_critic_lstm='separate',observation_dimension=119,privileged_inputs=False,
        mixture='20% each natural close/near/centered/varied pass and front-start full-lap episodes',
        reward_normalization='VecNormalize rewards only; no observation normalization; raw rewards retained in fixed evaluation',
        evaluation='unchanged fixed5M suites25episodes, LSTM state carried within episode and reset at every true episode start',
        early_stop='at65536 if centered and varied each<5/25 and no fullmission successes; fixed budget otherwise',
        promotion='never promote unless existing passing/fullmission/three-checkpoint/fresh7M acceptance criteria satisfied',
        midflight_snapshots_used=False,reserved_bank_layouts_used=False,independent_7M_used=False))
    env=VecNormalize(DummyVecEnv([CompleteFlightMixture for _ in range(8)]),norm_obs=False,norm_reward=True,gamma=.995)
    model=RecurrentPPO('MlpLstmPolicy',env,device='cuda',seed=41,n_steps=256,batch_size=256,n_epochs=5,
        learning_rate=3e-4,gamma=.995,gae_lambda=.95,clip_range=.2,target_kl=.01,ent_coef=.01,
        policy_kwargs=dict(lstm_hidden_size=128,net_arch=dict(pi=[128],vf=[128])),verbose=0)
    assert next(model.policy.parameters()).is_cuda
    reason='bounded_budget_complete'
    for step in (0,32768,65536,98304,131072):
        if step:
            save('status.json',dict(mode='training',target=step,steps=int(model.num_timesteps)))
            model.learn(step-model.num_timesteps,reset_num_timesteps=False)
        assert model.num_timesteps==step
        model.save(OUT/f'model_{step}.zip');env.save(OUT/f'reward_normalizer_{step}.pkl')
        save('status.json',dict(mode='evaluating',steps=step))
        before=fingerprint(model);predictor=RecurrentPredictor(model)
        with torch.inference_mode():
            report=evaluate(predictor,4,metric_ring=True,geometric_return=True,episode_reset=predictor.reset_episode)
        assert fingerprint(model)==before
        save(f'evaluation_{step}.json',report)
        counts={k:v['successful_episodes'] for k,v in report['suites'].items()}
        print(step,counts,flush=True)
        if step==65536 and counts['original_centered']<5 and counts['varied_approach']<5 and not any(counts[k] for k in ('lesson','full_task','full_task_delay_200ms')):
            reason='no_early_learning_signal';break
    save('status.json',dict(mode='stopped',steps=int(model.num_timesteps),reason=reason,mission_accepted=False,physical_flight_validated=False))
    env.close()
if __name__=='__main__':main()
