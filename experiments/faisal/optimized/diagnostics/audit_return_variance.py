"""FIT-only repeated authentic returns; unchanged actor, no optimization."""
from pathlib import Path
import sys,json,shutil,hashlib,pickle
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'diagnostics/acquisition-return-mixture-20260928'
OUT=ROOT/'diagnostics/authentic-return-variance-20260928'
sys.path.insert(0,str(SOURCE/'source_snapshot'))
from src.recurrent_snapshot_rollout import collect_segment,_initial_memory

def main():
    OUT.mkdir(exist_ok=False);shutil.copytree(SOURCE/'source_snapshot',OUT/'source_snapshot');shutil.copy2(__file__,OUT/'runner.py')
    torch.set_num_threads(1)
    source=SOURCE/'model_0.zip';digest=hashlib.sha256(source.read_bytes()).hexdigest()
    model=RecurrentPPO.load(source,device='cuda')
    critics={name:RecurrentPPO.load(ROOT/f'diagnostics/recurrent-value-{name}-20260928/model_16.zip',device='cuda') for name in ('broad','balanced')}
    entries=json.loads((SOURCE/'protocol.json').read_text())['snapshots']
    indices=[1,2,4,5,6,7,10,11,12,13,14]
    for i in indices:
        if i>=8:assert entries[i]['snapshot'] not in (0,3)
    (OUT/'protocol.json').write_text(json.dumps(dict(indices=indices,entries=[entries[i] for i in indices],repeats=8,
        torch_seed_rule='993000+100*entry_index+repeat',source_sha256=digest,gamma=.999,fit_only=True,optimization_steps=0),indent=2))
    rows=[];summaries=[]
    for i in indices:
        entry=entries[i];payload=(Path(entry['bank'])/'snapshots'/f"{entry['id']}.pkl").read_bytes()
        assert hashlib.sha256(payload).hexdigest()==entry['sha256']
        d=pickle.loads(payload);predictions={}
        with torch.no_grad():
            for name,critic in critics.items():
                critic.policy.set_training_mode(False)
                state,start=_initial_memory(critic,d['prefix_observations'],False)
                predictions[name]=float(critic.policy.predict_values(torch.as_tensor(d['observation'][None],device='cuda'),state.vf,start).item()*40)
        d['environment'].close()
        for repeat in range(8):
            d=pickle.loads(payload);env=d['environment'];initial=int(env.unwrapped.passes)
            seed=993000+100*i+repeat;torch.manual_seed(seed)
            s=collect_segment(model,env,d['observation'],d['prefix_observations'],600);assert s.mission_ended
            assert np.array_equal(s.observations[0],d['observation']) and np.array_equal(s.prefix,d['prefix_observations'])
            returns=np.zeros(len(s.rewards),np.float32);carry=0.
            for j in range(len(returns)-1,-1,-1):carry=float(s.rewards[j])+.999*carry;returns[j]=carry
            np.savez_compressed(OUT/f'case_{i}_repeat_{repeat}.npz',prefix=s.prefix,observations=s.observations,raw_actions=s.actions,rewards=s.rewards,returns=returns,natural_start=False)
            rows.append(dict(index=i,repeat=repeat,torch_seed=seed,steps=len(returns),outcome=s.final_info['outcome'],new_passes=int(s.final_info['passes'])-initial,initial_return=float(returns[0])))
            env.close()
        these=[r for r in rows if r['index']==i];values=np.array([r['initial_return'] for r in these])
        summaries.append(dict(index=i,group='early_return' if i<8 else 'reapproach',mean=float(values.mean()),std=float(values.std(ddof=1)),
            new_pass_episodes=sum(r['new_passes']>0 for r in these),crashes=sum(r['outcome'] in ('hit','left_cage') for r in these),predictions=predictions))
        (OUT/'progress.json').write_text(json.dumps(dict(rows=rows,summaries=summaries),indent=2));print('completed',i,flush=True)
    metrics={g:{name:float(np.sqrt(np.mean([(s['predictions'][name]-s['mean'])**2 for s in summaries if s['group']==g]))) for name in critics} for g in ('early_return','reapproach')}
    assert hashlib.sha256(source.read_bytes()).hexdigest()==digest
    result=dict(rows=rows,summaries=summaries,rmse_against_repeat_means=metrics,actor_unchanged=True,optimization_steps=0)
    (OUT/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(summaries=summaries,metrics=metrics)),flush=True)
if __name__=='__main__':main()
