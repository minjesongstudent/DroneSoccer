"""Bounded critic-only fit; trajectory-family holdout, no actor optimization."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,shutil,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'diagnostics/acquisition-return-mixture-20260928'
OUT=ROOT/'diagnostics/recurrent-value-calibration-20260928'
sys.path.insert(0,str(DATA/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory

def main():
    OUT.mkdir(exist_ok=False);shutil.copytree(DATA/'source_snapshot',OUT/'source_snapshot');shutil.copy2(__file__,OUT/'runner.py')
    torch.set_num_threads(1);torch.manual_seed(987000)
    model=RecurrentPPO.load(DATA/'model_0.zip',device='cuda');p=model.policy
    assert not p.shared_lstm and p.lstm_critic is not None
    critic_names=tuple(n for n,_ in p.named_parameters() if n.startswith(('lstm_critic.','mlp_extractor.value_net.','value_net.')))
    frozen={n:v.detach().clone() for n,v in p.named_parameters() if n not in critic_names}
    for n,v in p.named_parameters():v.requires_grad_(n in critic_names)
    optimizer=torch.optim.Adam([v for v in p.parameters() if v.requires_grad],lr=1e-3)
    # Hold out whole source flight families 0 and 3, including their derived align snapshots.
    heldout=[0,3,8,9,16,20,23,25];fit=[i for i in range(27) if i not in heldout]
    entries=json.loads((DATA/'protocol.json').read_text())['snapshots']
    for i in (8,9):assert entries[i]['snapshot'] in (0,3)
    for i in range(8,15):assert (i in heldout)==(entries[i]['snapshot'] in (0,3))
    arrays=[np.load(DATA/f'rollout_0_{i}.npz') for i in range(27)]
    protocol=dict(data=str(DATA),batch=0,fit=fit,heldout=heldout,epochs=16,lr=.001,
        target='Recorded gamma .999 Monte Carlo returns divided by40; critic outputs scaled units, not drop-in raw-value PPO',
        actor_frozen=True,critic_names=critic_names,selection='All checkpoints reported, heldout not optimized',
        holdout_grouping='Parent source flight families 0/3 plus separate natural episodes; derived snapshots stay with parent',
        source_sha256=hashlib.sha256((DATA/'model_0.zip').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def predict(i,grad=False):
        a=arrays[i];natural=bool(a['natural_start']);p.set_training_mode(grad)
        state,_=_initial_memory(model,a['prefix'],natural)
        obs=torch.as_tensor(a['observations'],device='cuda');starts=torch.zeros(len(obs),device='cuda')
        if natural:starts[0]=1
        return p.predict_values(obs,state.vf,starts).flatten()
    def metrics(indices):
        with torch.no_grad():
            predictions=np.concatenate([predict(i).cpu().numpy()*40 for i in indices])
        targets=np.concatenate([arrays[i]['returns'] for i in indices])
        constant=float(np.concatenate([arrays[i]['returns'] for i in fit]).mean())
        return dict(frames=len(targets),rmse=float(np.sqrt(np.mean((predictions-targets)**2))),
            explained_variance=float(1-np.var(targets-predictions)/np.var(targets)),
            constant_fit_mean_rmse=float(np.sqrt(np.mean((targets-constant)**2))))
    history=[]
    for epoch in range(17):
        if epoch in (0,4,8,16):
            result=dict(epoch=epoch,fit=metrics(fit),heldout=metrics(heldout))
            history.append(result);model.save(OUT/f'model_{epoch}')
            (OUT/'metrics.json').write_text(json.dumps(history,indent=2));print(json.dumps(result),flush=True)
        if epoch==16:break
        for i in np.random.default_rng(987000+epoch).permutation(fit):
            values=predict(i,grad=True);targets=torch.as_tensor(arrays[i]['returns']/40,device='cuda')
            loss=(values-targets).square().mean();assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad],1.)
            optimizer.step()
        assert all(torch.equal(v,dict(p.named_parameters())[n]) for n,v in frozen.items())
    torch.save(optimizer.state_dict(),OUT/'critic_optimizer.pt')
    (OUT/'result.json').write_text(json.dumps(dict(history=history,actor_parameters_exact=True,actor_updates=0,
        raw_value_scale=40,flight_improvement_claimed=False),indent=2))
if __name__=='__main__':main()
