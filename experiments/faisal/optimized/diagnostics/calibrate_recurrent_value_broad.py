"""Bounded critic-only fit; trajectory-family holdout, no actor optimization."""
from pathlib import Path
from types import SimpleNamespace
import sys,json,shutil,hashlib
import numpy as np,torch
from sb3_contrib import RecurrentPPO
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'diagnostics/acquisition-return-mixture-20260928'
OUT=ROOT/'diagnostics/recurrent-value-broad-20260928'
sys.path.insert(0,str(DATA/'source_snapshot'))
from src.recurrent_snapshot_rollout import _initial_memory

def main():
    OUT.mkdir(exist_ok=False);shutil.copytree(DATA/'source_snapshot',OUT/'source_snapshot');shutil.copy2(__file__,OUT/'runner.py')
    torch.set_num_threads(1);torch.manual_seed(989000)
    model=RecurrentPPO.load(DATA/'model_0.zip',device='cuda');p=model.policy
    assert not p.shared_lstm and p.lstm_critic is not None
    critic_names=tuple(n for n,_ in p.named_parameters() if n.startswith(('lstm_critic.','mlp_extractor.value_net.','value_net.')))
    frozen={n:v.detach().clone() for n,v in p.named_parameters() if n not in critic_names}
    for n,v in p.named_parameters():v.requires_grad_(n in critic_names)
    optimizer=torch.optim.Adam([v for v in p.parameters() if v.requires_grad],lr=1e-3)
    coverage=ROOT/'diagnostics/value-natural-coverage-20260928'
    entries=json.loads((DATA/'protocol.json').read_text())['snapshots']
    records=[]
    for i in range(15):
        held=i in (0,3,8,9)
        if i>=8:assert held==(entries[i]['snapshot'] in (0,3))
        records.append(dict(path=str(DATA/f'rollout_0_{i}.npz'),split='heldout' if held else 'fit',
            group='early_return' if i<8 else 'reapproach'))
    for case in json.loads((coverage/'protocol.json').read_text())['cases']:
        records.append(dict(path=str(coverage/f"episode_{case['index']}.npz"),split=case['split'],group=case['group']))
    arrays=[]
    for record in records:
        record['sha256']=hashlib.sha256(Path(record['path']).read_bytes()).hexdigest()
        with np.load(record['path']) as archive:arrays.append({k:archive[k] for k in archive.files})
    fit=[i for i,r in enumerate(records) if r['split']=='fit']
    heldout=[i for i,r in enumerate(records) if r['split']=='heldout']
    assert len(fit)==71 and len(heldout)==24 and not set(fit)&set(heldout)
    groups=sorted({r['group'] for r in records})
    protocol=dict(records=records,fit=fit,heldout=heldout,epochs=16,lr=.001,
        target='Recorded gamma .999 Monte Carlo returns divided by40; multiply predictions by40 for raw units',
        actor_frozen=True,critic_names=critic_names,selection='All checkpoints reported; no heldout optimization or best-checkpoint selection',
        holdout_grouping='Old parent source flights 0/3 including derived snapshots; new predeclared every fourth episode',
        weighting='One mean squared loss per episode per epoch, shuffled; metrics frame weighted within category',
        source_sha256=hashlib.sha256((DATA/'model_0.zip').read_bytes()).hexdigest())
    (OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
    def predict(i,grad=False):
        a=arrays[i];natural=bool(a['natural_start']);p.set_training_mode(grad)
        state,_=_initial_memory(model,a['prefix'],natural)
        obs=torch.as_tensor(a['observations'],device='cuda');starts=torch.zeros(len(obs),device='cuda')
        if natural:starts[0]=1
        return p.predict_values(obs,state.vf,starts).flatten()
    def metrics(indices,category=None):
        with torch.no_grad():
            predictions=np.concatenate([predict(i).cpu().numpy()*40 for i in indices])
        targets=np.concatenate([arrays[i]['returns'] for i in indices])
        constant=float(np.concatenate([arrays[i]['returns'] for i in fit]).mean())
        category_fit=[i for i in fit if category is None or records[i]['group']==category]
        category_constant=float(np.concatenate([arrays[i]['returns'] for i in category_fit]).mean())
        return dict(episodes=len(indices),frames=len(targets),category_fit_constant_rmse=float(np.sqrt(np.mean((targets-category_constant)**2))),rmse=float(np.sqrt(np.mean((predictions-targets)**2))),
            explained_variance=float(1-np.var(targets-predictions)/np.var(targets)),
            constant_fit_mean_rmse=float(np.sqrt(np.mean((targets-constant)**2))))
    history=[]
    for epoch in range(17):
        if epoch in (0,4,8,16):
            result=dict(epoch=epoch,actor_parameters_exact=True,fit=metrics(fit),heldout=metrics(heldout),
                categories={g:{split:metrics([i for i in indices if records[i]['group']==g],g)
                    for split,indices in [('fit',fit),('heldout',heldout)]} for g in groups})
            history.append(result);model.save(OUT/f'model_{epoch}')
            (OUT/'metrics.json').write_text(json.dumps(history,indent=2));print(json.dumps(result),flush=True)
        if epoch==16:break
        for i in np.random.default_rng(989000+epoch).permutation(fit):
            values=predict(i,grad=True);targets=torch.as_tensor(arrays[i]['returns']/40,device='cuda')
            loss=(values-targets).square().mean();assert torch.isfinite(loss)
            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_([v for v in p.parameters() if v.requires_grad],1.)
            optimizer.step()
        assert all(torch.equal(v,dict(p.named_parameters())[n]) for n,v in frozen.items())
    torch.save(optimizer.state_dict(),OUT/'critic_optimizer.pt')
    (OUT/'result.json').write_text(json.dumps(dict(history=history,actor_parameters_exact=True,actor_updates=0,
        raw_value_scale=40,flight_improvement_claimed=False),indent=2))
if __name__=='__main__':main()
