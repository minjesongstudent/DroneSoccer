"""Measure basic/lap gradient agreement without changing any model weights."""
from pathlib import Path
import sys, json, hashlib
import numpy as np
import torch

RUN = Path('D:/DroneSoccerData/runs/recurrent-lap-transfer-20260928')
sys.path.insert(0, str(RUN / 'source_snapshot'))
from sb3_contrib import RecurrentPPO

def digest(model):
    h = hashlib.sha256()
    for name, value in model.policy.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def gradient(model, arrays, params):
    total = torch.zeros(sum(p.numel() for p in params), dtype=torch.float64)
    loss_total = 0.
    count = len(arrays[0])
    for begin in range(0, count, 4):
        o, a, mask = [torch.as_tensor(x[begin:begin+4], device='cuda', dtype=torch.float32) for x in arrays]
        b, t, _ = o.shape
        model.policy.zero_grad(set_to_none=True)
        z = torch.zeros(1,b,model.policy.lstm_actor.hidden_size,device='cuda')
        d, _ = model.policy.get_distribution(o.reshape(-1,119), (z,z.clone()), torch.zeros(b*t,device='cuda'))
        loss = ((((d.distribution.mean.reshape(b,t,4)-a)**2).mean(-1)*mask).sum(1)/mask.sum(1)).mean()
        loss.backward()
        g = torch.cat([(torch.zeros_like(p) if p.grad is None else p.grad).detach().flatten() for p in params])
        total += g.cpu().double() * b / count
        loss_total += float(loss.detach()) * b / count
    return total, loss_total

def main():
    torch.set_num_threads(1)
    assert torch.cuda.is_available()
    seeds=json.loads((RUN/'protocol.json').read_text())['fit_lap_seeds']
    assert len(seeds)==9 and all(s%4!=3 for s in seeds)
    with np.load('D:/DroneSoccerData/runs/recurrent-learned-transfer-20260928/data_600000.npz') as d:
        basic=tuple(d[k].copy() for k in ('observations','actions','mask'))
    laps=[]
    for s in seeds:
        with np.load(f'D:/DroneSoccerData/diagnostics/learned-success-collection-v3/trajectories/{s}.npz') as d:
            laps.append((d['observations'][:-1].copy(),d['actions'].copy()))
    lap=(np.stack([d[0] for d in laps]),np.stack([d[1] for d in laps]),np.ones((9,600),np.float32))
    results=[]
    for checkpoint in (300,500):
        model=RecurrentPPO.load(RUN/f'model_update_{checkpoint}.zip',device='cuda')
        before=digest(model)
        model.policy.set_training_mode(True)  # CuDNN recurrent backward requires training mode.
        params=[p for n,p in model.policy.named_parameters() if n.startswith(('lstm_actor.','mlp_extractor.policy_net.','action_net.'))]
        gb,lb=gradient(model,basic,params)
        gl,ll=gradient(model,lap,params)
        mixed=.5*(gb+gl)
        dot=float(torch.dot(gb,gl))
        result=dict(checkpoint=checkpoint,basic_mse=lb,lap_mse=ll,
            basic_gradient_norm=float(gb.norm()),lap_gradient_norm=float(gl.norm()),
            cosine=dot/float(gb.norm()*gl.norm()),
            lap_gradient_dot_mixed=float(torch.dot(gl,mixed)),
            basic_gradient_dot_mixed=float(torch.dot(gb,mixed)),
            policy_sha256=before,weights_unchanged=digest(model)==before)
        assert result['weights_unchanged']
        results.append(result)
        print(json.dumps(result),flush=True)
    (Path(__file__).parent/'transfer_gradient_audit.json').write_text(json.dumps(dict(
        diagnostic_only=True,optimizer_steps=0,device=torch.cuda.get_device_name(),
        reserved_or_acceptance_used=False,results=results),indent=2))

if __name__=='__main__':
    main()
