"""Time-indexed baselines from other independent flights of the same start."""
import numpy as np

def leave_one_out_baselines(returns, groups):
    if len(returns)!=len(groups):raise ValueError('Mismatched groups')
    values=[np.asarray(x,dtype=np.float64) for x in returns]
    if any(x.ndim!=1 or not len(x) or not np.isfinite(x).all() for x in values):
        raise ValueError('Finite nonempty return sequences required')
    result=[]
    for i,target in enumerate(values):
        peers=[x for j,x in enumerate(values) if j!=i and groups[j]==groups[i]]
        if not peers:raise ValueError('Independent peers required')
        baseline=np.zeros(len(target),dtype=np.float64)
        for x in peers:
            n=min(len(x),len(target));baseline[:n]+=x[:n]
        result.append(baseline/len(peers))
    return result
