"""Training-only bounded approach penalty. Never changes actor inputs or commands."""
import numpy as np

MAX_PENALTY = 0.2


def alignment_penalties(env, positions, headings):
    positions=np.asarray(positions);headings=np.asarray(headings)
    along=(positions-env.hoop_centre)@env.hoop_normal
    # Continuous band, zero outside [-0.4, 0] m; maximum at -0.1 m.
    gate=np.clip((along+.4)/.3,0,1)*np.clip(-along/.1,0,1)
    result=np.zeros(len(positions))
    ids=np.flatnonzero(gate>0)
    if not len(ids):return result
    projected=positions[ids]-along[ids,None]*env.hoop_normal
    delta=env._centreline[None,:,:]-projected[:,None,:]
    c=np.cos(headings[ids])[:,None];s=np.sin(headings[ids])[:,None]
    body=np.stack([c*delta[:,:,0]+s*delta[:,:,1],-s*delta[:,:,0]+c*delta[:,:,1],delta[:,:,2]],axis=2)
    distance=np.linalg.norm(np.maximum(np.abs(body)-env._half_box,0),axis=2).min(1)
    deficit=np.clip(1-distance/(env.hoop.thickness/2),0,1)
    result[ids]=-MAX_PENALTY*gate[ids]*deficit
    return result
