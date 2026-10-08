"""Differentiable simulation-loss geometry, never an actor observation/control.

The original simulator alone scores collisions/passages. Signed margins below
are a training surrogate: positive means separated; zero is a boundary. Ring
distance follows the simulator's sampled noodle centreline. Hanger distance is
a separating-axis margin, not Euclidean distance. Cage uses its exact box limits.
"""
import torch


def geometry_from_env(env, *, device='cpu'):
    """Copy truth ONLY for a loss or numerical diagnostic. No actor may read it."""
    f=lambda x:torch.as_tensor(x,dtype=torch.float64,device=device).clone()
    return dict(half_box=f(env._half_box),centreline=f(env._centreline),
                thickness=f(env.hoop.thickness),centre=f(env.hoop_centre),
                normal=f(env.hoop_normal),side=f(env._axis_h),
                hanger_bottom=f(env._hanger_bottom),hanger_half=f(env.hanger_half_width),
                cage=f([env.cage.length,env.cage.width,env.cage.height,env.cage.min_altitude]),
                hanger=bool(env.hanger))


def margins(position, heading, *, half_box, centreline, thickness, centre,
            normal, side, hanger_bottom, hanger_half, cage, hanger):
    """One scene, N finite poses; outputs [N,3] for ring, hanger and cage."""
    if position.ndim!=2 or position.shape[1]!=3 or heading.shape!=(len(position),):
        raise ValueError('Expected N positions and headings')
    values=(position,heading,half_box,centreline,thickness,centre,normal,side,hanger_bottom,hanger_half,cage)
    if any(x.dtype!=torch.float64 or x.device!=position.device or not torch.isfinite(x).all() for x in values):
        raise ValueError('Finite float64 geometry on one device required')
    if (half_box.shape!=(3,) or centreline.ndim!=2 or centreline.shape[1]!=3 or len(centreline)<4
        or any(x.shape!=(3,) for x in (centre,normal,side)) or cage.shape!=(4,)
        or any(x.ndim!=0 for x in (thickness,hanger_bottom,hanger_half)) or type(hanger) is not bool):
        raise ValueError('Malformed scene geometry')
    if (torch.any(half_box<=0) or thickness<=0 or hanger_half<0
        or not torch.allclose(torch.linalg.vector_norm(normal),normal.new_tensor(1.),atol=1e-12,rtol=0)
        or not torch.allclose(torch.linalg.vector_norm(side),side.new_tensor(1.),atol=1e-12,rtol=0)
        or abs(normal@side)>1e-12 or abs(normal[2])>1e-12 or abs(side[2])>1e-12):
        raise ValueError('Positive box and upright orthonormal hoop geometry required')
    c,s=torch.cos(heading),torch.sin(heading)
    delta=centreline[None]-position[:,None]
    body=torch.stack((c[:,None]*delta[:,:,0]+s[:,None]*delta[:,:,1],
                      -s[:,None]*delta[:,:,0]+c[:,None]*delta[:,:,1],delta[:,:,2]),-1)
    distance=torch.linalg.vector_norm(torch.clamp(abs(body)-half_box,min=0),dim=-1)
    ring=distance.amin(1)-thickness/2
    forward=torch.stack((c,s),-1);left=torch.stack((-s,c),-1)
    if hanger:
        axes=torch.stack((forward,left,normal[:2].expand_as(forward),side[:2].expand_as(forward)),1)
        gap=position[:,:2]-centre[:2]
        reach=(half_box[0]*abs((axes*forward[:,None]).sum(-1))
               +half_box[1]*abs((axes*left[:,None]).sum(-1))
               +thickness/2*abs((axes*normal[:2]).sum(-1))
               +hanger_half*abs((axes*side[:2]).sum(-1)))
        horizontal=(abs((axes*gap[:,None]).sum(-1))-reach).amax(1)
        vertical=hanger_bottom-position[:,2]-half_box[2]
        hanging=torch.maximum(horizontal,vertical)
    else:
        hanging=torch.full_like(ring,torch.inf)
    rx=abs(c)*half_box[0]+abs(s)*half_box[1]
    ry=abs(s)*half_box[0]+abs(c)*half_box[1]
    low=torch.stack((rx,ry,cage[3].expand_as(rx)),-1)
    high=torch.stack((cage[0]-rx,cage[1]-ry,(cage[2]-half_box[2]).expand_as(rx)),-1)
    boundary=torch.minimum(position-low,high-position).amin(-1)
    return torch.stack((ring,hanging,boundary),-1)


def clearance_loss(values, *, buffer_m=.02):
    """Mean squared deficit from a DESIGN 2cm margin, measured in metres squared.

    Does not change physical geometry or task scoring. Operate on genuine rows
    only, never padded rows. Zero/ties/contact switches are nonsmooth. Min-distance
    gradients can vanish after deep penetration or an already-clamped wall hit.
    Actual learned-policy outcome validation is necessary before any promotion.
    """
    if (values.ndim!=2 or values.shape[1]!=3 or not len(values)
        or buffer_m<=0 or not torch.isfinite(torch.amin(values,dim=1)).all()):
        raise ValueError('Genuine finite nearest-clearance rows and positive design buffer required')
    return torch.square(torch.relu(buffer_m-values.amin(1))).mean()
