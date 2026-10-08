"""Differentiable reset-to-trajectory motion model for simulation training research.

Truth parameters are loss/diagnostic inputs ONLY. This module is not an actor,
observation adapter, flight controller, passage scorer or replacement simulator.
It replays learned commands; it never chooses commands. Camera visibility,
collision termination and pass events remain with the original simulator.
"""
import math
import torch


def rollout(actions, dt, lengths, initial_position, initial_heading,
            half_box, cage, speed, yaw_degrees, tau, delay, check=lambda: None):
    """Batched whole-episode motion with real reset (zero velocity/pending queue).

    Float64 mirrors NumPy physics. actions[B,T,4] are normalized issued commands;
    dt[B,T] is the actual simulation tick duration. cage[B,4] is length,width,
    height,min_altitude. Delays are integer ticks; lengths freeze genuine ends.
    Piecewise derivatives at contact/clipping/wrap switches are not smooth
    derivatives of full closed-loop flight outcomes.
    """
    if actions.ndim != 3 or actions.shape[2] != 4 or not 1 <= actions.shape[1] <= 600:
        raise ValueError('Expected B by T<=600 by four learned commands')
    b, t, _ = actions.shape
    tensors = (actions, dt, initial_position, initial_heading, half_box, cage, speed, yaw_degrees, tau)
    if any(x.dtype != torch.float64 or x.device != actions.device or not torch.isfinite(x).all()
           for x in tensors):
        raise ValueError('Finite float64 tensors on one device required')
    shapes = ((b,t),(b,3),(b,),(b,3),(b,4),(b,),(b,),(b,))
    if any(x.shape != shape for x,shape in zip(tensors[1:],shapes)):
        raise ValueError('Mismatched state/parameter shape')
    if (lengths.shape != (b,) or delay.shape != (b,) or lengths.dtype != torch.int64
        or delay.dtype != torch.int64 or lengths.device != actions.device or delay.device != actions.device
        or torch.any((lengths < 1) | (lengths > t)) or torch.any((delay < 1) | (delay > 4))):
        raise ValueError('Valid episode lengths and existing one-to-four tick delays required')
    if (torch.any(abs(actions)>1) or torch.any(dt<=0) or torch.any(tau<=0)
        or torch.any(half_box<=0) or torch.any(speed<=0) or torch.any(yaw_degrees<=0)
        or torch.any(cage[:,:3]<=0) or torch.any(cage[:,3]>=cage[:,2]-half_box[:,2])):
        raise ValueError('Invalid normalized commands or physical parameters')
    p, h = initial_position, initial_heading
    v, w = torch.zeros_like(p), torch.zeros_like(h)
    batch = torch.arange(b, device=actions.device)
    positions=[]; headings=[]; velocities=[]; yaw_rates=[]; walls=[]
    for step in range(t):
        check()
        active = step < lengths
        index = step-delay
        command = actions[batch, index.clamp(min=0)]
        command = torch.where((index>=0)[:,None],command,torch.zeros_like(command))
        right, forward, up, clockwise = command.unbind(-1)
        c,s = torch.cos(h),torch.sin(h)
        target = speed[:,None]*torch.stack((forward*c+right*s,forward*s-right*c,up),dim=-1)
        yaw_target = -clockwise*yaw_degrees*(math.pi/180)
        delta = dt[:,step]; alpha=1-torch.exp(-delta/tau)
        nv = v+(target-v)*alpha[:,None]
        nw = w+(yaw_target-w)*alpha
        raw = p+nv*delta[:,None]
        nh = torch.remainder(h+nw*delta+math.pi,2*math.pi)-math.pi
        c,s = abs(torch.cos(nh)),abs(torch.sin(nh))
        reachx = c*half_box[:,0]+s*half_box[:,1]
        reachy = s*half_box[:,0]+c*half_box[:,1]
        low = torch.stack((reachx,reachy,cage[:,3]),-1)
        high = torch.stack((cage[:,0]-reachx,cage[:,1]-reachy,cage[:,2]-half_box[:,2]),-1)
        clipped = torch.minimum(torch.maximum(raw,low),high)
        touched = clipped != raw
        nv = torch.where(touched,torch.zeros_like(nv),nv)
        p=torch.where(active[:,None],clipped,p);h=torch.where(active,nh,h)
        v=torch.where(active[:,None],nv,v);w=torch.where(active,nw,w)
        positions.append(p);headings.append(h);velocities.append(v);yaw_rates.append(w)
        walls.append(active & touched.any(-1))
    return dict(positions=torch.stack(positions,1),headings=torch.stack(headings,1),
                velocities=torch.stack(velocities,1),yaw_rates=torch.stack(yaw_rates,1),
                wall_contacts=torch.stack(walls,1))
