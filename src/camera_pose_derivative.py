"""Loss-only numerical camera Jacobian research; never a deployable observation.

Queries the unchanged source detector at saved poses and nearby numeric points.
No env.step, policy command or replacement rendering is supplied here. Discrete
visibility/marker/clip/arc changes are reported instead of silently smoothed.
"""
import numpy as np
from src import contract


def query(env, pose):
    pose=np.asarray(pose,np.float64)
    if pose.shape!=(4,) or not np.isfinite(pose).all():
        raise ValueError('Finite xyz/heading pose required')
    old_position,old_heading=env.pos,env.heading
    try:
        env.pos=pose[:3].copy();env.heading=float(pose[3])
        obs=env.observe()
        if obs.shape!=(8,) or not np.isfinite(obs).all():raise ValueError('Invalid source camera result')
        cam,axes=env._camera_pose();_,_,inside=env._project(env._sight_points,cam,axes)
        # Invisible interiors have identically zero outputs regardless of which
        # insufficient arc points project into the image.
        arc=inside[:-1].tobytes() if obs[6] else b''
        clip=(abs(obs[:3])>=contract.CLIP).tobytes()
        signature=(bool(obs[6]),bool(obs[7]),arc,clip)
        return obs.copy(),signature
    finally:env.pos,env.heading=old_position,old_heading


def central(env,pose,epsilon,reference_signature,check=lambda:None):
    if not np.isfinite(epsilon) or epsilon<=0:raise ValueError('Positive finite epsilon required')
    jac=np.zeros((6,4));stable=True
    for axis in range(4):
        check();delta=np.zeros(4);delta[axis]=epsilon
        a,sa=query(env,pose+delta);b,sb=query(env,pose-delta)
        stable &= sa==reference_signature and sb==reference_signature
        jac[:,axis]=(a[:6].astype(np.float64)-b[:6].astype(np.float64))/(2*epsilon)
    return jac,stable


def consistency(env,pose,direction,check=lambda:None):
    """Fixed 1mm/1mrad stencil against half-size and an independent direction."""
    pose=np.asarray(pose,np.float64);direction=np.asarray(direction,np.float64)
    if direction.shape!=(4,) or not np.isfinite(direction).all() or not np.isclose(np.linalg.norm(direction),1):
        raise ValueError('Unit finite xyz/heading test direction required')
    base,signature=query(env,pose)
    coarse,sc=central(env,pose,.001,signature,check)
    fine,sf=central(env,pose,.0005,signature,check)
    a,sa=query(env,pose+.0005*direction);b,sb=query(env,pose-.0005*direction)
    directional=(a[:6].astype(np.float64)-b[:6].astype(np.float64))/.001
    predicted=coarse@direction
    je=float(np.linalg.norm(coarse-fine));jr=je/max(float(np.linalg.norm(fine)),1e-12)
    de=float(np.linalg.norm(predicted-directional));dr=de/max(float(np.linalg.norm(directional)),1e-12)
    return dict(observation=base,jacobian=coarse,half_jacobian=fine,directional=directional,
        stable=bool(sc and sf and sa==signature and sb==signature),visible=bool(base[6]),
        jacobian_absolute=je,jacobian_relative=jr,directional_absolute=de,directional_relative=dr,
        agrees=bool((je<=.002 or jr<=.05) and (de<=.002 or dr<=.05)))
