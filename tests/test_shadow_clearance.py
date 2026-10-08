import numpy as np
import pytest
import torch
from src.sim import HoopEnv
from src.shadow_clearance import geometry_from_env,margins,clearance_loss


def scene():
    e=HoopEnv(randomise=False,hanger=True);e.reset(seed=10)
    e.place([.5,1,1],0,[1,1,1],[1,0,0],hoop_diameter=.45,hanger_half_width=.08)
    return e


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_pose_geometry_matches_source(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    e=scene();g=geometry_from_env(e,device=device)
    rng=np.random.default_rng(42);p=rng.uniform([.01,.01,.49],[2.04,2.04,2.04],(200,3));h=rng.uniform(-3,3,200)
    v=margins(torch.tensor(p,device=device),torch.tensor(h,device=device),**g).cpu().numpy()
    for i in range(200):
        e.pos=p[i].copy();e.heading=h[i];e.vel=np.zeros(3)
        assert abs(v[i,0]-(e._box_distance(e._centreline).min()-e.hoop.thickness/2))<1e-12
        assert (v[i,0]<0)==e._hit()
        assert (v[i,1]<=0)==e._hits_hanger()
        wall,_=e._touch_walls();assert (v[i,2]<0)==wall
    e.close()


@pytest.mark.parametrize('device',['cpu','cuda'])
def test_gradient_smooth_near_ring(device):
    if device=='cuda' and not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    e=scene();g=geometry_from_env(e,device=device)
    p=torch.tensor([[.96,1.16,1.03]],dtype=torch.float64,device=device,requires_grad=True)
    h=torch.tensor([.27],dtype=torch.float64,device=device,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x,y:margins(x,y,**g),(p,h),eps=1e-6,atol=1e-7,rtol=1e-4)
    e.close()


def test_hanger_touch_inclusive_and_disabled():
    e=scene();g=geometry_from_env(e)
    p=torch.tensor([[1.,1.,e._hanger_bottom-e.drone.height/2]],dtype=torch.float64)
    h=torch.zeros(1,dtype=torch.float64)
    e.pos=p[0].numpy();e.heading=0
    m=margins(p,h,**g)
    # Floating subtraction can leave a few ulps at the vertical contact boundary.
    assert abs(m[0,1])<1e-14 and e._hits_hanger()
    g['hanger']=False
    assert torch.isposinf(margins(p,h,**g)[0,1])


def test_loss_nearest_object_units_and_backprop():
    v=torch.tensor([[.01,.2,.3],[.2,float('inf'),.1]],dtype=torch.float64,requires_grad=True)
    loss=clearance_loss(v);assert float(loss.detach())==pytest.approx(.00005)
    loss.backward();assert v.grad[0,0]<0 and torch.count_nonzero(v.grad)==1


def test_deep_ring_contact_has_finite_loss_gradient():
    e=scene();p=torch.tensor(e._centreline[:1],requires_grad=True);h=torch.zeros(1,dtype=torch.float64)
    loss=clearance_loss(margins(p,h,**geometry_from_env(e)));loss.backward()
    assert torch.isfinite(p.grad).all()


@pytest.mark.parametrize('which',['nonfinite','shape','axis'])
def test_malformed_geometry_rejected(which):
    e=scene();g=geometry_from_env(e);p=torch.tensor([[.5,1.,1.]],dtype=torch.float64);h=torch.zeros(1,dtype=torch.float64)
    if which=='nonfinite':p[0,0]=float('nan')
    elif which=='shape':p=p[:,:2]
    else:g['normal']*=2
    with pytest.raises(ValueError):margins(p,h,**g)
