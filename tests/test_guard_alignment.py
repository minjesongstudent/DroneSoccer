import numpy as np
from src.sim import HoopEnv
from tools.audit_guard_alignment import projected_contacts


def test_projection_agrees_with_native_noodle_collision_geometry():
    env=HoopEnv(level=1,randomise=True,hanger=True);env.reset(seed=11001234)
    rng=np.random.default_rng(82)
    poses=env.hoop_centre+rng.uniform(-.22,.22,(100,3))
    headings=rng.uniform(-np.pi,np.pi,100)
    actual=projected_contacts(env,poses,headings)
    expected=[]
    for p,h in zip(poses,headings):
        env.pos=p-((p-env.hoop_centre)@env.hoop_normal)*env.hoop_normal
        env.heading=h
        expected.append(env._hit())
    np.testing.assert_array_equal(actual,expected)
    env.close()


def test_translation_along_normal_does_not_change_projection():
    env=HoopEnv(level=1,randomise=False,hanger=True);env.reset(seed=11000000)
    poses=env.hoop_centre+np.array([[0,0,0],[.15,.15,0]])
    headings=np.array([0.,1.])
    np.testing.assert_array_equal(projected_contacts(env,poses,headings),
                                  projected_contacts(env,poses+.4*env.hoop_normal,headings))
    env.close()
