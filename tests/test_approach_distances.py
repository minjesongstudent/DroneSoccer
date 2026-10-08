import numpy as np
from tools.audit_approach_distances import samples


def fixture():
    trace=dict(positions=np.array([[-.4,0,0],[-.29,.1,0],[-.2,.2,0],[-.1,.3,0],[.1,.4,0]]),
               headings=np.zeros(5),hoop_normal=np.array([1,0,0]),hoop_centre=np.zeros(3),
               actor_observations=np.zeros((5,85)),actions=np.zeros((5,4)))
    trace['actions'][2]=[.2,.7,0,.4]
    row=dict(seed=1,wall_steps=0,clean_first_pass=True,full_body_steps=[5],steps=5,
             outcome='timeout',full_body_passes=1)
    return row,trace


def test_features_use_pre_action_position_and_current_command():
    row,trace=fixture()
    value=samples(row,trace)[0]
    assert value['index']==2 and value['lateral']==.1
    assert value['forward']==.7 and value['yaw']==.4
    assert value['command_lateral']==-.2


def test_contact_episodes_and_absent_thresholds_excluded():
    row,trace=fixture();row['wall_steps']=1
    assert samples(row,trace)==[]
    row['wall_steps']=0;trace['positions'][:,0]=-.5
    assert samples(row,trace)==[]
