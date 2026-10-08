import socket
import time

import numpy as np
import pytest

from src.command_rehearsal import LoopbackLink,RehearsalController,encode_action


@pytest.fixture
def rig():
    receiver=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
    receiver.bind(('127.0.0.1',0));receiver.settimeout(1)
    now=[100.]
    link=LoopbackLink('127.0.0.1',receiver.getsockname()[1])
    controller=RehearsalController(link,clock=lambda:now[0])
    yield controller,receiver,now
    controller.close();receiver.close()


def read(receiver): return receiver.recvfrom(1024)[0].decode()


@pytest.mark.parametrize('host',['192.168.10.1','0.0.0.0','8.8.8.8','::1','localhost'])
def test_no_real_network_destination(host):
    with pytest.raises(ValueError): LoopbackLink(host,8889)


def test_action_wire_order_and_quantized_history(rig):
    control,receiver,now=rig
    control.arm(now[0],now[0],now[0])
    issued=control.submit([.5,-.5,.25,-.25],now[0],now[0],now[0])
    assert read(receiver)=='rc 25 -25 25 -18'
    np.testing.assert_allclose(issued,[.5,-.5,.25,-18/70])
    np.testing.assert_array_equal(control.last_issued,issued)


@pytest.mark.parametrize('key',[' ','Escape','l','L','Ctrl+C'])
def test_stop_and_land_keys_latch_before_new_actions(rig,key):
    control,receiver,now=rig;control.arm(100,100,100)
    control.handle_key(key)
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'
    assert control.state=='stopped'
    assert not control.submit([1]*4,100,100,100).any()
    assert len(control.events)==2
    with pytest.raises(RuntimeError):control.arm(100,100,100)


@pytest.mark.parametrize('action',[[float('nan'),0,0,0],[2,0,0,0],[0,0,0]])
def test_invalid_prediction_stops(rig,action):
    control,receiver,now=rig;control.arm(100,100,100)
    control.submit(action,100,100,100)
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'


@pytest.mark.parametrize('stamps',[(99.,100.,100.),(100.,99.,100.),(100.,100.,99.),(101.,100.,100.)])
def test_stale_or_future_input_stops(rig,stamps):
    control,receiver,now=rig;control.arm(100,100,100)
    control.submit([.1]*4,*stamps)
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'


def test_predicting_again_cannot_refresh_an_old_frame(rig):
    control,receiver,now=rig;control.arm(100,100,100)
    now[0]=100.2;control.submit([.1]*4,100,100.2,100.2);read(receiver)
    now[0]=100.26;control.submit([.1]*4,100,100.26,100.26)
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'


def test_watchdog_runs_without_inference(rig):
    control,receiver,now=rig;control.arm(100,100,100)
    control.start_watchdog();now[0]=101
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'
    assert control.reason=='watchdog'


def test_send_failure_latches_and_still_attempts_land(rig,monkeypatch):
    control,receiver,now=rig;control.arm(100,100,100)
    attempts=[]
    def fail(command):
        attempts.append(command)
        raise OSError('simulated socket loss')
    monkeypatch.setattr(control.link,'send',fail)
    control.submit([.1]*4,100,100,100)
    assert control.state=='stopped' and not control.last_issued.any()
    assert attempts[-2:]==['rc 0 0 0 0','land']


def test_takeoff_is_not_supported(rig):
    with pytest.raises(ValueError):rig[0].link.send('takeoff')


def test_disarmed_and_expired_rehearsal_cannot_send_motion(rig):
    control,receiver,now=rig
    assert not control.submit([1]*4,100,100,100).any()
    assert control.events==[]
    control.arm(100,100,100);now[0]=131
    control.submit([1]*4,131,131,131)
    assert read(receiver)=='rc 0 0 0 0' and read(receiver)=='land'
    assert control.reason=='rehearsal time limit'
