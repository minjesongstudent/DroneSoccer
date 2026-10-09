"""Fault injection with fake clock/transport/policy: NO physical connection."""
import threading
import unittest
import numpy as np
from flight_control import FlightControl


class Clock:
    def __init__(self):self.t=100.
    def __call__(self):return self.t


class Policy:
    def __init__(self):self.previous=[];self.hook=None;self.action=np.array([.21,-.22,.13,-.14])
    def reset(self):pass
    def predict(self,features,previous):
        self.previous.append(previous.copy())
        if self.hook:self.hook()
        return np.zeros(85,np.float32),self.action


class FlightTests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();self.commands=[];self.records=[];self.policy=Policy()
        self.c=FlightControl(self.policy,self.commands.append,self.records.append,self.clock)

    def fresh(self,h=0):
        self.c.capture_ready=True;self.c.receive_camera(self.clock.t)
        self.c.receive_telemetry(dict(bat=80,h=h,temph=30),self.clock.t)
        self.c.heartbeat()

    def activate(self):
        self.fresh();self.c.start()
        self.clock.t+=.1;self.fresh(80);self.c.acknowledge('ok',self.clock.t);self.c.watch()
        self.assertEqual(self.c.state,'active')

    def test_no_takeoff_without_recording_and_fresh_inputs(self):
        with self.assertRaises(ValueError):self.c.start()
        self.assertEqual(self.commands,[])
        self.fresh();self.c.operator_at=None
        with self.assertRaises(ValueError):self.c.start()
        self.assertEqual(self.commands,[])

    def test_takeoff_ack_and_measured_height_both_required(self):
        self.fresh();self.c.start();self.assertEqual(self.commands,['takeoff'])
        self.clock.t+=.1;self.fresh(80);self.c.watch()
        self.assertEqual(self.c.state,'taking_off')
        self.c.acknowledge('ok',self.clock.t);self.c.watch();self.assertEqual(self.c.state,'active')

    def test_action_mapping_and_actual_quantized_history(self):
        self.activate();a=self.c.step(1,self.clock.t,np.zeros(8))
        self.assertEqual(self.commands[-1],'rc 10 -11 13 -10')
        self.clock.t+=.1;self.fresh(80);self.c.step(2,self.clock.t,np.zeros(8))
        np.testing.assert_allclose(self.policy.previous[1],np.array([10/50,-11/50,13/100,-10/70]))
        self.assertTrue(a['sent']);self.assertEqual(len(self.records),4)

    def test_release_latches_and_late_prediction_cannot_send(self):
        self.activate();self.policy.hook=lambda:self.c.stop('released during inference')
        self.assertIsNone(self.c.step(1,self.clock.t,np.zeros(8)))
        self.assertEqual(self.commands,['takeoff','rc 0 0 0 0','land'])
        with self.assertRaises(ValueError):self.c.start()

    def test_camera_telemetry_operator_and_inference_watchdogs(self):
        for field in ('frame_at','telemetry_at','operator_at','last_prediction'):
            with self.subTest(field=field):
                self.setUp();self.activate();setattr(self.c,field,self.clock.t-1)
                self.c.watch();self.assertEqual(self.c.state,'landing')
                self.assertEqual(self.commands[-2:],['rc 0 0 0 0','land'])

    def test_stop_during_takeoff_retries_land_and_has_no_restart(self):
        self.fresh();self.c.start();self.c.stop('release')
        self.clock.t+=1.1;self.c.watch();self.assertEqual(self.commands.count('land'),2)
        self.c.acknowledge('ok',self.clock.t);self.assertEqual(self.c.state,'landing')
        self.clock.t+=20;self.c.watch();self.assertEqual(self.c.state,'stopped')
        self.assertFalse(self.c.snapshot()['physical_landing_confirmed'])

    def test_timeout_and_height_limits(self):
        self.activate();self.clock.t+=10.1;self.fresh(80);self.c.last_prediction=self.clock.t
        self.c.watch();self.assertIn('10-second',self.c.reason)
        self.setUp();self.activate();self.clock.t+=.1;self.fresh(160);self.c.watch()
        self.assertIn('ceiling',self.c.reason)

    def test_invalid_action_and_log_failure_land(self):
        self.activate();self.policy.action[0]=float('nan')
        self.c.step(1,self.clock.t,np.zeros(8));self.assertEqual(self.c.state,'landing')
        self.setUp();self.activate()
        def fail(row):raise OSError('disk full')
        self.c.log=fail;self.c.step(1,self.clock.t,np.zeros(8))
        self.assertEqual(self.commands,['takeoff','rc 0 0 0 0','land'])

    def test_rc_send_failure_requests_landing(self):
        self.activate()
        def send(command):
            self.commands.append(command)
            if command.startswith('rc ') and command!='rc 0 0 0 0':raise OSError('UDP failed')
        self.c.send=send;record=self.c.step(1,self.clock.t,np.zeros(8))
        self.assertFalse(record['sent']);self.assertEqual(record['issued_action'],[])
        self.assertEqual(self.commands[-1],'land')

    def test_emergency_is_explicit_and_distinct_from_land(self):
        self.activate();self.c.emergency();self.assertEqual(self.commands[-1],'emergency')
        self.assertEqual(self.c.state,'stopped')

    def test_implausible_or_missing_telemetry_prevents_takeoff(self):
        for fields in (dict(bat=200,h=0,temph=30),dict(bat=80,h=-5,temph=30),dict(bat=80,h=0)):
            with self.subTest(fields=fields):
                self.setUp();self.fresh();self.clock.t+=.01
                self.c.receive_telemetry(fields,self.clock.t)
                with self.assertRaises(ValueError):self.c.start()
                self.assertEqual(self.commands,[])

    def test_blocking_inference_can_be_interrupted_from_other_thread(self):
        self.activate();entered=threading.Event();release=threading.Event()
        def block():entered.set();release.wait(2)
        self.policy.hook=block
        thread=threading.Thread(target=lambda:self.c.step(1,self.clock.t,np.zeros(8)))
        thread.start();self.assertTrue(entered.wait(1));self.c.stop('operator')
        self.assertEqual(self.commands[-1],'land');release.set();thread.join(1)
        self.assertFalse(thread.is_alive());self.assertEqual(self.commands.count('land'),1)


if __name__=='__main__':unittest.main()
