"""Bounded startup flight tests with fake time/transport only."""
import unittest
import threading
import numpy as np
from flight_control import FlightControl
from startup_search import StartupSearch,LIMITS

class SearchPolicy:
    startup_search=True
    require_visible_start=True
    inputs=89
    def __init__(self):self.previous=[];self.predictions=0;self.hook=None
    def reset(self):self.previous=[];self.predictions=0
    def observe(self,camera,previous):
        self.previous.append(previous.copy())
        if self.hook:self.hook()
        return np.zeros(89,np.float32)
    def predict(self,camera,previous):
        self.predictions+=1
        return self.observe(camera,previous),np.array([.1,.2,.1,0],np.float32)

class SearchTests(unittest.TestCase):
    def setUp(self):
        self.now=100.;self.commands=[];self.rows=[];self.policy=SearchPolicy()
        self.c=FlightControl(self.policy,self.commands.append,self.rows.append,lambda:self.now)
        self.fresh(0);self.c.start();self.now+=.1;self.fresh(80)
        self.c.acknowledge('ok',self.now);self.c.watch()
        self.seq=0

    def fresh(self,h,yaw=0):
        self.c.capture_ready=True;self.c.receive_camera(self.now)
        self.c.receive_telemetry(dict(h=h,yaw=yaw,bat=80,temph=30),self.now);self.c.heartbeat()

    def step(self,visible=False,h=80,dt=.1,yaw=0):
        self.now+=dt;self.fresh(h,yaw);self.seq+=1
        camera=np.zeros(8,np.float32);camera[6]=visible
        return self.c.step(self.seq,self.now,camera)

    def test_slow_rise_then_bounded_turns_no_translation(self):
        self.assertEqual(self.c.state,'acquiring')
        self.step(dt=.6);self.assertEqual(self.commands[-1],'rc 0 0 15 0')
        self.step(h=120);self.assertEqual(self.commands[-1],'rc 0 0 0 15')
        for _ in range(31):self.step(h=120)
        self.assertEqual(self.commands[-1],'rc 0 0 0 -15')
        for _ in range(31):self.step(h=120)
        self.assertEqual(self.c.state,'landing');self.assertEqual(self.commands[-1],'land')
        self.assertEqual(self.policy.predictions,0)
        self.assertTrue(all(row['controller']=='scripted_startup_search' for row in self.rows))
        self.assertEqual(self.c.snapshot()['policy_steps'],0)

    def test_stable_detection_neutral_before_policy_and_actual_history(self):
        self.step(dt=.6);self.step()
        np.testing.assert_allclose(self.policy.previous[-1],[0,0,.15,0])
        for _ in range(7):
            row=self.step(visible=True)
            if self.c.state=='active':break
        self.assertEqual(self.c.state,'active');self.assertEqual(self.commands[-1],'rc 0 0 0 0')
        self.assertEqual(self.policy.predictions,0)
        row=self.step(visible=True)
        self.assertEqual(row['event'],'send_result');self.assertEqual(self.policy.predictions,1)
        self.assertEqual(self.commands[-1],'rc 5 10 10 0')
        self.assertEqual(self.c.snapshot()['policy_steps'],1)

    def test_visibility_flicker_cannot_start_policy(self):
        self.step(dt=.6)
        for _ in range(8):self.step(visible=True);self.step(visible=False)
        self.assertEqual(self.c.state,'acquiring');self.assertEqual(self.policy.predictions,0)

    def test_timeout_while_detection_flickers_still_lands(self):
        self.step(h=120,dt=.6)
        for i in range(150):
            if self.c.state!='acquiring':break
            self.step(visible=i%2==0,h=120)
        self.assertEqual(self.c.state,'landing');self.assertEqual(self.policy.predictions,0)

    def test_no_unsafe_ascent_with_stuck_height(self):
        for _ in range(60):
            if self.c.state!='acquiring':break
            self.step()
        self.assertIn('ascent timeout',self.c.reason);self.assertEqual(self.commands[-1],'land')

    def test_existing_watchdogs_apply_during_search(self):
        for field in ('frame_at','telemetry_at','operator_at','search_updated'):
            with self.subTest(field=field):
                self.setUp();setattr(self.c,field,self.now-1);self.c.watch()
                self.assertEqual(self.c.state,'landing');self.assertEqual(self.commands[-1],'land')

    def test_search_hard_timeout_independent_of_frame_loop(self):
        self.now+=14;self.fresh(100);self.c.search_updated=self.now;self.c.watch()
        self.assertEqual(self.c.state,'landing');self.assertIn('timeout',self.c.reason)

    def test_missing_yaw_prevents_takeoff(self):
        c=FlightControl(SearchPolicy(),self.commands.append,self.rows.append,lambda:self.now)
        c.capture_ready=True;c.receive_camera(self.now);c.heartbeat()
        c.receive_telemetry(dict(bat=80,h=0,temph=30),self.now)
        before=list(self.commands)
        with self.assertRaisesRegex(ValueError,'yaw'):c.start()
        self.assertEqual(self.commands,before)

    def test_yaw_bounds_wrap_and_bad_sensor(self):
        s=StartupSearch(0,179);s.step(.1,120,-179,False)
        self.assertEqual(s.yaw_excursion,2)
        for yaw in (None,float('nan'),181):
            with self.assertRaises(ValueError):StartupSearch(0,yaw)
        s=StartupSearch(0,0);s.step(.6,120,30,False);s.step(.7,120,60,False)
        with self.assertRaisesRegex(ValueError,'bound'):s.step(.8,120,61,False)

    def test_search_stop_limits_and_send_log_failures(self):
        for kind in ('ceiling','floor','negative','battery','temperature','log','send'):
            with self.subTest(kind=kind):
                self.setUp();self.now+=.6;self.fresh(80)
                if kind=='ceiling':self.c.telemetry['h']=151
                if kind=='floor':self.c.telemetry['h']=20
                if kind=='negative':self.c.telemetry['h']=-20
                if kind=='battery':self.c.telemetry['bat']=29
                if kind=='temperature':self.c.telemetry['temph']=80
                if kind=='log':
                    def fail(_):raise OSError('disk full')
                    self.c.log=fail
                if kind=='send':
                    def sink(command):
                        self.commands.append(command)
                        if command=='rc 0 0 15 0':raise OSError('UDP failed')
                    self.c.send=sink
                self.c.step(1,self.now,np.zeros(8,np.float32))
                self.assertEqual(self.c.state,'landing');self.assertEqual(self.commands[-1],'land')

    def test_operator_landing_interrupts_history_inference(self):
        entered=threading.Event();release=threading.Event()
        self.policy.hook=lambda:(entered.set(),release.wait(2))
        thread=threading.Thread(target=lambda:self.step(dt=.6));thread.start()
        self.assertTrue(entered.wait(1));self.c.stop('operator land');release.set();thread.join(1)
        self.assertFalse(thread.is_alive());self.assertEqual(self.commands,['takeoff','rc 0 0 0 0','land'])

    def test_lost_target_at_first_policy_frame_aborts(self):
        for _ in range(8):
            self.step(visible=True)
            if self.c.state=='active':break
        self.step(visible=False)
        self.assertEqual(self.c.state,'landing');self.assertEqual(self.policy.predictions,0)

    def test_duplicate_frame_cannot_confirm_or_advance_history(self):
        self.step(visible=True,dt=.6);count=len(self.policy.previous)
        self.c.step(self.seq,self.now,np.array([0,0,0,0,0,0,1,0],np.float32))
        self.assertEqual(len(self.policy.previous),count);self.assertEqual(self.c.state,'acquiring')

    def test_policy_time_freezes_at_last_learned_step(self):
        for _ in range(8):
            self.step(visible=True)
            if self.c.state=='active':break
        self.step(visible=True);self.step(visible=True)
        before=self.c.snapshot()['policy_seconds'];self.c.stop('operator land');self.now+=20
        self.assertEqual(self.c.snapshot()['policy_seconds'],before)

if __name__=='__main__':unittest.main()
