"""Bounded startup flight tests with fake time/transport only."""
import unittest
import json
from pathlib import Path
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

    def test_staged_measured_search_covers_altitudes_and_both_headings(self):
        # A deliberately quantized fake plant, not a claimed physical result.
        height=40.;yaw=0.;reached=set();turns=set()
        for _ in range(1200):
            if self.c.state!='acquiring':break
            row=self.step(h=10*int(height/10),yaw=yaw)
            if row:
                rc=list(map(int,row['command'].split()[1:]))
                self.assertEqual(rc[:2],[0,0]);self.assertFalse(rc[2] and rc[3])
                if row['phase'].startswith('scanning'):
                    reached.add(self.c.search.stage);turns.add((self.c.search.stage,rc[3]))
                if rc[2]>0:self.assertLess(row['height_cm'],180)
                height+=rc[2]/15*.6;yaw+=rc[3]/15*1.2
        self.assertEqual(reached,{0,1,2})
        for stage in range(3):
            self.assertIn((stage,15),turns);self.assertIn((stage,-15),turns)
        self.assertEqual(self.c.state,'landing')
        self.assertIn('search complete',self.c.reason)
        self.assertLessEqual(self.now-self.c.search.started,LIMITS['timeout_s'])
        self.assertEqual(self.policy.predictions,0)
        self.assertTrue(all(row['controller']=='scripted_startup_search' for row in self.rows))

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
        for i in range(1250):
            if self.c.state!='acquiring':break
            self.step(visible=i%2==0,h=120)
        self.assertEqual(self.c.state,'landing');self.assertEqual(self.policy.predictions,0)

    def test_no_unsafe_ascent_with_stuck_height(self):
        for _ in range(60):
            if self.c.state!='acquiring':break
            self.step()
        self.assertIn('climb stalled',self.c.reason);self.assertEqual(self.commands[-1],'land')

    def test_recorded_slow_climb_rate_reaches_scan_without_old_timeout(self):
        for i in range(155):
            h=min(120,40+10*int(max(0,i*.1-.5)*.6))
            self.step(h=h)
            if self.c.search.scan_at is not None:break
        self.assertEqual(self.c.state,'acquiring')
        self.assertIsNotNone(self.c.search.scan_at)
        self.assertGreater(self.now-self.c.search.started,5.5)
        self.assertEqual(self.commands[-1],'rc 0 0 0 0')
        self.assertEqual(self.policy.predictions,0)

    def test_progress_does_not_remove_absolute_ascent_deadline(self):
        # Exhausted climb budget cannot be reset by completing a scan stage.
        self.c.search.ascent_seconds=LIMITS['ascent_timeout_s']
        self.step(dt=.6)
        self.assertEqual(self.c.state,'landing')
        self.assertIn('ascent timeout',self.c.reason)

    def test_target_visibility_stops_long_climb_before_handoff(self):
        for i in range(90):self.step(h=40+10*int(i*.1*.6))
        self.assertEqual(self.c.state,'acquiring')
        self.step(visible=True,h=100)
        self.assertEqual(self.commands[-1],'rc 0 0 0 0')
        for _ in range(6):
            if self.c.state=='active':break
            self.step(visible=True,h=100)
        self.assertEqual(self.c.state,'active')
        self.assertEqual(self.policy.predictions,0)

    def test_uploaded_climb_traces_do_not_abort_at_old_five_second_limit(self):
        traces=json.loads((Path(__file__).parent/'testdata/recorded-climb-traces.json').read_text())
        self.assertEqual(len(traces),4)
        for trace in traces:
            with self.subTest(session=trace['session']):
                frames=trace['frames'];search=StartupSearch(0,frames[0]['yaw'])
                for frame in frames:
                    rc,ready=search.step(frame['t'],frame['h'],frame['yaw'],frame['visible'])
                    self.assertFalse(ready)
                    self.assertEqual(rc[:2],(0,0))
                # The next unseen frame used to request landing at 5.5 seconds.
                rc,ready=search.step(5.6,frames[-1]['h'],frames[-1]['yaw'],False)
                self.assertEqual(rc,(0,0,15,0));self.assertFalse(ready)

    def test_existing_watchdogs_apply_during_search(self):
        for field in ('frame_at','telemetry_at','operator_at','search_updated'):
            with self.subTest(field=field):
                self.setUp();setattr(self.c,field,self.now-1);self.c.watch()
                self.assertEqual(self.c.state,'landing');self.assertEqual(self.commands[-1],'land')

    def test_search_hard_timeout_independent_of_frame_loop(self):
        self.now+=LIMITS['timeout_s'];self.fresh(100);self.c.search_updated=self.now;self.c.watch()
        self.assertEqual(self.c.state,'landing');self.assertIn('timeout',self.c.reason)

    def test_missing_yaw_prevents_takeoff(self):
        c=FlightControl(SearchPolicy(),self.commands.append,self.rows.append,lambda:self.now)
        c.capture_ready=True;c.receive_camera(self.now);c.heartbeat()
        c.receive_telemetry(dict(bat=80,h=0,temph=30),self.now)
        before=list(self.commands)
        with self.assertRaisesRegex(ValueError,'yaw'):c.start()
        self.assertEqual(self.commands,before)

    def test_detection_while_turning_stops_and_hands_over(self):
        self.step(h=120,dt=.6)
        for i in range(10):self.step(h=120,yaw=i)
        self.assertEqual(self.commands[-1],'rc 0 0 0 15')
        for _ in range(7):
            self.step(visible=True,h=120,yaw=9)
            if self.c.state=='active':break
        self.assertEqual(self.c.state,'active');self.assertEqual(self.commands[-1],'rc 0 0 0 0')
        self.assertEqual(self.policy.predictions,0)
        self.step(visible=True,h=120,yaw=9);self.assertEqual(self.policy.predictions,1)

    def test_yaw_stalls_then_lands(self):
        self.step(h=120,dt=.6)
        for _ in range(55):
            if self.c.state!='acquiring':break
            self.step(h=120)
        self.assertEqual(self.c.state,'landing');self.assertIn('turn stalled',self.c.reason)

    def test_horizontal_proposal_rejected_at_command_boundary(self):
        self.c.search.step=lambda *args:((10,0,0,0),False)
        self.step(dt=.6)
        self.assertEqual(self.commands,['takeoff','rc 0 0 0 0','land'])
        self.assertIn('forbidden horizontal',self.c.reason)

    def test_upward_proposal_at_stopping_height_rejected_at_boundary(self):
        self.c.search.step=lambda *args:((0,0,15,0),False)
        self.step(dt=.6,h=180)
        self.assertEqual(self.commands,['takeoff','rc 0 0 0 0','land'])
        self.assertIn('stopping height',self.c.reason)

    def test_high_initial_altitude_skips_lower_stages_without_rising(self):
        for i in range(10):self.step(h=180,yaw=i)
        self.assertEqual(self.c.state,'acquiring');self.assertEqual(self.c.search.stage,2)
        self.assertTrue(all(row['command'].split()[3]=='0' for row in self.rows))

    def test_high_ring_handoff_and_ceiling_persist_into_policy(self):
        for _ in range(8):
            self.step(h=180,visible=True)
            if self.c.state=='active':break
        self.assertEqual(self.c.state,'active')
        self.step(h=180,visible=True);self.assertEqual(self.policy.predictions,1)
        self.step(h=190,visible=True)
        self.assertEqual(self.c.state,'landing');self.assertEqual(self.policy.predictions,1)
        self.assertIn('190',self.c.reason)

    def test_measured_slow_reverse_turn_is_not_cut_at_twelve_seconds(self):
        s=StartupSearch(0,0);yaw=0.;now=0.;rc=(0,0,0,0);reverse_at=None
        while now<30:
            now+=.1;yaw+=rc[3]/15*1.2
            rc,ready=s.step(now,120,yaw,False)
            self.assertFalse(ready)
            if s.leg==1 and reverse_at is None:reverse_at=now
            if s.leg==2:break
        self.assertIsNotNone(reverse_at);self.assertEqual(s.leg,2)
        self.assertGreater(now-reverse_at,11.5)
        self.assertLess(now-reverse_at,18)

    def test_yaw_progress_does_not_remove_leg_deadline(self):
        s=StartupSearch(0,0);s.step(.6,120,0,False)
        # 3deg/s makes >=10degree progress each4s but cannot reach75 in18s.
        with self.assertRaisesRegex(ValueError,'yaw target timeout'):
            for i in range(1,205):
                now=.6+i*.1;s.step(now,120,max(0,now-1.2)*3,False)

    def test_sensor_validation_and_ceiling_precede_visible_handoff(self):
        for height in (float('nan'),None,True,20,190,200):
            with self.subTest(height=height):
                s=StartupSearch(0,0)
                with self.assertRaises(ValueError):s.step(.6,height,0,True)

    def test_yaw_bounds_wrap_and_bad_sensor(self):
        s=StartupSearch(0,179);s.step(.1,120,-179,False)
        self.assertEqual(s.yaw_excursion,2)
        for yaw in (None,float('nan'),181):
            with self.assertRaises(ValueError):StartupSearch(0,yaw)
        s=StartupSearch(0,0);s.step(.6,120,30,False);s.step(.7,120,60,False);s.step(.8,120,90,False)
        with self.assertRaisesRegex(ValueError,'bound'):s.step(.9,120,91,False)

    def test_search_stop_limits_and_send_log_failures(self):
        for kind in ('ceiling','floor','negative','battery','temperature','log','send'):
            with self.subTest(kind=kind):
                self.setUp();self.now+=.6;self.fresh(80)
                if kind=='ceiling':self.c.telemetry['h']=190
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
