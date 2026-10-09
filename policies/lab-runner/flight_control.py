"""Explicit supervised-flight state machine; no sockets are opened here.

Startup search is explicitly scripted and separately logged. After handoff all
navigation RC comes from the learned policy. This is not flight acceptance.
"""
import math
import threading
import time
import numpy as np
from src.command_rehearsal import encode_action
from startup_search import StartupSearch,LIMITS


class FlightControl:
    def __init__(self, policy, send, log, clock=time.monotonic):
        self.policy, self.send, self.log, self.clock = policy, send, log, clock
        self.lock = threading.RLock()
        self.state, self.reason = 'ground', ''
        self.telemetry = {}; self.telemetry_at = self.frame_at = self.operator_at = None
        self.last_prediction = None; self.last_issued = np.zeros(4)
        self.takeoff_at = self.active_at = self.stop_at = None
        self.takeoff_ack = False; self.may_airborne = False
        self.last_land = -math.inf
        self.last_sequence = -1
        self.capture_ready = False
        self.first_stop_reason = ""
        self.search=None; self.last_search_sequence=-1; self.search_updated=None
        self.policy_steps=0; self.first_policy_at=self.last_policy_at=None

    def fresh(self, stamp, age):
        return stamp is not None and math.isfinite(stamp) and 0 <= self.clock()-stamp <= age

    def problem(self, require_operator=True):
        if not self.capture_ready: return 'recording not ready'
        if not self.fresh(self.frame_at, .25): return 'stale camera'
        if not self.fresh(self.telemetry_at, .5): return 'stale telemetry'
        if require_operator and not self.fresh(self.operator_at, .5): return 'flight UI contact lost'
        for name in ('bat', 'h', 'temph'):
            v=self.telemetry.get(name)
            if isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v):
                return 'missing or invalid telemetry '+name
        if not 0 <= self.telemetry['bat'] <= 100 or self.telemetry['h'] < 0 or self.telemetry['temph'] < 0:
            return 'telemetry outside valid range'
        if getattr(self.policy,'startup_search',False):
            yaw=self.telemetry.get('yaw')
            if isinstance(yaw,bool) or not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or abs(yaw)>180:
                return 'missing or invalid yaw for startup search'
        if self.telemetry['bat'] < 30: return 'battery below 30%'
        if self.telemetry['temph'] >= 80: return 'temperature at or above 80 C'
        if getattr(self.policy,'startup_search',False):
            if self.telemetry['h'] >= LIMITS['ceiling_height_cm']:
                return f"height at or above test ceiling {LIMITS['ceiling_height_cm']} cm"
        elif self.telemetry['h'] > 150: return 'height above test ceiling 150 cm'
        if self.state in ('acquiring','active') and self.telemetry['h'] < 30: return 'height below test floor 30 cm'
        return ''

    def heartbeat(self):
        with self.lock: self.operator_at=self.clock()

    def expire_idle(self, deadline):
        # Serialize with start so an expired preflight window cannot stop a flight.
        with self.lock:
            if self.state == 'ground' and self.clock() >= deadline:
                self.stop('three-minute preflight inactivity limit')

    def receive_telemetry(self, fields, received):
        with self.lock:
            if self.telemetry_at is None or received > self.telemetry_at:
                self.telemetry, self.telemetry_at = dict(fields), received

    def receive_camera(self, received):
        with self.lock:
            if self.frame_at is None or received > self.frame_at: self.frame_at=received

    def _send(self, command):
        # Sink must be nonblocking. Log I/O failure cannot prevent a landing send.
        try:
            self.send(command)
            return True
        except Exception as exc:
            self.reason='send failure: '+str(exc)
            return False

    def start(self):
        with self.lock:
            if self.state != 'ground': raise ValueError('One flight per launch; restart for another test')
            problem=self.problem()
            if problem: raise ValueError(problem)
            if self.telemetry['h'] > 10: raise ValueError('Takeoff requires the drone on the ground')
            self.policy.reset()
            self.state='taking_off'; self.takeoff_at=self.clock(); self.may_airborne=True
            if not self._send('takeoff'): self.stop('takeoff send failure')

    def acknowledge(self, reply, received):
        with self.lock:
            if self.state == 'taking_off' and received >= self.takeoff_at:
                if reply.lower() == 'ok': self.takeoff_ack=True
                else: self.stop('takeoff rejected: '+reply)

    def stop(self, reason='operator land'):
        with self.lock:
            if self.state in ('landing','stopped'): return
            self.reason=reason; self.first_stop_reason=self.first_stop_reason or reason; self.stop_at=self.clock()
            self.state='landing' if self.may_airborne else 'stopped'
            if self.may_airborne: self._land()

    def _land(self):
        if self._send('rc 0 0 0 0'): self.last_issued=np.zeros(4)
        self._send('land'); self.last_land=self.clock()

    def emergency(self):
        with self.lock:
            self.state='stopped';self.reason='operator EMERGENCY motor cut'; self.stop_at=self.clock()
            self.first_stop_reason=self.first_stop_reason or self.reason
            self._send('emergency')

    def watch(self):
        with self.lock:
            now=self.clock()
            if self.state in ('taking_off','acquiring','active'):
                problem=self.problem()
                if problem: self.stop(problem); return
                if self.state == 'taking_off':
                    if now-self.takeoff_at >= 12: self.stop('takeoff acknowledgement/height timeout')
                    elif self.takeoff_ack and self.telemetry_at > self.takeoff_at and self.telemetry['h'] >= 30:
                        if getattr(self.policy,'startup_search',False):
                            try:self.search=StartupSearch(now,self.telemetry.get('yaw'))
                            except ValueError as exc:self.stop(str(exc));return
                            self.state='acquiring';self.search_updated=now
                        else:
                            self.state='active'; self.active_at=now; self.last_prediction=now
                elif self.state == 'acquiring':
                    if now-self.search.started>=LIMITS['timeout_s']:self.stop('startup search timeout: ring not acquired')
                    elif not self.fresh(self.search_updated,.3):self.stop('startup detection/history processing stalled')
                elif not self.fresh(self.last_prediction,.3): self.stop('policy inference stalled')
            elif self.state == 'landing':
                # Keep requesting landing even if takeoff was cancelled mid-ascent.
                # A UDP acknowledgement or height reading alone is not motor-state proof.
                if now-self.stop_at >= 20: self.state='stopped'
                elif now-self.last_land >= 1: self._land()

    def step(self, sequence, received, features):
        if self.state == 'acquiring':return self.search_step(sequence,received,features)
        with self.lock:
            if self.state != 'active' or sequence <= self.last_sequence: return None
            problem=self.problem()
            if problem: self.stop(problem); return None
            if not self.fresh(received,.25): self.stop('stale policy frame'); return None
            if self.policy_steps == 0 and getattr(self.policy,'require_visible_start',False):
                camera=np.asarray(features)
                if camera.shape != (8,) or not np.isfinite(camera).all() or camera[6] != 1:
                    self.stop('ring not acquired at policy takeover');return None
            previous=self.last_issued.copy()
        started=self.clock()
        try:
            observation, raw=self.policy.predict(features, previous)
            command, issued=encode_action(raw)
            if np.asarray(observation).shape != (getattr(self.policy,'inputs',85),) or not np.isfinite(observation).all():
                raise ValueError('invalid selected-policy observation')
        except Exception as exc:
            self.stop('policy error: '+str(exc)); return None
        with self.lock:
            # A delayed model result must not undo a release/watchdog landing.
            if self.state != 'active': return None
            problem=self.problem()
            if problem or not self.fresh(received,.25):
                self.stop(problem or 'frame expired during inference');return None
            record=dict(frame_id=sequence,receipt_monotonic=received,
                analysis_started_monotonic=started,analysis_completed_monotonic=self.clock(),
                camera_features=np.asarray(features).tolist(),observation=observation.tolist(),
                raw_policy_action=raw.tolist(),suggested_action=raw.tolist(),
                previous_issued_action=previous.tolist(),issued_action=[],
                command=command,frame_age_s=self.clock()-received,
                mode='physical supervised policy test',
                policy_identity=getattr(self.policy,'identity',None))
            # Require writable policy evidence before any nonzero movement send.
            try: self.log(dict(record,event='proposal'))
            except Exception as exc: self.stop('policy log failure: '+str(exc));return None
            sent=self._send(command)
            if sent:
                self.last_issued=issued; self.last_sequence=sequence; self.last_prediction=self.clock()
                self.policy_steps+=1
                if self.first_policy_at is None:self.first_policy_at=self.clock()
                self.last_policy_at=self.clock()
                record['issued_action']=issued.tolist()
            record['sent']=sent;record['event']='send_result'
            try: self.log(record)
            except Exception as exc: self.stop('policy result log failure: '+str(exc))
            if not sent: self.stop('policy send failure')
            return record

    def search_step(self,sequence,received,features):
        with self.lock:
            if self.state!='acquiring' or sequence<=self.last_search_sequence:return None
            problem=self.problem()
            if problem or not self.fresh(received,.25):self.stop(problem or 'stale search frame');return None
            camera=np.asarray(features,dtype=np.float32)
            if camera.shape!=(8,) or not np.isfinite(camera).all():self.stop('invalid search camera features');return None
            previous=self.last_issued.copy()
        # Allow operator/watchdog landing while the history estimator runs.
        try:observation=self.policy.observe(camera,previous)
        except Exception as exc:self.stop('startup history failure: '+str(exc));return None
        with self.lock:
            if self.state!='acquiring':return None
            problem=self.problem()
            if problem or not self.fresh(received,.25):self.stop(problem or 'search frame expired');return None
            if np.asarray(observation).shape!=(getattr(self.policy,'inputs',89),) or not np.isfinite(observation).all():
                self.stop('invalid startup history observation');return None
            try:rc,ready=self.search.step(self.clock(),self.telemetry['h'],self.telemetry.get('yaw'),camera[6]==1)
            except ValueError as exc:self.stop(str(exc));return None
            # Acquisition must never issue horizontal translation, even if a
            # future search implementation accidentally proposes one.
            if len(rc)!=4 or rc[0]!=0 or rc[1]!=0:
                self.stop('forbidden horizontal startup movement');return None
            if rc[2]>0 and self.telemetry['h']>=LIMITS['target_height_cm']:
                self.stop('forbidden climb above startup stopping height');return None
            command='rc '+' '.join(map(str,rc));issued=np.asarray(rc,np.float64)/[50,50,100,70]
            row=dict(event='startup_proposal',phase=self.search.phase,controller='scripted_startup_search',
                frame_id=sequence,receipt_monotonic=received,analysis_completed_monotonic=self.clock(),
                camera_features=camera.tolist(),observation=observation.tolist(),
                previous_issued_action=previous.tolist(),command=command,issued_action=[],
                ring_confirmed=bool(ready),height_cm=self.telemetry['h'],yaw=self.telemetry.get('yaw'),
                policy_identity=getattr(self.policy,'identity',None))
            try:self.log(row.copy())
            except Exception as exc:self.stop('startup log failure: '+str(exc));return None
            sent=self._send(command);row.update(event='startup_send_result',sent=sent)
            if sent:
                self.last_issued=issued;self.last_search_sequence=sequence;self.search_updated=self.clock()
                row['issued_action']=issued.tolist()
            try:self.log(row)
            except Exception as exc:self.stop('startup result log failure: '+str(exc));return None
            if not sent:self.stop('startup send failure');return row
            if ready:
                # Neutral command has been issued and recorded. Keep authentic
                # camera/action/GRU history; never reset it to invented zeros.
                self.state='active';self.active_at=self.clock();self.last_prediction=self.clock()
                self.last_sequence=sequence
            return row

    def snapshot(self):
        with self.lock:
            return dict(state=self.state,reason=self.reason,first_stop_reason=self.first_stop_reason,
                policy_identity=getattr(self.policy,'identity',None),telemetry=dict(self.telemetry),
                readiness=self.problem(require_operator=False),last_issued=self.last_issued.tolist(),
                startup_search_phase=None if self.search is None else self.search.phase,
                startup_search_limits=LIMITS if getattr(self.policy,'startup_search',False) else None,
                startup_search_seconds=None if self.search is None else max(0,(self.active_at or self.stop_at or self.clock())-self.search.started),
                policy_steps=self.policy_steps,
                policy_seconds=0. if self.first_policy_at is None else self.last_policy_at-self.first_policy_at,
                physical_landing_confirmed=False)
