"""Operator-authorized acquisition only; never a learned-policy result.

No sockets or position estimate. Commands use fresh measured height/yaw and real
detector flags. No forward/lateral search. Limits are experimental settings,
not measured ring height or a physical-flight qualification.
"""
import math

LIMITS = dict(target_height_cm=120,lower_height_cm=80,up_rc=15,down_rc=15,yaw_rc=15,timeout_s=20,
              ascent_timeout_s=5,descent_timeout_s=5,sweep_leg_s=3,settle_s=.5,
              confirm_s=.4,confirm_frames=3,max_yaw_excursion_deg=60)

class StartupSearch:
    def __init__(self,now,yaw):
        if isinstance(yaw,bool) or not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or abs(yaw)>180:
            raise ValueError('valid yaw telemetry required for startup search')
        self.started=now;self.initial_yaw=yaw;self.last_yaw=yaw;self.yaw_excursion=0.
        self.last_time=now;self.scan_at=None;self.visible_at=None;self.visible_count=0
        self.phase='settling after takeoff';self.last_frame_at=now
        self.descent_at=None

    def step(self,now,height,yaw,visible):
        if now<self.last_time:raise ValueError('search clock moved backwards')
        self.last_time=now
        if isinstance(yaw,bool) or not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or abs(yaw)>180:
            raise ValueError('valid yaw telemetry required for startup search')
        delta=(yaw-self.last_yaw+180)%360-180
        if abs(delta)>45:raise ValueError('implausible yaw jump during startup search')
        self.yaw_excursion+=delta;self.last_yaw=yaw
        if abs(self.yaw_excursion)>LIMITS['max_yaw_excursion_deg']:
            raise ValueError('startup search yaw bound exceeded')
        if now-self.started>=LIMITS['timeout_s']:raise ValueError('startup search timeout: ring not acquired')
        # A single flicker or a gap in analysed frames cannot authorize handoff.
        if now-self.last_frame_at>.25:self.visible_at=None;self.visible_count=0
        self.last_frame_at=now
        if visible:
            if self.visible_at is None:self.visible_at=now;self.visible_count=0
            self.visible_count+=1;self.phase='holding still to confirm ring'
            ready=(now-self.started>=LIMITS['settle_s'] and
                   now-self.visible_at>=LIMITS['confirm_s'] and
                   self.visible_count>=LIMITS['confirm_frames'])
            return (0,0,0,0),ready
        self.visible_at=None;self.visible_count=0
        elapsed=now-self.started
        if elapsed<LIMITS['settle_s']:
            self.phase='settling after takeoff';return (0,0,0,0),False
        if self.scan_at is None and height<LIMITS['target_height_cm']:
            if elapsed>=LIMITS['settle_s']+LIMITS['ascent_timeout_s']:
                raise ValueError('startup search ascent timeout')
            self.phase='rising slowly to at most 120 cm';return (0,0,LIMITS['up_rc'],0),False
        if self.scan_at is None:self.scan_at=now
        scan=now-self.scan_at
        if scan>=2*LIMITS['sweep_leg_s']:
            if self.descent_at is None:self.descent_at=now
            if height<=LIMITS['lower_height_cm']:
                raise ValueError('vertical startup search complete: ring not acquired')
            if now-self.descent_at>=LIMITS['descent_timeout_s']:
                raise ValueError('startup search descent timeout')
            self.phase='lowering slowly toward 80 cm to find ring'
            return (0,0,-LIMITS['down_rc'],0),False
        direction=1 if scan<LIMITS['sweep_leg_s'] else -1
        self.phase='looking clockwise' if direction==1 else 'looking counterclockwise'
        # Neutral at the angular bound; time and opposite sweep still advance.
        turning=0 if direction*self.yaw_excursion>=LIMITS['max_yaw_excursion_deg'] else direction*LIMITS['yaw_rc']
        return (0,0,0,turning),False
