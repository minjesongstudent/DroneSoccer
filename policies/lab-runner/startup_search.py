"""Operator-authorized acquisition only; never a learned-policy result.

Fresh measured height/yaw and real detector flags drive a bounded staged search.
Zero horizontal RC is not a position hold. Limits are experimental settings,
not validated clearance or physical-flight qualification.
"""
import math

LIMITS = dict(stage_heights_cm=(120,150,180),target_height_cm=180,
              ceiling_height_cm=190,upward_stopping_allowance_cm=10,
              up_rc=15,yaw_rc=15,timeout_s=120,
              ascent_timeout_s=32,ascent_progress_s=4,ascent_progress_cm=10,
              sweep_targets_deg=(75,-75,0),sweep_leg_s=18,
              yaw_progress_s=4,yaw_progress_deg=10,
              yaw_tolerance_deg=5,settle_s=.5,
              confirm_s=.4,confirm_frames=3,max_yaw_excursion_deg=90)


def _finite_number(value):
    return not isinstance(value,bool) and isinstance(value,(int,float)) and math.isfinite(value)


class StartupSearch:
    def __init__(self,now,yaw):
        if not _finite_number(yaw) or abs(yaw)>180:
            raise ValueError('valid yaw telemetry required for startup search')
        self.started=now;self.initial_yaw=yaw;self.last_yaw=yaw;self.yaw_excursion=0.
        self.last_time=now;self.scan_at=None;self.visible_at=None;self.visible_count=0
        self.phase='settling after takeoff';self.last_frame_at=now
        self.progress_at=None;self.progress_height=None
        self.stage=0;self.leg=0;self.leg_at=None
        self.yaw_progress_at=None;self.yaw_progress_error=None
        self.ascent_seconds=0.;self.previous_rc=(0,0,0,0)

    def _result(self,rc=(0,0,0,0),ready=False):
        self.previous_rc=rc
        return rc,ready

    def step(self,now,height,yaw,visible):
        if not _finite_number(now) or now<self.last_time:
            raise ValueError('search clock moved backwards or is invalid')
        if self.previous_rc[2]>0:self.ascent_seconds+=now-self.last_time
        self.last_time=now
        if not _finite_number(height) or height<30:
            raise ValueError('invalid or below-floor startup height')
        if height>=LIMITS['ceiling_height_cm']:
            raise ValueError('height at or above test ceiling 190 cm')
        if not _finite_number(yaw) or abs(yaw)>180:
            raise ValueError('valid yaw telemetry required for startup search')
        delta=(yaw-self.last_yaw+180)%360-180
        if abs(delta)>45:raise ValueError('implausible yaw jump during startup search')
        self.yaw_excursion+=delta;self.last_yaw=yaw
        if abs(self.yaw_excursion)>LIMITS['max_yaw_excursion_deg']:
            raise ValueError('startup search yaw bound exceeded')
        if now-self.started>=LIMITS['timeout_s']:
            raise ValueError('startup search timeout: ring not acquired')
        # A flicker or a gap in analysed frames cannot authorize handoff.
        if now-self.last_frame_at>.25:self.visible_at=None;self.visible_count=0
        self.last_frame_at=now
        if visible:
            self.progress_at=None;self.progress_height=None
            self.yaw_progress_at=None;self.yaw_progress_error=None
            if self.visible_at is None:self.visible_at=now;self.visible_count=0
            self.visible_count+=1;self.phase='holding still to confirm ring'
            ready=(now-self.started>=LIMITS['settle_s'] and
                   now-self.visible_at>=LIMITS['confirm_s'] and
                   self.visible_count>=LIMITS['confirm_frames'])
            return self._result(ready=ready)
        self.visible_at=None;self.visible_count=0
        if now-self.started<LIMITS['settle_s']:
            self.phase='settling after takeoff';return self._result()

        heights=LIMITS['stage_heights_cm']
        if self.scan_at is None:
            # If takeoff/drift has already passed a lower stage, scan at the
            # present higher stage instead of commanding a descent/re-ascent.
            while self.stage<len(heights)-1 and height>=heights[self.stage+1]:self.stage+=1
            target=heights[self.stage]
            if height<target:
                if self.ascent_seconds>=LIMITS['ascent_timeout_s']:
                    raise ValueError('startup search cumulative ascent timeout')
                # Recorded h is quantized in 10cm steps (~6cm/s at RC15).
                if self.progress_at is None or height>=self.progress_height+LIMITS['ascent_progress_cm']:
                    self.progress_at=now;self.progress_height=height
                elif now-self.progress_at>=LIMITS['ascent_progress_s']:
                    raise ValueError('startup climb stalled: no 10 cm height gain in 4 s')
                self.phase=f'rising slowly toward stage {self.stage+1}: {target} cm'
                return self._result((0,0,LIMITS['up_rc'],0))
            self.scan_at=now;self.leg_at=None;self.leg=0
            self.progress_at=None;self.progress_height=None
        if now-self.scan_at<LIMITS['settle_s']:
            self.phase=f'settling at search stage {self.stage+1}'
            return self._result()

        targets=LIMITS['sweep_targets_deg']
        target=targets[self.leg]
        if self.leg_at is None:
            self.leg_at=now;self.yaw_progress_at=None;self.yaw_progress_error=None
        if abs(target-self.yaw_excursion)<=LIMITS['yaw_tolerance_deg']:
            self.leg+=1;self.leg_at=None
            if self.leg==len(targets):
                if self.stage==len(heights)-1:
                    raise ValueError('staged startup search complete: ring not acquired')
                self.stage+=1;self.scan_at=None
            self.phase='holding between search stages or turns'
            return self._result()
        if now-self.leg_at>=LIMITS['sweep_leg_s']:
            raise ValueError('startup yaw target timeout')
        error=abs(target-self.yaw_excursion)
        if self.yaw_progress_at is None or error<=self.yaw_progress_error-LIMITS['yaw_progress_deg']:
            self.yaw_progress_at=now;self.yaw_progress_error=error
        elif now-self.yaw_progress_at>=LIMITS['yaw_progress_s']:
            raise ValueError('startup turn stalled: no 10 degree progress in 4 s')
        direction=1 if target>self.yaw_excursion else -1
        self.phase=f'scanning stage {self.stage+1}: yaw {target:+} degrees from start'
        return self._result((0,0,0,direction*LIMITS['yaw_rc']))
