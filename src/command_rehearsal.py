"""Offline command-path rehearsal. The UDP destination MUST be IPv4 loopback.

This is not a flight launcher. No takeoff command exists. It exercises the wire
format and stop/land path against a local SDK stand-in while policy qualification
is incomplete. It does not establish that a physical drone obeys these commands.
"""
import ipaddress
import socket
import threading
import time

import numpy as np

# Existing team mapping from src/labcheck.py; vertical speed is still not fully
# calibrated to the simulator. These values are used only in offline rehearsal.
RC_FULL = np.array([50,50,100,70],dtype=np.float64)


def encode_action(action):
    values=np.asarray(action,dtype=np.float64)
    if values.shape!=(4,) or not np.isfinite(values).all() or np.any(abs(values)>1):
        raise ValueError('Four finite normalized actions in [-1,1] required')
    rc=np.rint(values*RC_FULL).astype(int)
    return 'rc '+' '.join(map(str,rc)),rc/RC_FULL


class LoopbackLink:
    def __init__(self,host,port):
        address=ipaddress.ip_address(host)
        if address.version!=4 or not address.is_loopback:
            raise ValueError('Rehearsal accepts IPv4 loopback only; no drone connection')
        if type(port) is not int or not 1<=port<=65535: raise ValueError('Invalid port')
        self.socket=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        self.socket.connect((str(address),port))
        self.socket.setblocking(False)

    def send(self,command):
        # Keep arbitrary SDK configuration and takeoff out of this test path.
        if command!='land':
            words=command.split()
            if len(words)!=5 or words[0]!='rc': raise ValueError('Unsupported command')
            values=[int(x) for x in words[1:]]
            if any(abs(x)>100 for x in values): raise ValueError('RC exceeds SDK range')
        self.socket.send(command.encode('ascii'))

    def close(self):
        self.socket.close()


class RehearsalController:
    """Latching stop with an independent watchdog, no automatic restart.

    Receive timestamps must come from the frame/telemetry/operator producers.
    Repeated inference on a stale frame cannot refresh the camera timestamp.
    STOP and LAND both cancel motion then request landing. No motor-cut command
    is mapped to an ordinary stop key.
    """
    def __init__(self,link,clock=time.monotonic):
        if not isinstance(link,LoopbackLink): raise TypeError('LoopbackLink required')
        self.link=link;self.clock=clock;self.lock=threading.RLock()
        self.state='disarmed';self.reason='';self.last_issued=np.zeros(4)
        self.frame_at=self.telemetry_at=self.operator_at=self.prediction_at=None
        self.events=[];self.errors=[];self.deadline=None
        self.stop_event=threading.Event();self.thread=None

    def _send(self,command):
        self.link.send(command)
        self.events.append({'at':self.clock(),'command':command})

    def _fresh(self,now,stamp,limit):
        return stamp is not None and np.isfinite(stamp) and 0<=now-stamp<=limit

    def _inputs_fresh(self,now,frame_at,telemetry_at,operator_at):
        return (self._fresh(now,frame_at,.25) and self._fresh(now,telemetry_at,.5)
                and self._fresh(now,operator_at,.5))

    def arm(self,frame_at,telemetry_at,operator_at):
        with self.lock:
            if self.state!='disarmed': raise RuntimeError('Create a new rehearsal after a stop')
            now=self.clock()
            if not self._inputs_fresh(now,frame_at,telemetry_at,operator_at):
                raise ValueError('Fresh camera, telemetry and operator input required')
            self.frame_at,self.telemetry_at,self.operator_at=frame_at,telemetry_at,operator_at
            self.prediction_at=now;self.deadline=now+30;self.state='active'

    def submit(self,action,frame_at,telemetry_at,operator_at):
        with self.lock:
            if self.state!='active': return np.zeros(4)
            now=self.clock()
            if now>=self.deadline:
                self.stop_and_land('rehearsal time limit');return np.zeros(4)
            if not self._inputs_fresh(now,frame_at,telemetry_at,operator_at):
                self.stop_and_land('stale input');return np.zeros(4)
            # Never accept a delayed result from an older camera frame.
            if frame_at<self.frame_at:
                self.stop_and_land('out-of-order frame');return np.zeros(4)
            try:
                command,issued=encode_action(action)
                self._send(command)
            except (ValueError,OSError) as exc:
                self.errors.append(str(exc));self.stop_and_land('invalid action or send failure')
                return np.zeros(4)
            self.frame_at,self.telemetry_at,self.operator_at=frame_at,telemetry_at,operator_at
            self.prediction_at=now;self.last_issued=issued
            return issued.copy()

    def stop_and_land(self,reason='manual stop'):
        with self.lock:
            if self.state!='active': return
            # Latch before I/O so a failing socket cannot let another prediction run.
            self.state='stopped';self.reason=reason;self.last_issued=np.zeros(4)
            for command in ('rc 0 0 0 0','land'):
                try: self._send(command)
                except OSError as exc: self.errors.append(str(exc))

    def watchdog_once(self):
        with self.lock:
            if self.state!='active': return
            now=self.clock()
            if (not self._inputs_fresh(now,self.frame_at,self.telemetry_at,self.operator_at)
                    or not self._fresh(now,self.prediction_at,.25) or now>=self.deadline):
                self.stop_and_land('watchdog')

    def start_watchdog(self):
        if self.thread is not None: raise RuntimeError('Watchdog already started')
        def watch():
            while not self.stop_event.wait(.02): self.watchdog_once()
        self.thread=threading.Thread(target=watch,daemon=True);self.thread.start()

    def handle_key(self,key):
        if key in (' ','Escape','l','L','q','Q','Ctrl+C'):
            self.stop_and_land('operator '+repr(key))

    def close(self):
        self.stop_and_land('shutdown')
        self.stop_event.set()
        if self.thread is not None: self.thread.join(timeout=1)
        self.link.close()
