import unittest
from unittest.mock import Mock
import json
from pathlib import Path
import cv2
import numpy as np
from bridge import ObservationBridge,detect
from camera_policy import send
from src.curriculum import LessonEnv
from src.return_reset_curriculum import wrap

class BridgeTests(unittest.TestCase):
    def test_exact_simulator_public_history(self):
        env=wrap(LessonEnv(stage=0),single=True,metric_ring=True,geometric_return=True)
        expected,_=env.reset(seed=5000001)
        bridge=ObservationBridge()
        np.testing.assert_array_equal(bridge.update(env.unwrapped.observe()),expected)
        rng=np.random.default_rng(3)
        for _ in range(60):
            action=rng.uniform(-.2,.2,4).astype(np.float32)
            expected,_,term,trunc,_=env.step(action)
            np.testing.assert_array_equal(bridge.update(env.unwrapped.observe(),action),expected)
            if term or trunc:break
        env.close()
    def test_detector_never_invents_marker(self):
        config=json.loads((Path(__file__).parent/'config.json').read_text())
        image=np.zeros((720,960,3),np.uint8)
        cv2.circle(image,(480,360),150,(255,0,0),25)
        obs,ellipse,_=detect(image,config)
        self.assertEqual(obs[6],1);self.assertEqual(obs[7],0)
        self.assertLess(abs(obs[0]),.01)
        blank,_,_=detect(np.zeros_like(image),config)
        np.testing.assert_array_equal(blank,np.zeros(8))
    def test_flight_commands_blocked(self):
        sock=Mock()
        for command in ('takeoff','land','rc 0 10 0 0','emergency'):
            with self.assertRaises(ValueError):send(sock,command)
        sock.sendto.assert_not_called()

if __name__=='__main__':unittest.main()
