import unittest
import numpy as np
from tools.audit_motion_prediction import body_frame, episode_features


class MotionPredictionTests(unittest.TestCase):
    def test_body_axes(self):
        np.testing.assert_allclose(body_frame([[1,0,0],[0,1,0]], [0,np.pi/2]),
                                   [[1,0,0],[1,0,0]], atol=1e-6)

    def test_features_do_not_read_future_positions_or_commands(self):
        n = 20
        actions = np.arange(n*4,dtype=np.float32).reshape(n,4)/100
        obs = np.zeros((n,85),np.float32)
        for i in range(n):
            previous = np.zeros((3,4),np.float32)
            for lag in range(1,4):
                if i >= lag:
                    previous[3-lag] = actions[i-lag]
            obs[i,72:84] = previous.ravel()
        z = dict(actions=actions,actor_observations=obs,
                 positions=np.arange(n*3).reshape(n,3)*.001,headings=np.zeros(n))
        original = episode_features(z)
        changed = {k:v.copy() for k,v in z.items()}
        changed['positions'][10:] += 100
        # Target row for action10 changes; its prior-motion features cannot.
        altered = episode_features(changed)
        for name in ['public','commands12','prior_motion_truth']:
            np.testing.assert_array_equal(original[name][:9],altered[name][:9])
        self.assertFalse(np.array_equal(original['target'][8],altered['target'][8]))
        np.testing.assert_allclose(original['prior'][8],[3,-3,3],atol=1e-5)
        # History reset has zero commands before this recorded episode.
        np.testing.assert_array_equal(original['commands12'][0,89:],np.zeros(36))
        changed = {k:v.copy() for k,v in z.items()}
        changed['actions'][11:] *= -1
        for i in range(n):
            for lag in range(1,4):
                if i >= lag:
                    changed['actor_observations'][i,72+4*(3-lag):76+4*(3-lag)] = changed['actions'][i-lag]
        altered = episode_features(changed)
        for name in ['public','commands12','prior_motion_truth']:
            np.testing.assert_array_equal(original[name][:9],altered[name][:9])


if __name__ == '__main__':
    unittest.main()
