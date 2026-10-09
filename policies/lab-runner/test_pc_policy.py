"""Real PC weights through the flight command boundary, with mock transport only."""
import json
import unittest
from unittest.mock import patch
import numpy as np
from pc_runtime import PCRuntime,ROOT
from src.estimated_motion_actor import EstimatedMotionPolicy
from src.command_rehearsal import encode_action
from flight_control import FlightControl

class CompositeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.runtime=PCRuntime()

    def test_real_weights_match_direct_composite_and_differ_from_baseline(self):
        r=self.runtime;r.reset()
        e=r.identity
        direct=EstimatedMotionPolicy.load(ROOT/e['path'],ROOT/e['shared_models']['baseline']['path'],
            ROOT/e['shared_models']['estimator']['path'],device='cpu',sha256=e['sha256'])
        from src.camera_history import CameraHistory
        from src.policy_runtime import SPECS
        history=CameraHistory(SPECS['seenflag-s1-level1'].lags)
        rng=np.random.default_rng(271);previous=np.zeros(4,np.float32);differences=[]
        for _ in range(30):
            camera=rng.uniform(-.4,.4,8).astype(np.float32);camera[6:]=[1,0]
            obs,action=r.predict(camera,previous)
            expected=direct.observe(history.update(camera,previous));expected_action,_=direct.predict(expected)
            np.testing.assert_array_equal(obs,expected)
            np.testing.assert_allclose(action,expected_action,atol=1e-7,rtol=0)
            baseline,_=direct.actor.base_actor.predict(obs[:85],deterministic=True)
            differences.append(float(np.max(abs(action-baseline))))
            _,previous=encode_action(action)
        self.assertGreater(max(differences),1e-6)
        self.assertTrue(np.any(r.policy.actor.residual_weight.detach().numpy()!=0))

    def test_takeoff_pc_rc_and_land_with_complete_identity_no_real_socket(self):
        r=self.runtime;r.reset();now=[100.];commands=[];logs=[]
        c=FlightControl(r,commands.append,logs.append,lambda:now[0])
        def fresh(h):
            c.capture_ready=True;c.receive_camera(now[0]);c.receive_telemetry(dict(bat=80,h=h,temph=30,yaw=0),now[0]);c.heartbeat()
        with patch('socket.socket',side_effect=AssertionError('No hardware/network in this test')):
            fresh(0);c.start();now[0]+=.1;fresh(80);c.acknowledge('ok',now[0]);c.watch()
            for seq in range(8):
                now[0]+=.1;fresh(80)
                c.step(seq,now[0],np.array([0,0,.2,.2,0,0,1,0],np.float32))
                if c.state=='active':break
            self.assertEqual(c.state,'active')
            for seq in range(20,24):
                now[0]+=.1;fresh(80)
                row=c.step(seq,now[0],np.array([0,0,.2,.2,0,0,1,0],np.float32))
                self.assertEqual(len(row['observation']),89)
                self.assertEqual(row['policy_identity']['sha256'],r.identity['sha256'])
                self.assertEqual(row['command'],encode_action(row['raw_policy_action'])[0])
            c.stop('operator land')
        self.assertEqual(commands[0],'takeoff');self.assertEqual(commands[-1],'land')
        self.assertEqual(len([x for x in logs if x['event']=='send_result']),4)

    def test_every_saved_checkpoint_loads_by_exact_identity(self):
        entries=json.loads((ROOT/'policies/registry.json').read_text())['checkpoints']
        for e in entries:
            with self.subTest(checkpoint=e['id']):
                runtime=PCRuntime(checkpoint_id=e['id'])
                features,action=runtime.predict(np.zeros(8,np.float32),np.zeros(4,np.float32))
                self.assertEqual(features.shape,(89,));self.assertTrue(np.isfinite(action).all())

    def test_archive_is_idempotent_and_refuses_overwrite(self):
        import importlib.util,shutil,tempfile
        from pathlib import Path
        spec=importlib.util.spec_from_file_location('checkpoint_archiver',ROOT/'policies/save_checkpoint.py')
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'policies').mkdir()
            shutil.copy2(ROOT/'policies/registry.json',root/'policies/registry.json')
            evidence=root/'results.json';evidence.write_text('{"randomised":{"crashes":12,"episodes":25}}')
            source=ROOT/self.runtime.identity['path']
            first=mod.save(source,evidence,'test-run','arm',1,root)
            self.assertEqual(first,mod.save(source,evidence,'test-run','arm',1,root))
            self.assertEqual((root/first['path']).read_bytes(),source.read_bytes())
            evidence.write_text('{"changed":true}')
            with self.assertRaises(FileExistsError):mod.save(source,evidence,'test-run','arm',1,root)

if __name__=='__main__':unittest.main()
