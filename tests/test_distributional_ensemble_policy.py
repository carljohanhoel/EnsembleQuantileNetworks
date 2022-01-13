import unittest
import numpy as np

import sys
sys.path.append('../src')
from policy import DistributionalEnsembleTestPolicy


class Tester(unittest.TestCase):

    def test_standard(self):
        policy = DistributionalEnsembleTestPolicy()
        for i in range(0, 100):
            z_values_all_nets = np.random.rand(10, 32, 4)
            z_values_all_nets[:, :, 0] += 1   # Action 0 should then be 'best'
            action, policy_info = policy.select_action(z_values_all_nets=z_values_all_nets)
            self.assertEqual(action, 0)

        # Test for batch
        z_values_all_nets = np.random.rand(32, 10, 32, 4)
        best_idx = np.random.randint(0, 4, 32)
        for batch in range(0, 32):
            idx = best_idx[batch]
            z_values_all_nets[batch, :, :, idx] += 1
        action, policy_info = policy.select_action(z_values_all_nets=z_values_all_nets)
        self.assertTrue((action == best_idx).all())

    def test_safe_policy(self):
        policy = DistributionalEnsembleTestPolicy(aleatoric_threshold=0.1)
        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[..., 2] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[0:5, :, 2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[:, 0:10, 2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 4)
        self.assertTrue(policy_info['safe_action'])

        policy = DistributionalEnsembleTestPolicy(epistemic_threshold=0.1)
        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[..., 2] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[:, 0:10, 2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[0:5, :, :2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 4)
        self.assertTrue(policy_info['safe_action'])

        policy = DistributionalEnsembleTestPolicy(aleatoric_threshold=0.1, epistemic_threshold=0.1)
        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[..., 2] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[:, 0:10, 2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 4)
        self.assertTrue(policy_info['safe_action'])

        z_values_all_nets = np.ones([10, 32, 4])
        z_values_all_nets[0:5, :, :2] *= 100
        z_values_all_nets[..., 3] *= 1.001
        action, policy_info = policy.select_action(z_values_all_nets)
        self.assertEqual(action, 4)
        self.assertTrue(policy_info['safe_action'])


if __name__ == '__main__':
    unittest.main()
