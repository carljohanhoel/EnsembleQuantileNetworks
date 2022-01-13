import unittest
import numpy as np

import sys
sys.path.append('../src')
from policy import DistributionalEpsGreedyPolicy, DistributionalTestPolicy


class Tester(unittest.TestCase):

    def test_policy(self):
        policy = DistributionalEpsGreedyPolicy(eps=0.)
        for i in range(0, 100):
            z_values = np.random.rand(32, 4)
            z_values[:, 0] += 1   # Action 0 should then be 'best'
            action, info = policy.select_action(z_values=z_values)
            self.assertEqual(action, 0)

        policy = DistributionalEpsGreedyPolicy(eps=0.5)
        action_vec = []
        for i in range(0, 100):
            z_values = np.random.rand(32, 4)
            z_values[:, 0] += 1   # Action 0 should then be 'best'
            action = policy.select_action(z_values=z_values)
            action_vec.append(action)
        self.assertTrue((np.array(action_vec) == 0).any())   # All actions should be chosen sometimes
        self.assertTrue((np.array(action_vec) == 1).any())
        self.assertTrue((np.array(action_vec) == 2).any())
        self.assertTrue((np.array(action_vec) == 3).any())

        # Test for batch
        policy = DistributionalEpsGreedyPolicy(eps=0.)
        z_values = np.random.rand(32, 32, 4)
        best_idx = np.random.randint(0, 4, 32)
        for batch in range(0, 32):
            idx = best_idx[batch]
            z_values[batch, :, idx] += 1
        action, info = policy.select_action(z_values=z_values)
        self.assertTrue((action == best_idx).all())

    def test_safe_policy(self):
        safety_threshold = 0.1
        nb_actions = 8

        policy = DistributionalTestPolicy(safety_threshold=safety_threshold)

        z_values = np.ones([32, nb_actions])   # nb quantiles, nb actions
        z_values[5:, 2] *= 1.01
        action, policy_info = policy.select_action(z_values)
        self.assertEqual(action, 2)
        self.assertFalse(policy_info['safe_action'])

        z_values = np.ones([32, nb_actions])
        z_values[:10, :] *= 0.5
        action, policy_info = policy.select_action(z_values)
        self.assertEqual(action, nb_actions)
        self.assertTrue(policy_info['safe_action'])

        z_values = np.ones([32, nb_actions])
        z_values[0, :] *= -10
        action, policy_info = policy.select_action(z_values)
        self.assertEqual(action, nb_actions)
        self.assertTrue(policy_info['safe_action'])


if __name__ == '__main__':
    unittest.main()
