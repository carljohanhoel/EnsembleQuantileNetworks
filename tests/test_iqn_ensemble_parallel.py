import unittest
import numpy as np

import sys
sys.path.append('../src')
from iqn_ensemble import IqnRpfAgentParallel
from memory import BootstrappingMemory
from policy import DistributionalEpsGreedyPolicy, DistributionalEnsembleTestPolicy


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.nb_nets = 3
        greedy_policy = DistributionalEpsGreedyPolicy(eps=0)
        test_policy = DistributionalEnsembleTestPolicy()
        memory = BootstrappingMemory(nb_nets=self.nb_nets, limit=10000, adding_prob=0.5, window_length=1)
        self.agent = IqnRpfAgentParallel(nb_models=self.nb_nets,
                                         nb_actions=4,
                                         memory=memory,
                                         cnn_architecture=True,
                                         learning_rate=0.01,
                                         nb_ego_states=1,
                                         nb_states_per_vehicle=3,
                                         nb_vehicles=3,
                                         nb_conv_layers=2,
                                         nb_conv_filters=32,
                                         nb_hidden_fc_layers=2,
                                         nb_hidden_neurons=100,
                                         nb_cos_embeddings=64,
                                         network_seed=13,
                                         policy=greedy_policy, test_policy=test_policy,
                                         enable_double_dqn=True,
                                         enable_dueling_dqn=True,
                                         nb_samples_policy=32,
                                         nb_sampled_quantiles=32,
                                         cvar_eta=1,
                                         gamma=0.99, batch_size=64,
                                         nb_steps_warmup=1000,
                                         train_interval=1,
                                         memory_interval=1,
                                         window_length=1,
                                         target_model_update=1000,
                                         delta_clip=10,
                                         prior_scale_factor=1)

    def test_sample_tau_values(self):
        tau = self.agent.sample_tau_values(max_tau=1)
        self.assertEqual(tau.size, 32)
        self.assertTrue((tau < 1).all)
        self.assertTrue((tau > 0).all)

        # Risk sensitive policy
        state = [np.random.rand(10)]
        z_values, tau = self.agent.compute_sampled_z_values(state, max_tau=0.25, net=0)
        self.assertEqual(tau.size, 32)
        self.assertTrue((tau < 0.25).all)
        self.assertTrue((tau > 0).all)

        # Uniform sampling
        max_tau = 0.5
        tau = self.agent.sample_tau_values(max_tau=max_tau, uniform=True)
        self.assertEqual(tau[0, 0, 0], 0)
        self.assertEqual(tau[0, 0, -1], max_tau)
        d_tau = np.diff(tau)
        self.assertTrue(np.isclose(d_tau, d_tau[0, 0, 0]).all())

        [worker.terminate() for worker in self.agent.workers]

    def test_compute_sampled_z_values(self):
        state = [np.random.rand(10)]
        z_values, tau = self.agent.compute_sampled_z_values(state, max_tau=1, net=np.random.randint(self.nb_nets))
        self.assertEqual(z_values.shape, (1, self.agent.nb_sampled_quantiles, self.agent.nb_actions))
        self.assertEqual(tau.shape, (1, 1, 32))
        d_tau = np.diff(tau)
        self.assertFalse(np.isclose(d_tau, d_tau[0, 0, 0]).all())   # Should use random sampling

        [worker.terminate() for worker in self.agent.workers]

    def test_compute_z_values_all_nets(self):
        state = [np.random.rand(10)]
        z_values_all_nets, tau = self.agent.compute_z_values_all_nets(state, max_tau=1)
        self.assertEqual(z_values_all_nets.shape, (self.nb_nets, self.agent.nb_sampled_quantiles, self.agent.nb_actions))
        self.assertEqual(tau.shape, (1, 1, 32))
        d_tau = np.diff(tau)
        self.assertTrue(np.isclose(d_tau, d_tau[0, 0, 0]).all())  # Should use uniform sampling

        [worker.terminate() for worker in self.agent.workers]

    def test_forward(self):
        observation = np.random.rand(10)
        self.agent.training = True
        action, info = self.agent.forward(observation)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 4)
        d_tau = np.diff(info['quantiles'])
        self.assertFalse(np.isclose(d_tau, d_tau[0]).all())  # Should use random sampling

        self.agent.training = False
        action, info = self.agent.forward(observation)
        self.assertGreaterEqual(action, 0)
        self.assertLess(action, 4)
        d_tau = np.diff(info['quantiles'])
        self.assertTrue(np.isclose(d_tau, d_tau[0, 0, 0]).all())  # Should use uniform sampling

        [worker.terminate() for worker in self.agent.workers]

    def test_backward(self):
        self.agent.training = True
        for _ in range(0, 1000):  # Fill up buffer
            observation = np.random.rand(10)
            self.agent.forward(observation)
            self.agent.backward(0, terminal=False)
        self.agent.step = 1001   # Start training
        observation = np.random.rand(10)
        self.agent.forward(observation)
        self.agent.backward(0, terminal=False)
        for i in range(10):
            observation = np.random.rand(10)
            self.agent.forward(observation)
            self.agent.backward(0, terminal=False)

        [worker.terminate() for worker in self.agent.workers]

    def test_training(self):
        self.agent.training = True
        obs1 = np.random.rand(10)
        obs2 = np.random.rand(10)
        for _ in range(0, 500):  # Fill up buffer
            self.agent.forward(obs1)
            self.agent.backward(0, terminal=False)
        for _ in range(0, 500):  # Fill up buffer
            self.agent.forward(obs2)
            self.agent.backward(10, terminal=False)
        self.agent.step = 1001   # Start training
        for _ in range(30):
            for _ in range(10):
                action1, action_info1 = self.agent.forward(obs1)
                self.agent.backward(0, terminal=False)
            for _ in range(10):
                action2, action_info2 = self.agent.forward(obs2)
                self.agent.backward(10, terminal=False)
        # Since actions are picked greedily, only one Q-value is expected to converge to the correct value
        self.assertTrue((np.round(np.mean(action_info1['z_values'], axis=0)) == 0).any())
        self.assertTrue((np.round(np.mean(action_info2['z_values'], axis=0)) == 10).any())

        # All nets should be trained
        for i in range(self.nb_nets):
            self.agent.active_model = i
            action1, action_info1 = self.agent.forward(obs1)
            action2, action_info2 = self.agent.forward(obs2)
            self.assertTrue((np.round(np.mean(action_info1['z_values'], axis=0)) == 0).any())
            self.assertTrue((np.round(np.mean(action_info2['z_values'], axis=0)) == 10).any())

        [worker.terminate() for worker in self.agent.workers]

    def test_no_double_dqn(self):
        self.agent.enable_double_dqn = False
        self.agent.training = True
        for _ in range(0, 1000):  # Fill up buffer
            observation = np.random.rand(10)
            self.agent.forward(observation)
            self.agent.backward(0, terminal=False)
        self.agent.step = 1001  # Start training
        observation = np.random.rand(10)
        self.agent.forward(observation)
        self.agent.backward(0, terminal=False)
        for i in range(10):
            observation = np.random.rand(10)
            self.agent.forward(observation)
            self.agent.backward(0, terminal=False)
        self.agent.enable_double_dqn = True

        [worker.terminate() for worker in self.agent.workers]

    def test_get_config(self):
        self.agent.get_config()

        [worker.terminate() for worker in self.agent.workers]


if __name__ == '__main__':
    unittest.main()
