import unittest
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy, GreedyQPolicy
from rl.memory import SequentialMemory

import sys
sys.path.append('../src')
from iqn import IQNAgent
from network_architecture_distributional import NetworkMLPDistributional, NetworkCNNDistributional
from policy import DistributionalEpsGreedyPolicy


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.model = NetworkCNNDistributional(nb_ego_states=1, nb_states_per_vehicle=3, nb_vehicles=3, nb_actions=4,
                                              nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                              nb_hidden_neurons=100, duel=True, prior=False,
                                              nb_quantiles=32, nb_cos_embeddings=64).model
        self.policy = LinearAnnealedPolicy(DistributionalEpsGreedyPolicy(eps=None), attr='eps', value_max=1.,
                                           value_min=0.1, value_test=.0,
                                           nb_steps=10000)
        self.test_policy = DistributionalEpsGreedyPolicy(eps=0)
        self.memory = SequentialMemory(limit=10000, window_length=1)
        self.agent = IQNAgent(model=self.model, policy=self.policy, test_policy=self.test_policy,
                              enable_double_dqn=True,
                              nb_samples_policy=32,
                              nb_sampled_quantiles=32,
                              cvar_eta=1,
                              nb_actions=4, memory=self.memory,
                              gamma=0.99, batch_size=64,
                              nb_steps_warmup=1000,
                              train_interval=1,
                              memory_interval=1,
                              target_model_update=1000,
                              delta_clip=10)

    def test_sample_tau_values(self):
        tau = self.agent.sample_tau_values(max_tau=1)
        self.assertEqual(tau.size, 32)
        self.assertTrue((tau < 1).all)
        self.assertTrue((tau > 0).all)

        # Risk sensitive policy
        state = [np.random.rand(10)]
        z_values, tau = self.agent.compute_sampled_z_values(state, max_tau=0.25)
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

    def test_compute_sampled_z_values(self):
        self.agent.training = True
        state = [np.random.rand(10)]
        z_values, tau = self.agent.compute_sampled_z_values(state, max_tau=1)
        self.assertEqual(z_values.shape, (1, self.agent.nb_sampled_quantiles, self.agent.nb_actions))
        self.assertEqual(tau.shape, (1, 1, 32))
        d_tau = np.diff(tau)
        self.assertFalse(np.isclose(d_tau, d_tau[0, 0, 0]).all())  # Should use random sampling

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
        self.assertTrue(np.isclose(d_tau, d_tau[0]).all())  # Should use uniform sampling

    def test_compile(self):
        self.agent.compile(Adam(lr=0.01))

    def test_backward(self):
        self.agent.compile(Adam(lr=0.01))
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

        plot_model(self.agent.trainable_model, to_file='trainable_model.png', show_shapes=True)

    def test_training(self):
        self.agent.compile(Adam(lr=0.001))
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
        self.assertTrue((np.round(np.mean(action_info1['z_values'], axis=0)) == 0).all())
        self.assertTrue((np.round(np.mean(action_info2['z_values'], axis=0)) == 10).all())

    def test_trainable_model(self):
        nb_inputs = 10
        nb_actions = 5
        nb_quantiles = 32
        batch_size = 64
        delta_clip = 1
        model = NetworkMLPDistributional(nb_inputs=nb_inputs, nb_outputs=nb_actions, nb_hidden_layers=2,
                                         nb_hidden_neurons=100, nb_quantiles=nb_quantiles,
                                         nb_cos_embeddings=64, duel=True,
                                         prior=False, activation='relu', duel_type='avg',
                                         window_length=1).model
        policy = LinearAnnealedPolicy(DistributionalEpsGreedyPolicy(eps=None), attr='eps', value_max=1.,
                                      value_min=0.1, value_test=.0,
                                      nb_steps=10000)
        test_policy = DistributionalEpsGreedyPolicy(eps=0)
        memory = SequentialMemory(limit=10000, window_length=1)
        agent = IQNAgent(model=model, policy=policy, test_policy=test_policy,
                         enable_double_dqn=True,
                         nb_samples_policy=nb_quantiles,
                         nb_sampled_quantiles=nb_quantiles,
                         cvar_eta=1,
                         nb_actions=nb_actions, memory=memory,
                         gamma=0.99, batch_size=1,
                         nb_steps_warmup=1000,
                         train_interval=1,
                         memory_interval=1,
                         target_model_update=1000,
                         delta_clip=delta_clip)

        agent.compile(Adam(lr=0.01))
        plot_model(agent.trainable_model, to_file='trainable_model_2.png', show_shapes=True)

        # Test input
        states = np.random.rand(batch_size, 1, nb_inputs)
        actions = np.random.randint(nb_actions, size=batch_size)
        quantiles = np.random.rand(batch_size, 1, nb_quantiles)
        targets = np.random.rand(batch_size, nb_quantiles)

        predictions = agent.model.predict_on_batch([states, quantiles])

        def huber(deltas, quantile):
            if np.abs(deltas) < delta_clip:
                loss = 0.5 * deltas ** 2
            else:
                loss = delta_clip * (np.abs(deltas) - 0.5 * delta_clip)
            if deltas > 0:
                loss *= quantile / delta_clip
            else:
                loss *= (1 - quantile) / delta_clip
            if loss < 0:
                raise Exception("Loss should always be positive")
            return loss

        true_loss = np.zeros(batch_size)
        for idx in range(batch_size):
            for i in range(nb_quantiles):
                for j in range(nb_quantiles):
                    true_loss[idx] += huber(targets[idx, j] - predictions[idx, i, actions[idx]],
                                            quantiles[idx, 0, i])
            true_loss[idx] *= 1 / nb_quantiles

        masks = np.zeros((batch_size, nb_actions))
        masks[range(batch_size), actions] = 1
        targets_expanded = np.zeros((batch_size, nb_quantiles, nb_actions))
        targets_expanded[range(batch_size), :, actions] = targets[range(batch_size), :]
        out = agent.trainable_model.predict_on_batch([states, quantiles, targets_expanded, masks])

        self.assertTrue(np.isclose(true_loss, out[0]).all())
        self.assertTrue((predictions == out[1]).all())

        metrics = agent.trainable_model.train_on_batch([states, quantiles, targets_expanded, masks],
                                                       [targets, targets_expanded])
        self.assertTrue(np.isclose(np.mean(true_loss), metrics[0]))

        average_q_value = np.mean(predictions)
        average_max_q_value = np.mean(np.max(np.mean(predictions, axis=1), axis=-1))
        self.assertTrue(np.isclose(average_q_value, metrics[3]))
        self.assertTrue(np.isclose(average_max_q_value, metrics[4]))

    def test_no_double_dqn(self):
        self.agent.compile(Adam(lr=0.01))
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

    def test_get_config(self):
        self.agent.get_config()


if __name__ == '__main__':
    unittest.main()
