import unittest
import numpy as np
from keras.optimizers import Adam
from keras.utils import plot_model
from rl.policy import LinearAnnealedPolicy
from rl.memory import SequentialMemory

import sys
sys.path.append('../src')
from iqn import IQNAgent
from network_architecture_distributional import NetworkMLPDistributional
from policy import DistributionalEpsGreedyPolicy


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.model = NetworkMLPDistributional(nb_inputs=10, nb_outputs=4, nb_hidden_layers=2,
                                              nb_hidden_neurons=100, nb_quantiles=32,
                                              nb_cos_embeddings=64, duel=True,
                                              prior=False, activation='relu', duel_type='avg',
                                              window_length=1).model
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
                              gamma=0.99, batch_size=32,
                              nb_steps_warmup=1000,
                              train_interval=1,
                              memory_interval=1,
                              target_model_update=1000,
                              delta_clip=10)

    def test_quantile_regression(self):
        nb_inputs = 10
        nb_actions = 3
        nb_quantiles = 32
        batch_size = 64
        delta_clip = 1
        model = NetworkMLPDistributional(nb_inputs=nb_inputs, nb_outputs=nb_actions, nb_hidden_layers=2,
                                         nb_hidden_neurons=100, nb_quantiles=nb_quantiles,
                                         nb_cos_embeddings=64, duel=True,
                                         prior=False, activation='relu', duel_type='avg',
                                         window_length=1).model
        policy = LinearAnnealedPolicy(DistributionalEpsGreedyPolicy(eps=1), attr='eps', value_max=1.,
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
                         gamma=0.99, batch_size=batch_size,
                         nb_steps_warmup=1000,
                         train_interval=1,
                         memory_interval=1,
                         target_model_update=1000,
                         delta_clip=delta_clip)

        agent.compile(Adam(lr=0.0001))
        plot_model(agent.trainable_model, to_file='trainable_model_2.png', show_shapes=True)

        # Test input
        states = np.random.rand(batch_size, 1, nb_inputs)
        actions = np.random.randint(nb_actions, size=batch_size)
        test_quantiles = np.linspace(0, 1, nb_quantiles)
        z_values = agent.model.predict_on_batch([states, test_quantiles[None, None, :]])
        # print(z_values[0])

        for i in range(3000):
            quantiles = np.random.rand(batch_size, 1, nb_quantiles)
            # targets = np.random.choice([1, 2, 3], batch_size)
            targets = np.random.choice([10, 22, 35], batch_size)
            targets = np.repeat(targets[:, None], nb_quantiles, axis=1)

            predictions = agent.model.predict_on_batch([states, quantiles])

            masks = np.zeros((batch_size, nb_actions))
            masks[range(batch_size), actions] = 1
            targets_expanded = np.zeros((batch_size, nb_quantiles, nb_actions))
            targets_expanded[range(batch_size), :, actions] = targets[range(batch_size), :]

            loss = agent.trainable_model.predict_on_batch([states, quantiles, targets_expanded, masks])

            metrics = agent.trainable_model.train_on_batch([states, quantiles, targets_expanded, masks],
                                                           [targets, targets_expanded])

            if np.mod(i, 100) == 0:
                test_quantiles = np.linspace(0, 1, nb_quantiles)
                z_values = agent.model.predict_on_batch([states, test_quantiles[None, None, :]])

        self.assertTrue(np.abs(np.mean(z_values[:, 1:10, :]) - 10) < 1.0)
        self.assertTrue(np.abs(np.mean(z_values[:, 12:20, :]) - 22) < 1.0)
        self.assertTrue(np.abs(np.mean(z_values[:, 23:31, :]) - 35) < 1.0)


if __name__ == '__main__':
    unittest.main()
