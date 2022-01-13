import unittest
import numpy as np
from keras.optimizers import Adam
from rl.policy import EpsGreedyQPolicy, LinearAnnealedPolicy
from rl.memory import SequentialMemory

import sys
sys.path.append('../src')
from iqn import IQNAgent
from network_architecture_distributional import NetworkMLPDistributional
from policy import DistributionalEpsGreedyPolicy
from network_architecture import NetworkMLP
from dqn_standard import DQNAgent


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        self.agent_name = 'iqn'
        self.verbose = False

        if self.agent_name == 'iqn':
            self.nb_quantiles = 32
            self.model = NetworkMLPDistributional(nb_inputs=10, nb_outputs=4, nb_hidden_layers=2,
                                                  nb_hidden_neurons=100, nb_quantiles=self.nb_quantiles,
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
                                  nb_samples_policy=self.nb_quantiles,
                                  nb_sampled_quantiles=self.nb_quantiles,
                                  cvar_eta=1,
                                  nb_actions=4, memory=self.memory,
                                  gamma=0.99, batch_size=48,
                                  nb_steps_warmup=1000,
                                  train_interval=1,
                                  memory_interval=1,
                                  target_model_update=1000,
                                  delta_clip=1)
        elif self.agent_name == 'dqn':
            self.model = NetworkMLP(nb_inputs=10, nb_outputs=4, nb_hidden_layers=2,
                                    nb_hidden_neurons=100, duel=True,
                                    prior=False, activation='relu', duel_type='avg',
                                    window_length=1).model
            self.policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1.,
                                               value_min=0.1, value_test=.0,
                                               nb_steps=10000)
            self.test_policy = EpsGreedyQPolicy(eps=0)
            self.memory = SequentialMemory(limit=10000, window_length=1)
            self.agent = DQNAgent(model=self.model, policy=self.policy, test_policy=self.test_policy,
                                  enable_double_dqn=True,
                                  nb_actions=4, memory=self.memory,
                                  gamma=0.99, batch_size=48,
                                  nb_steps_warmup=1000,
                                  train_interval=1,
                                  memory_interval=1,
                                  target_model_update=1000,
                                  delta_clip=1)

    def test_simple_iqn_training(self):
        self.agent.compile(Adam(lr=0.01))
        self.agent.training = True
        # obs = np.zeros(10)
        obs = np.ones(10)
        for _ in range(0, 500):  # Fill up buffer
            self.agent.forward(obs)
            self.agent.backward(10, terminal=True)
            self.agent.forward(obs)
            self.agent.backward(10, terminal=False)
        for _ in range(0, 500):  # Fill up buffer
            self.agent.forward(obs)
            self.agent.backward(0, terminal=True)
            self.agent.forward(obs)
            self.agent.backward(0, terminal=False)
        self.agent.step = 1001   # Start training
        for b in range(10):
            for _ in range(100):
                action1, action_info1 = self.agent.forward(obs)
                metrics = self.agent.backward(10, terminal=True)
                action1, action_info1 = self.agent.forward(obs)
                metrics = self.agent.backward(10, terminal=False)
                self.agent.step += 1

                action2, action_info2 = self.agent.forward(obs)
                metrics = self.agent.backward(0, terminal=True)
                action2, action_info2 = self.agent.forward(obs)
                metrics = self.agent.backward(0, terminal=False)
                self.agent.step += 1
            if self.verbose:
                print(b)
            if self.agent_name == 'iqn':
                test_quantiles = np.linspace(0, 1, self.nb_quantiles)
                z_values = self.agent.model.predict([obs[None, None, :], test_quantiles[None, None, :]])
                if self.verbose:
                    print(z_values)
            elif self.agent_name == 'dqn':
                q_values = self.agent.model.predict(obs[None, None, :])
                if self.verbose:
                    print(q_values)

        if self.agent_name == 'iqn':
            test_quantiles = np.linspace(0, 1, self.nb_quantiles)
            z_values = self.agent.model.predict([obs[None, None, :], test_quantiles[None, None, :]])
            if self.verbose:
                print(z_values)
                print(np.abs(np.mean(z_values[:, :15, :]) - 0) < 0.5)
                print(np.abs(np.mean(z_values[:, 17:, :]) - 10) < 0.5)

            self.assertTrue(np.abs(np.mean(z_values[:, :15, :]) - 0) < 0.5)
            self.assertTrue(np.abs(np.mean(z_values[:, 17:, :]) - 10) < 1.0)

        elif self.agent_name == 'dqn':
            q_values = self.agent.model.predict(obs[None, None, :])
            if self.verbose:
                print(q_values)


if __name__ == '__main__':
    unittest.main()
