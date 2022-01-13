import unittest
import numpy as np
import copy
import keras as K
from keras.utils import plot_model
from rl.util import clone_model

import sys
sys.path.append('../src')
from network_architecture_distributional import NetworkMLPDistributional, NetworkCNNDistributional


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

    def test_mlp_distributional_network(self):
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=False, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 1, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        # Cos embedding
        state = np.random.rand(1, 1, 3)
        tau = np.zeros((1, 1, 8))
        tau[0, 0, 0] = 0
        tau[0, 0, 1] = 1/64
        tau[0, 0, 2] = 0.5
        net_input = [state, tau]
        cos_embedding_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('cos_tau').output)
        cos_embedding_output = cos_embedding_layer.predict(net_input)
        self.assertTrue((cos_embedding_output[0, 0, :] == 1).all())
        self.assertTrue(all([np.isclose(cos_embedding_output[0, 1, i], np.cos(np.pi*i*1/64), atol=1e-7)
                             for i in range(cos_embedding_output.shape[2])]))

        # Merge
        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        tau_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_tau').output)
        state_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_state_extra_dim').output)
        merge_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merge').output)
        tau_net_output = tau_net_layer.predict(net_input)
        state_net_output = state_net_layer.predict(net_input)
        merge_output = merge_layer.predict(net_input)
        self.assertTrue(np.isclose(tau_net_output[None, :, 0, :] * state_net_output, merge_output[:, 0, :]).all())

        plot_model(net.model, to_file='mlp_distributional.png', show_shapes=True)

        # Test clone model, mainly to see that no custom objects are missing
        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        target_model = clone_model(net.model)
        target_model.compile(optimizer='sgd', loss='mse')
        out = net.model.predict(net_input)
        out_clone = target_model.predict(net_input)
        self.assertTrue((out == out_clone).all())

        # Window length > 1
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=False, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64, window_length=5)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 5, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        state = np.random.rand(1, 5, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, 0, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(
            np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())  # Equal values of tau should give equal values of Z_tau

        plot_model(net.model, to_file='mlp_window_5_distributional.png', show_shapes=True)

    def test_mlp_dueling_network(self):
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=True, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 1, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        before_dueling_layer = K.Model(inputs=net.model.inputs, outputs=net.model.layers[-2].output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        true_output = before_dueling_output[:, :, 0, None] + before_dueling_output[:, :, 1:] - \
                      np.mean(before_dueling_output[:, :, 1:, None], axis=2)
        self.assertTrue(np.isclose(out, true_output).all())

        single_input = [net_input[0][None, 0], net_input[1][None, 0]]
        self.assertTrue(np.isclose(out[0], net.model.predict(single_input)).all())

        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 7):
            tau[:, :, i+1] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        plot_model(net.model, to_file='mlp_duel_distributional.png', show_shapes=True)

    def test_mlp_distributional_network_with_prior(self):
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=False, prior=True,
                                       prior_scale_factor=1, nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        for layer in net.model.layers:
            if 'prior' in layer.name and not not layer.weights:
                self.assertFalse(layer.trainable)

        state = np.random.rand(32, 1, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        # Cos embedding
        state = np.random.rand(1, 1, 3)
        tau = np.zeros((1, 1, 8))
        tau[0, 0, 0] = 0
        tau[0, 0, 1] = 1/64
        tau[0, 0, 2] = 0.5
        net_input = [state, tau]
        cos_embedding_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('cos_tau').output)
        cos_embedding_output = cos_embedding_layer.predict(net_input)
        self.assertTrue((cos_embedding_output[0, 0, :] == 1).all())
        self.assertTrue(all([np.isclose(cos_embedding_output[0, 1, i], np.cos(np.pi*i*1/64), atol=1e-7)
                             for i in range(cos_embedding_output.shape[2])]))

        # Merge
        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        tau_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_tau_trainable').output)
        state_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_state_extra_dim_trainable').output)
        merge_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merge_trainable').output)
        tau_net_output = tau_net_layer.predict(net_input)
        state_net_output = state_net_layer.predict(net_input)
        merge_output = merge_layer.predict(net_input)
        self.assertTrue(np.isclose(tau_net_output[None, :, 0, :] * state_net_output, merge_output[:, 0, :]).all())

        plot_model(net.model, to_file='mlp_distributional_with_prior.png', show_shapes=True)

        # Test clone model, mainly to see that no custom objects are missing
        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        target_model = clone_model(net.model)
        target_model.compile(optimizer='sgd', loss='mse')
        out = net.model.predict(net_input)
        out_clone = target_model.predict(net_input)
        self.assertTrue((out == out_clone).all())

        # Window length > 1
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=False, prior=True,
                                       prior_scale_factor=1, nb_quantiles=8, nb_cos_embeddings=64, window_length=5)
        self.assertTrue(net.model.trainable)
        for layer in net.model.layers:
            if 'prior' in layer.name and not not layer.weights:
                self.assertFalse(layer.trainable)

        state = np.random.rand(32, 5, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        state = np.random.rand(1, 5, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, 0, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(
            np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())  # Equal values of tau should give equal values of Z_tau

        plot_model(net.model, to_file='mlp_window_5_distributional_with_prior.png', show_shapes=True)

    def test_mlp_duel_distributional_with_prior_network(self):
        net = NetworkMLPDistributional(3, 4, nb_hidden_layers=2, nb_hidden_neurons=100, duel=True, prior=True,
                                       prior_scale_factor=1, nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        for layer in net.model.layers:
            if 'prior' in layer.name and not not layer.weights:
                self.assertFalse(layer.trainable)

        state = np.random.rand(32, 1, 3)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 4))

        net.model.get_config()  # This crashes for custom lambda layers

        before_dueling_layer = K.Model(inputs=net.model.inputs,
                                       outputs=net.model.get_layer('output_trainable_wo_dueling').output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        after_dueling_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('output_trainable').output)
        after_dueling_output = after_dueling_layer.predict(net_input)
        true_dueling_output = before_dueling_output[:, :, 0, None] + before_dueling_output[:, :, 1:] - \
                              np.mean(before_dueling_output[:, :, 1:, None], axis=2)
        self.assertTrue((after_dueling_output == true_dueling_output).all())

        single_input = [net_input[0][None, 0], net_input[1][None, 0]]
        self.assertTrue(np.isclose(out[0], net.model.predict(single_input)).all())

        state = np.random.rand(1, 1, 3)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 7):
            tau[:, :, i+1] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        plot_model(net.model, to_file='mlp_duel_distributional_with_prior.png', show_shapes=True)

    def test_cnn_distributional_network(self):
        net = NetworkCNNDistributional(nb_ego_states=7, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=100, duel=False, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 1, 47)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 9))

        state1 = np.random.rand(1, 1, 47)
        tau = np.random.rand(32, 1, 8)
        input1 = [state1, tau]
        out1 = net.model.predict(input1)
        input2 = [np.copy(state1), np.copy(tau)]
        input2[0][0, 0, 7:15] = input1[0][0, 0, 15:23]
        input2[0][0, 0, 15:23] = input1[0][0, 0, 7:15]
        self.assertFalse((input1[0] == input2[0]).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        # Cos embedding
        state = np.random.rand(1, 1, 47)
        tau = np.zeros((1, 1, 8))
        tau[0, 0, 0] = 0
        tau[0, 0, 1] = 1/64
        tau[0, 0, 2] = 0.5
        net_input = [state, tau]
        cos_embedding_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('cos_tau').output)
        cos_embedding_output = cos_embedding_layer.predict(net_input)
        self.assertTrue((cos_embedding_output[0, 0, :] == 1).all())
        self.assertTrue(all([np.isclose(cos_embedding_output[0, 1, i], np.cos(np.pi*i*1/64), atol=1e-7)
                             for i in range(cos_embedding_output.shape[2])]))

        # Merge
        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        tau_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_tau').output)
        state_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merged_extra_dim').output)
        merge_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merge').output)
        tau_net_output = tau_net_layer.predict(net_input)
        state_net_output = state_net_layer.predict(net_input)
        merge_output = merge_layer.predict(net_input)
        self.assertTrue(np.isclose(tau_net_output[None, :, 0, :] * state_net_output, merge_output[:, 0, :]).all())

        plot_model(net.model, to_file='cnn_distributional.png', show_shapes=True)

        # Test clone model, mainly to see that no custom objects are missing
        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        target_model = clone_model(net.model)
        target_model.compile(optimizer='sgd', loss='mse')
        out = net.model.predict(net_input)
        out_clone = target_model.predict(net_input)
        self.assertTrue((out == out_clone).all())

        # Window length > 1
        net = NetworkCNNDistributional(nb_ego_states=7, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=100, duel=False, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64, window_length=5)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 5, 47)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 9))

        state1 = np.random.rand(1, 5, 47)
        tau = np.random.rand(32, 1, 8)
        input1 = [state1, tau]
        out1 = net.model.predict(input1)
        input2 = [np.copy(state1), np.copy(tau)]
        input2[0][0, :, 7:15] = input1[0][0, :, 15:23]
        input2[0][0, :, 15:23] = input1[0][0, :, 7:15]
        self.assertFalse((input1[0] == input2[0]).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        state = np.random.rand(1, 5, 47)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, 0, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())  # Equal values of tau should give equal values of Z_tau

        plot_model(net.model, to_file='cnn_window_5_distributional.png', show_shapes=True)

    def test_cnn_dueling_distributional_network(self):
        net = NetworkCNNDistributional(nb_ego_states=7, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=100, duel=True, prior=False,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 1, 47)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 9))

        before_dueling_layer = K.Model(inputs=net.model.inputs, outputs=net.model.layers[-2].output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        true_output = before_dueling_output[:, :, 0, None] + before_dueling_output[:, :, 1:] - \
                      np.mean(before_dueling_output[:, :, 1:, None], axis=2)
        self.assertTrue(np.isclose(out, true_output).all())

        single_input = [net_input[0][None, 0], net_input[1][None, 0]]
        self.assertTrue(np.isclose(out[0], net.model.predict(single_input)).all())

        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 7):
            tau[:, :, i+1] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        plot_model(net.model, to_file='cnn_duel_distributional.png', show_shapes=True)

    def test_cnn_dueling_distributional_with_prior(self):
        net = NetworkCNNDistributional(nb_ego_states=7, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=100, duel=True, prior=True, prior_scale_factor=1,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 1, 47)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 9))

        state1 = np.random.rand(1, 1, 47)
        tau = np.random.rand(32, 1, 8)
        input1 = [state1, tau]
        out1 = net.model.predict(input1)
        input2 = [np.copy(state1), np.copy(tau)]
        input2[0][0, 0, 7:15] = input1[0][0, 0, 15:23]
        input2[0][0, 0, 15:23] = input1[0][0, 0, 7:15]
        self.assertFalse((input1[0] == input2[0]).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, :, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())   # Equal values of tau -> equal Z_tau

        # Cos embedding
        state = np.random.rand(1, 1, 47)
        tau = np.zeros((1, 1, 8))
        tau[0, 0, 0] = 0
        tau[0, 0, 1] = 1/64
        tau[0, 0, 2] = 0.5
        net_input = [state, tau]
        cos_embedding_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('cos_tau').output)
        cos_embedding_output = cos_embedding_layer.predict(net_input)
        self.assertTrue((cos_embedding_output[0, 0, :] == 1).all())
        self.assertTrue(all([np.isclose(cos_embedding_output[0, 1, i], np.cos(np.pi*i*1/64), atol=1e-7)
                             for i in range(cos_embedding_output.shape[2])]))

        # Merge
        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        tau_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('fc_tau_trainable').output)
        state_net_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merged_extra_dim_trainable').output)
        merge_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('merge_trainable').output)
        tau_net_output = tau_net_layer.predict(net_input)
        state_net_output = state_net_layer.predict(net_input)
        merge_output = merge_layer.predict(net_input)
        self.assertTrue(np.isclose(tau_net_output[None, :, 0, :] * state_net_output, merge_output[:, 0, :]).all())

        plot_model(net.model, to_file='cnn_duel_distributional_with_prior.png', show_shapes=True)

        # Test clone model, mainly to see that no custom objects are missing
        state = np.random.rand(1, 1, 47)
        tau = np.random.rand(1, 1, 8)
        net_input = [state, tau]
        target_model = clone_model(net.model)
        target_model.compile(optimizer='sgd', loss='mse')
        out = net.model.predict(net_input)
        out_clone = target_model.predict(net_input)
        self.assertTrue((out == out_clone).all())

        # Prior nets not trainable
        self.assertTrue(net.model.trainable)
        for layer in net.model.layers:
            if 'prior' in layer.name and not not layer.weights:
                self.assertFalse(layer.trainable)

        net.model.get_config()  # This crashes for custom lambda layers

        before_dueling_layer = K.Model(inputs=net.model.inputs,
                                       outputs=net.model.get_layer('output_trainable_wo_dueling').output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        after_dueling_layer = K.Model(inputs=net.model.inputs, outputs=net.model.get_layer('output_trainable').output)
        after_dueling_output = after_dueling_layer.predict(net_input)
        true_dueling_output = before_dueling_output[:, :, 0, None] + before_dueling_output[:, :, 1:] - \
                              np.mean(before_dueling_output[:, :, 1:, None], axis=2)
        self.assertTrue(np.isclose(after_dueling_output, true_dueling_output).all())

        single_input = [net_input[0][None, 0], net_input[1][None, 0]]
        self.assertTrue(np.isclose(out[0], net.model.predict(single_input)).all())

        # Window length > 1
        net = NetworkCNNDistributional(nb_ego_states=7, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=100, duel=True, prior=True, prior_scale_factor=1,
                                       nb_quantiles=8, nb_cos_embeddings=64, window_length=5)
        self.assertTrue(net.model.trainable)
        state = np.random.rand(32, 5, 47)
        tau = np.random.rand(32, 1, 8)
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 8, 9))

        state1 = np.random.rand(1, 5, 47)
        tau = np.random.rand(32, 1, 8)
        input1 = [state1, tau]
        out1 = net.model.predict(input1)
        input2 = [np.copy(state1), np.copy(tau)]
        input2[0][0, :, 7:15] = input1[0][0, :, 15:23]
        input2[0][0, :, 15:23] = input1[0][0, :, 7:15]
        self.assertFalse((input1[0] == input2[0]).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        state = np.random.rand(1, 5, 47)
        tau = np.random.rand(1, 1, 8)
        for i in range(0, 8):
            tau[:, :, i] = tau[:, 0, 0]
        net_input = [state, tau]
        out = net.model.predict(net_input)
        self.assertTrue(np.abs((out[0, 0, :] - out[0, :, :]) < 1e-6).all())  # Equal values of tau should give equal values of Z_tau

        plot_model(net.model, to_file='cnn_window_5_duel_distributional_with_prior.png', show_shapes=True)

    def test_trainable_weights_mlp_prior(self):
        net = NetworkMLPDistributional(5, 3,  nb_hidden_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                                       prior_scale_factor=1, nb_quantiles=8, nb_cos_embeddings=64)
        net.model.compile(loss='mse', optimizer='adam')
        x = np.random.rand(100, 1, 5)
        tau = np.random.rand(100, 1, 8)
        y = np.random.rand(100, 8, 3)
        net_input = [x, tau]
        initial_model = copy.deepcopy(net.model)
        net.model.fit(net_input, y, epochs=10, batch_size=100, verbose=0)
        for layer_init, layer in zip(initial_model.layers, net.model.layers):
            if not layer.trainable:
                init_weights = layer_init.get_weights()
                weights = layer.get_weights()
                for row_init, row in zip(init_weights, weights):
                    tmp = row_init == row
                    self.assertTrue(tmp.all())
        get_prior_output_initial = K.backend.function([initial_model.get_layer('state_input').input, initial_model.get_layer('tau_input').input],
                                                      initial_model.get_layer('output_prior').output)
        prior_out_initial = get_prior_output_initial([x[0, :, :], tau[0, :, :]])[0]
        get_prior_output = K.backend.function([net.model.get_layer('state_input').input, net.model.get_layer('tau_input').input],
                                               net.model.get_layer('output_prior').output)
        prior_out = get_prior_output([x[0, :, :], tau[0, :, :]])[0]
        self.assertTrue((prior_out_initial == prior_out).all())
        get_trainable_output_initial = K.backend.function([initial_model.get_layer('state_input').input, initial_model.get_layer('tau_input').input],
                                                          initial_model.get_layer('output_trainable').output)
        trainable_out_initial = get_trainable_output_initial([x[0, :, :], tau[0, :, :]])[0]
        get_trainable_output = K.backend.function([net.model.get_layer('state_input').input, net.model.get_layer('tau_input').input],
                                                  net.model.get_layer('output_trainable').output)
        trainable_out = get_trainable_output([x[0, :, :], tau[0, :, :]])[0]
        self.assertTrue((trainable_out_initial != trainable_out).all())

    def test_trainable_weights_cnn_prior(self):
        net = NetworkCNNDistributional(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                       nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                       nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1,
                                       nb_quantiles=8, nb_cos_embeddings=64)
        net.model.compile(loss='mse', optimizer='adam')
        x = np.random.rand(100, 1, 44)
        tau = np.random.rand(100, 1, 8)
        y = np.random.rand(100, 8, 9)
        net_input = [x, tau]
        initial_model = copy.deepcopy(net.model)
        net.model.fit(net_input, y, epochs=10, batch_size=100, verbose=0)
        for layer_init, layer in zip(initial_model.layers, net.model.layers):
            if not layer.trainable:
                init_weights = layer_init.get_weights()
                weights = layer.get_weights()
                for row_init, row in zip(init_weights, weights):
                    tmp = row_init == row
                    self.assertTrue(tmp.all())
        get_prior_output_initial = K.backend.function([initial_model.get_layer('state_input').input, initial_model.get_layer('tau_input').input],
                                                      initial_model.get_layer('output_prior').output)
        prior_out_initial = get_prior_output_initial([x[None, 0, :, :], tau[None, 0, :, :]])[0]
        get_prior_output = K.backend.function([net.model.get_layer('state_input').input, net.model.get_layer('tau_input').input],
                                              net.model.get_layer('output_prior').output)
        prior_out = get_prior_output([x[None, 0, :, :], tau[None, 0, :, :]])[0]
        self.assertTrue((prior_out_initial == prior_out).all())
        get_trainable_output_initial = K.backend.function([initial_model.get_layer('state_input').input, initial_model.get_layer('tau_input').input],
                                                           initial_model.get_layer('output_trainable').output)
        trainable_out_initial = get_trainable_output_initial([x[None, 0, :, :], tau[None, 0, :, :]])[0]
        get_trainable_output = K.backend.function([net.model.get_layer('state_input').input, net.model.get_layer('tau_input').input],
                                                  net.model.get_layer('output_trainable').output)
        trainable_out = get_trainable_output([x[None, 0, :, :], tau[None, 0, :, :]])[0]
        self.assertTrue((trainable_out_initial != trainable_out).all())

    def test_random_initialization(self):
        net1 = NetworkCNNDistributional(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                        nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                        nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1,
                                        nb_quantiles=8, nb_cos_embeddings=64)
        net2 = NetworkCNNDistributional(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                        nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                        nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1,
                                        nb_quantiles=8, nb_cos_embeddings=64)
        all_weights_equal = True
        for layer1, layer2 in zip(net1.model.get_weights(), net2.model.get_weights()):
            all_weights_equal = all_weights_equal and (layer1 == layer2).all()
        self.assertFalse(all_weights_equal)
        np.random.seed(34)
        net1 = NetworkCNNDistributional(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                        nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                        nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1,
                                        nb_quantiles=8, nb_cos_embeddings=64)
        np.random.seed(34)
        net2 = NetworkCNNDistributional(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9,
                                        nb_conv_layers=2, nb_conv_filters=32, nb_hidden_fc_layers=2,
                                        nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1,
                                        nb_quantiles=8, nb_cos_embeddings=64)
        all_weights_equal = True
        for layer1, layer2 in zip(net1.model.get_weights(), net2.model.get_weights()):
            all_weights_equal = all_weights_equal and (layer1 == layer2).all()
        self.assertTrue(all_weights_equal)


if __name__ == '__main__':
    unittest.main()
