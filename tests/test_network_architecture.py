import unittest
import numpy as np
import copy
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model

import sys
sys.path.append('../src')
from network_architecture import NetworkMLP, NetworkCNN


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)

    def test_mlp_network(self):
        net = NetworkMLP(3, 4, nb_hidden_layers=2, nb_hidden_neurons=64, duel=False, prior=False)
        self.assertTrue(net.model.trainable)
        out = net.model.predict(np.random.rand(32, 1, 3))
        self.assertEqual(np.shape(out), (32, 4))
        plot_model(net.model, to_file='mlp.png', show_shapes=True)

    def test_mlp_dueling_network(self):
        net = NetworkMLP(3, 4, nb_hidden_layers=2, nb_hidden_neurons=64, duel=True, prior=False)
        self.assertTrue(net.model.trainable)
        net_input = np.random.rand(32, 1, 3)
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 4))

        before_dueling_layer = Model(inputs=net.model.inputs, outputs=net.model.layers[-2].output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        true_output = before_dueling_output[:, 0, None] + before_dueling_output[:, 1:] - \
                      np.mean(before_dueling_output[:, 1:, None], axis=1)
        self.assertTrue(np.isclose(out, true_output).all())

        plot_model(net.model, to_file='mlp_duel.png', show_shapes=True)

    def test_combined_network(self):
        net = NetworkMLP(3, 4, nb_hidden_layers=2, nb_hidden_neurons=64, duel=False, prior=True, prior_scale_factor=1)
        trainable = []
        for layer in net.model.layers:
            trainable.append(layer.trainable)
        self.assertFalse(all(trainable))   # All layers should not be trainable
        out = net.model.predict(np.random.rand(32, 1, 3))
        self.assertEqual(np.shape(out), (32, 4))
        net.model.get_config()   # This crashes for custom lambda layers

        plot_model(net.model, to_file='mlp_prior.png', show_shapes=True)

    def test_combined_network_dueling(self):
        net = NetworkMLP(3, 4, nb_hidden_layers=2, nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1)
        trainable = []
        for layer in net.model.layers:
            trainable.append(layer.trainable)
        self.assertFalse(all(trainable))   # All layers should not be trainable
        net_input = np.random.rand(32, 1, 3)
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 4))
        net.model.get_config()  # This crashes for custom lambda layers

        before_dueling_layer = Model(inputs=net.model.inputs,
                                     outputs=net.model.get_layer('trainable_out_wo_dueling').output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        after_dueling_layer = Model(inputs=net.model.inputs, outputs=net.model.get_layer('trainable_out').output)
        after_dueling_output = after_dueling_layer.predict(net_input)
        true_dueling_output = before_dueling_output[:, 0, None] + before_dueling_output[:, 1:] - \
                              np.mean(before_dueling_output[:, 1:, None], axis=1)
        self.assertTrue(np.isclose(after_dueling_output, true_dueling_output).all())

        plot_model(net.model, to_file='mlp_duel_prior.png', show_shapes=True)

    def test_cnn_network(self):
        net = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                         nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=False, prior=False)
        self.assertTrue(net.model.trainable)
        out = net.model.predict(np.random.rand(32, 1, 44))
        self.assertEqual(np.shape(out), (32, 9))
        input1 = np.random.rand(1, 1, 44)
        out1 = net.model.predict(input1)
        input2 = np.copy(input1)
        input2[0, 0, 4:12] = input1[0, 0, 12:20]
        input2[0, 0, 12:20] = input1[0, 0, 4:12]
        self.assertFalse((input1 == input2).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        plot_model(net.model, to_file='cnn.png', show_shapes=True)

        # window_length > 1
        net = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                         nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, window_length=5, duel=False,
                         prior=False)
        net.model.predict(np.random.rand(32, 5, 44))
        self.assertEqual(np.shape(out), (32, 9))

        input1 = np.random.rand(1, 5, 44)
        out1 = net.model.predict(input1)
        input2 = np.copy(input1)
        input2[0, :, 4:12] = input1[0, :, 12:20]
        input2[0, :, 12:20] = input1[0, :, 4:12]
        self.assertFalse((input1 == input2).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        plot_model(net.model, to_file='cnn_window.png', show_shapes=True)

    def test_cnn_dueling_network(self):
        net = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                         nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=False)
        self.assertTrue(net.model.trainable)
        net_input = np.random.rand(32, 1, 44)
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 9))
        input1 = np.random.rand(1, 1, 44)
        out1 = net.model.predict(input1)
        input2 = np.copy(input1)
        input2[0, 0, 4:12] = input1[0, 0, 12:20]
        input2[0, 0, 12:20] = input1[0, 0, 4:12]
        self.assertFalse((input1 == input2).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        before_dueling_layer = Model(inputs=net.model.inputs, outputs=net.model.layers[-2].output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        true_output = before_dueling_output[:, 0, None] + before_dueling_output[:, 1:] - \
                      np.mean(before_dueling_output[:, 1:, None], axis=1)
        self.assertTrue(np.isclose(out, true_output).all())

        plot_model(net.model, to_file='cnn_duel.png', show_shapes=True)

    def test_cnn_dueling_prior(self):
        net = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                         nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                         prior_scale_factor=1)
        self.assertTrue(net.model.trainable)
        net_input = np.random.rand(32, 1, 44)
        out = net.model.predict(net_input)
        self.assertEqual(np.shape(out), (32, 9))
        input1 = np.random.rand(1, 1, 44)
        out1 = net.model.predict(input1)
        input2 = np.copy(input1)
        input2[0, 0, 4:12] = input1[0, 0, 12:20]
        input2[0, 0, 12:20] = input1[0, 0, 4:12]
        self.assertFalse((input1 == input2).all())
        out2 = net.model.predict(input2)
        self.assertTrue((out1 == out2).all())

        trainable = []
        for layer in net.model.layers:
            trainable.append(layer.trainable)
        self.assertFalse(all(trainable))  # All layers should not be trainable
        net.model.get_config()  # This crashes for custom lambda layers

        before_dueling_layer = Model(inputs=net.model.inputs,
                                     outputs=net.model.get_layer('trainable_out_wo_dueling').output)
        before_dueling_output = before_dueling_layer.predict(net_input)
        after_dueling_layer = Model(inputs=net.model.inputs, outputs=net.model.get_layer('trainable_out').output)
        after_dueling_output = after_dueling_layer.predict(net_input)
        true_dueling_output = before_dueling_output[:, 0, None] + before_dueling_output[:, 1:] - \
                              np.mean(before_dueling_output[:, 1:, None], axis=1)
        self.assertTrue(np.isclose(after_dueling_output, true_dueling_output).all())

        plot_model(net.model, to_file='cnn_duel_prior.png', show_shapes=True)

    def test_trainable_weights_mlp(self):
        net = NetworkMLP(5, 3,  nb_hidden_layers=2, nb_hidden_neurons=64, duel=True, prior=True, prior_scale_factor=1)
        net.model.compile(loss='mse', optimizer='adam')
        x = np.random.rand(100, 1, 5)
        y = np.random.rand(100, 3)
        initial_model = copy.deepcopy(net.model)
        net.model.fit(x, y, epochs=10, batch_size=100, verbose=0)
        for layer_init, layer in zip(initial_model.layers, net.model.layers):
            if not layer.trainable:
                init_weights = layer_init.get_weights()
                weights = layer.get_weights()
                for row_init, row in zip(init_weights, weights):
                    tmp = row_init == row
                    self.assertTrue(tmp.all())
        get_prior_output_initial = K.function(initial_model.get_layer('input').input,
                                              initial_model.get_layer('prior_out').output)
        prior_out_initial = get_prior_output_initial([x[0, :, :]])[0]
        get_prior_output = K.function(net.model.get_layer('input').input, net.model.get_layer('prior_out').output)
        prior_out = get_prior_output([x[0, :, :]])[0]
        self.assertTrue((prior_out_initial == prior_out).all())
        get_trainable_output_initial = K.function(initial_model.get_layer('input').input,
                                                  initial_model.get_layer('trainable_out').output)
        trainable_out_initial = get_trainable_output_initial([x[0, :, :]])[0]
        get_trainable_output = K.function(net.model.get_layer('input').input,
                                          net.model.get_layer('trainable_out').output)
        trainable_out = get_trainable_output([x[0, :, :]])[0]
        self.assertTrue((trainable_out_initial != trainable_out).all())

    def test_trainable_weights_cnn(self):
        net = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                         nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                         prior_scale_factor=1)
        net.model.compile(loss='mse', optimizer='adam')
        x = np.random.rand(100, 1, 44)
        y = np.random.rand(100, 9)
        initial_model = copy.deepcopy(net.model)
        net.model.fit(x, y, epochs=10, batch_size=100, verbose=0)
        for layer_init, layer in zip(initial_model.layers, net.model.layers):
            if not layer.trainable:
                init_weights = layer_init.get_weights()
                weights = layer.get_weights()
                for row_init, row in zip(init_weights, weights):
                    tmp = row_init == row
                    self.assertTrue(tmp.all())
        get_prior_output_initial = K.function(initial_model.get_layer('input').input,
                                              initial_model.get_layer('prior_out').output)
        prior_out_initial = get_prior_output_initial([x[:1, :, :]])[0]
        get_prior_output = K.function(net.model.get_layer('input').input, net.model.get_layer('prior_out').output)
        prior_out = get_prior_output([x[:1, :, :]])[0]
        self.assertTrue((prior_out_initial == prior_out).all())
        get_trainable_output_initial = K.function(initial_model.get_layer('input').input,
                                                  initial_model.get_layer('trainable_out').output)
        trainable_out_initial = get_trainable_output_initial([x[:1, :, :]])[0]
        get_trainable_output = K.function(net.model.get_layer('input').input,
                                          net.model.get_layer('trainable_out').output)
        trainable_out = get_trainable_output([x[:1, :, :]])[0]
        self.assertTrue((trainable_out_initial != trainable_out).all())

    def test_random_initialization(self):
        net1 = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                          nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                          prior_scale_factor=1)
        net2 = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                          nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                          prior_scale_factor=1)
        all_weights_equal = True
        for layer1, layer2 in zip(net1.model.get_weights(), net2.model.get_weights()):
            all_weights_equal = all_weights_equal and (layer1 == layer2).all()
        self.assertFalse(all_weights_equal)
        np.random.seed(34)
        net1 = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                          nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                          prior_scale_factor=1)
        np.random.seed(34)
        net2 = NetworkCNN(nb_ego_states=4, nb_states_per_vehicle=4, nb_vehicles=10, nb_actions=9, nb_conv_layers=2,
                          nb_conv_filters=32, nb_hidden_fc_layers=2, nb_hidden_neurons=64, duel=True, prior=True,
                          prior_scale_factor=1)
        all_weights_equal = True
        for layer1, layer2 in zip(net1.model.get_weights(), net2.model.get_weights()):
            all_weights_equal = all_weights_equal and (layer1 == layer2).all()
        self.assertTrue(all_weights_equal)


if __name__ == '__main__':
    unittest.main()
