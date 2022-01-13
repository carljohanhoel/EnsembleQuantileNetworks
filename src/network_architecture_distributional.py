from keras.models import Model
from keras.layers import Dense, Flatten, Lambda, add, Input, Reshape, Conv1D, MaxPooling1D, concatenate
import keras.backend as K
import numpy as np


class NetworkMLPDistributional(object):
    """
    This class is used to build a neural network with an MLP structure.

    There are different functions that builds a standard MLP, w/wo dueling architecture,
    and w/wo additional untrainable prior network.

    The IQN architecture is created by feeding sampled tau values as an input, creating the cosine embedding,
    and feeding the signals through 1D convolutional layers (in effect, using the same weights for all the tau samples).

    Args:
        nb_inputs (int): Number of inputs to the network.
        nb_outputs (int): Number of outputs from the network.
        nb_hidden_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        prior (bool): Use an additional untrainable prior network.
        nb_quantiles (int): Number of tau values to evaluate.
        nb_cos_embeddings (int): Number of cosine units that are used to expand each tau value with.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        duel_type (str): 'avg', 'max', or 'naive'
        activation (str): Type of activation function, see Keras for definition
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
    """

    def __init__(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, duel, prior, nb_quantiles, 
                 nb_cos_embeddings, prior_scale_factor=None, duel_type='avg', activation='relu', window_length=1):
        if prior:
            assert prior_scale_factor is not None
        self.model = None
        if not prior and not duel:
            self.build_mlp(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings,
                           activation=activation, window_length=window_length)
        elif not prior and duel:
            self.build_mlp_dueling(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                                   nb_cos_embeddings, dueling_type=duel_type, activation=activation,
                                   window_length=window_length)
        elif prior and not duel:
            self.build_mlp_with_prior(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                                      nb_cos_embeddings, prior_scale_factor, activation=activation,
                                      window_length=window_length)
        elif prior and duel:
            self.build_mlp_dueling_with_prior(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                                              nb_cos_embeddings, prior_scale_factor, activation=activation,
                                              dueling_type=duel_type, window_length=window_length)
        else:
            raise Exception('Error in Network creation')

    def build_mlp(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                  nb_cos_embeddings, activation='relu', kernel_initializer='glorot_normal', window_length=1):
        """
        Create the MLP network.

        The input to the network, which consists of the state and the sampled tau values, is first split.
        The state input is passed through fully connected layers. The tau values are expanded with the cosine embedding,
        and then passed through 1D convolutional layers, which in practice means one fully connected layer,
        with shared weights between the different samples of tau. The output of the state and tau parts of the network
        are then multiplied together elementwise, before being passed to a joint fully connected net.
        """
        state_input = Input(shape=(window_length, nb_inputs), name='state_input')
        flat_input = Flatten()(state_input)
        state_fc_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                             name='fc_state')(flat_input)
        for i in range(0, nb_hidden_layers-1):
            state_fc_net = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                 name='fc_state_'+str(i+1))(state_fc_net)
        extra_dim_state_fc_net = Reshape((1, nb_hidden_neurons,), input_shape=(nb_hidden_neurons,),
                                         name='fc_state_extra_dim')(state_fc_net)

        tau_input = Input(shape=(1, nb_quantiles), name='tau_input')
        extra_dim_tau = Reshape((nb_quantiles, 1,), input_shape=(nb_quantiles,))(tau_input)
        cos_embedding = Lambda(lambda tau_: K.concatenate([K.cos(n * np.pi * tau_)
                                                           for n in range(0, nb_cos_embeddings)]),
                               name='cos_tau')(extra_dim_tau)
        tau_net = Conv1D(nb_hidden_neurons, 1, strides=1, activation='relu',
                         kernel_initializer=kernel_initializer, name='fc_tau')(cos_embedding)

        merge = Lambda(lambda x: np.multiply(x[1], x[0]), name='merge')([extra_dim_state_fc_net, tau_net])

        joint_net = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                           kernel_initializer=kernel_initializer, name='joint_net')(merge)
        output = Conv1D(nb_outputs, 1, strides=1, activation='linear',
                        kernel_initializer=kernel_initializer, name='output')(joint_net)

        self.model = Model(inputs=[state_input, tau_input], output=output)

    def build_mlp_dueling(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                          nb_cos_embeddings, dueling_type='avg', activation='relu', window_length=1):
        """
        Simply adds dueling architecture to MLP network.
        """
        self.build_mlp(nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings,
                       activation=activation, window_length=window_length)
        layer = self.model.layers[-2]
        y = Conv1D(nb_outputs + 1, 1, strides=1, activation='linear', name='dueling_output')(layer.output)
        if dueling_type == 'avg':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                 K.mean(a[:, :, 1:], axis=-1, keepdims=True), output_shape=(nb_quantiles, nb_outputs,),
                                 name='output')(y)
        elif dueling_type == 'max':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                 K.max(a[:, :, 1:], axis=-1, keepdims=True),
                                 output_shape=(nb_quantiles, nb_outputs,), name='output')(y)
        elif dueling_type == 'naive':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:],
                                 output_shape=(nb_quantiles, nb_outputs,), name='output')(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = Model(inputs=self.model.input, outputs=outputlayer)

    def build_mlp_with_prior(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                             nb_cos_embeddings, prior_scale_factor, activation='relu',
                             kernel_initializer='glorot_normal', window_length=1):
        """
        Two networks are created, with identical structure, see the description of build_mlp above. One of the networks
        have trainable parameters and the other network has fixed parameters, i.e., they cannot be updated during the
        training process. The final output is created as a linear combination of the output of the two networks.
        """
        # Trainable net
        state_input = Input(shape=(window_length, nb_inputs), name='state_input')
        flat_input = Flatten()(state_input)
        state_fc_net_trainable = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                       name='fc_state_trainable')(flat_input)
        for i in range(0, nb_hidden_layers-1):
            state_fc_net_trainable = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                           name='fc_state_trainable'+str(i+1))(state_fc_net_trainable)
        extra_dim_state_fc_net_trainable = Reshape((1, nb_hidden_neurons,), input_shape=(nb_hidden_neurons,),
                                                   name='fc_state_extra_dim_trainable')(state_fc_net_trainable)

        tau_input = Input(shape=(1, nb_quantiles), name='tau_input')
        extra_dim_tau = Reshape((nb_quantiles, 1,), input_shape=(nb_quantiles,))(tau_input)
        cos_embedding = Lambda(lambda tau_: K.concatenate([K.cos(n * np.pi * tau_)
                                                           for n in range(0, nb_cos_embeddings)]),
                               name='cos_tau')(extra_dim_tau)
        tau_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation='relu',
                                   kernel_initializer=kernel_initializer, name='fc_tau_trainable')(cos_embedding)

        merge_trainable = Lambda(lambda x: np.multiply(x[1], x[0]),
                                 name='merge_trainable')([extra_dim_state_fc_net_trainable, tau_net_trainable])

        joint_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                     kernel_initializer=kernel_initializer, name='joint_net_trainable')(merge_trainable)
        output_trainable = Conv1D(nb_outputs, 1, strides=1, activation='linear',
                                  kernel_initializer=kernel_initializer, name='output_trainable')(joint_net_trainable)

        # Prior net
        state_fc_net_prior = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                   trainable=False, name='fc_state_prior')(flat_input)
        for i in range(0, nb_hidden_layers - 1):
            state_fc_net_prior = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                       trainable=False, name='fc_state_prior' + str(i + 1))(state_fc_net_prior)
        extra_dim_state_fc_net_prior = Reshape((1, nb_hidden_neurons,), input_shape=(nb_hidden_neurons,),
                                               name='fc_state_extra_dim_prior')(state_fc_net_prior)

        tau_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation='relu', trainable=False,
                               kernel_initializer=kernel_initializer, name='fc_tau_prior')(cos_embedding)

        merge_prior = Lambda(lambda x: np.multiply(x[1], x[0]),
                             name='merge_prior')([extra_dim_state_fc_net_prior, tau_net_prior])

        joint_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                 kernel_initializer=kernel_initializer, trainable=False,
                                 name='joint_net_prior')(merge_prior)
        output_prior = Conv1D(nb_outputs, 1, strides=1, activation='linear',
                              kernel_initializer=kernel_initializer, trainable=False,
                              name='output_prior')(joint_net_prior)
        output_prior_scaled = Lambda(lambda x: x * prior_scale_factor, name='output_prior_scaled')(output_prior)

        # Merge trainable and prior net
        output_add = add([output_trainable, output_prior_scaled], name='add')

        self.model = Model(inputs=[state_input, tau_input], output=output_add)

    def build_mlp_dueling_with_prior(self, nb_inputs, nb_outputs, nb_hidden_layers, nb_hidden_neurons, nb_quantiles,
                                     nb_cos_embeddings, prior_scale_factor, activation='relu',
                                     kernel_initializer='glorot_normal', dueling_type='avg', window_length=1):
        """
        Creates the network architecture described in build_mlp_with_prior, with an additional dueling structure.
        """
        # Trainable net
        state_input = Input(shape=(window_length, nb_inputs), name='state_input')
        flat_input = Flatten()(state_input)
        state_fc_net_trainable = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                       name='fc_state_trainable')(flat_input)
        for i in range(0, nb_hidden_layers-1):
            state_fc_net_trainable = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                           name='fc_state_trainable'+str(i+1))(state_fc_net_trainable)
        extra_dim_state_fc_net_trainable = Reshape((1, nb_hidden_neurons,), input_shape=(nb_hidden_neurons,),
                                                   name='fc_state_extra_dim_trainable')(state_fc_net_trainable)

        tau_input = Input(shape=(1, nb_quantiles), name='tau_input')
        extra_dim_tau = Reshape((nb_quantiles, 1,), input_shape=(nb_quantiles,))(tau_input)
        cos_embedding = Lambda(lambda tau_: K.concatenate([K.cos(n * np.pi * tau_)
                                                           for n in range(0, nb_cos_embeddings)]),
                               name='cos_tau')(extra_dim_tau)
        tau_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation='relu',
                                   kernel_initializer=kernel_initializer, name='fc_tau_trainable')(cos_embedding)

        merge_trainable = Lambda(lambda x: np.multiply(x[1], x[0]),
                                 name='merge_trainable')([extra_dim_state_fc_net_trainable, tau_net_trainable])

        joint_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                     kernel_initializer=kernel_initializer, name='joint_net_trainable')(merge_trainable)
        output_trainable_wo_dueling = Conv1D(nb_outputs + 1, 1, strides=1, activation='linear',
                                             kernel_initializer=kernel_initializer,
                                             name='output_trainable_wo_dueling')(joint_net_trainable)
        if dueling_type == 'avg':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                      K.mean(a[:, :, 1:], axis=-1, keepdims=True), output_shape=(nb_quantiles, nb_outputs,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        elif dueling_type == 'max':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                      K.max(a[:, :, 1:], axis=-1, keepdims=True),
                                      output_shape=(nb_quantiles, nb_outputs,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        elif dueling_type == 'naive':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:],
                                      output_shape=(nb_quantiles, nb_outputs,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        # Prior net
        state_fc_net_prior = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                   trainable=False, name='fc_state_prior')(flat_input)
        for i in range(0, nb_hidden_layers - 1):
            state_fc_net_prior = Dense(nb_hidden_neurons, activation=activation, kernel_initializer=kernel_initializer,
                                       trainable=False, name='fc_state_prior' + str(i + 1))(state_fc_net_prior)
        extra_dim_state_fc_net_prior = Reshape((1, nb_hidden_neurons,), input_shape=(nb_hidden_neurons,),
                                               name='fc_state_extra_dim_prior')(state_fc_net_prior)

        tau_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation='relu', trainable=False,
                               kernel_initializer=kernel_initializer, name='fc_tau_prior')(cos_embedding)

        merge_prior = Lambda(lambda x: np.multiply(x[1], x[0]),
                             name='merge_prior')([extra_dim_state_fc_net_prior, tau_net_prior])

        joint_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                 kernel_initializer=kernel_initializer, trainable=False,
                                 name='joint_net_prior')(merge_prior)
        output_prior_wo_dueling = Conv1D(nb_outputs + 1, 1, strides=1, activation='linear',
                                         kernel_initializer=kernel_initializer, trainable=False,
                                         name='output_prior_wo_dueling')(joint_net_prior)
        if dueling_type == 'avg':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                  K.mean(a[:, :, 1:], axis=-1, keepdims=True), output_shape=(nb_quantiles, nb_outputs,),
                                  name='output_prior')(output_prior_wo_dueling)
        elif dueling_type == 'max':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                  K.max(a[:, :, 1:], axis=-1, keepdims=True),
                                  output_shape=(nb_quantiles, nb_outputs,), name='output_prior')(output_prior_wo_dueling)
        elif dueling_type == 'naive':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:],
                                  output_shape=(nb_quantiles, nb_outputs,), name='output_prior')(output_prior_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        output_prior_scaled = Lambda(lambda x: x * prior_scale_factor, name='output_prior_scaled')(output_prior)

        # Merge trainable and prior net
        output_add = add([output_trainable, output_prior_scaled], name='add')

        self.model = Model(inputs=[state_input, tau_input], output=output_add)


class NetworkCNNDistributional(object):
    """
    This class is used to build a neural network with a CNN structure.

    There are different functions that builds a standard CNN, w/wo dueling architecture,
    and w/wo additional untrainable prior network.

    The IQN architecture is created by feeding sampled tau values as an input, creating the cosine embedding,
    and feeding the signals through 1D convolutional layers (in effect, using the same weights for all the tau samples).

    Args:
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_states_per_vehicle (int): Number of states that describe each of the surrounding vehicles.
        nb_vehicles (int): Maximum number of surrounding vehicles.
        nb_actions: (int): Number of outputs from the network.
        nb_conv_layers (int): Number of convolutional layers.
        nb_conv_filters (int): Number of convolutional filters.
        nb_hidden_fc_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        nb_quantiles (int): Number of tau values to evaluate.
        nb_cos_embeddings (int): Number of cosine units that are used to expand each tau value with.
        prior (bool): Use an additional untrainable prior network.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        duel_type (str): 'avg', 'max', or 'naive'
        activation (str): Type of activation function, see Keras for definition
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
    """
    def __init__(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers, nb_conv_filters,
                 nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings, duel, prior,
                 prior_scale_factor=None, duel_type='avg',
                 activation='relu', window_length=1):
        if prior:
            assert prior_scale_factor is not None
        self.model = None
        if not prior and not duel:
            self.build_cnn(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                           nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings,
                           activation=activation, window_length=window_length)
        elif not prior and duel:
            self.build_cnn_dueling(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                   nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles,
                                   nb_cos_embeddings, dueling_type=duel_type, activation=activation,
                                   window_length=window_length)
        elif prior and duel:
            self.build_cnn_dueling_prior(nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                         nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles,
                                         nb_cos_embeddings, dueling_type=duel_type, activation=activation,
                                         prior_scale_factor=prior_scale_factor, window_length=window_length)
        else:
            raise Exception('Error in Network creation')

    def build_cnn(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers, nb_conv_filters,
                  nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings, activation='relu',
                  kernel_initializer='glorot_normal', window_length=1):
        """
        See the description of build_mlp, the same architecture is used here, except for that a
        convolutional structure is applied to the input that describes the surrounding vehicles.
        """
        nb_inputs = nb_ego_states + nb_states_per_vehicle * nb_vehicles

        state_input = Input(shape=(window_length, nb_inputs), name='input')
        flat_input = Flatten(data_format='channels_first')(state_input)

        input_ego = Lambda(lambda state: state[:, :nb_ego_states * window_length], name='input_ego')(flat_input)
        input_others = Lambda(lambda state: state[:, nb_ego_states * window_length:], name='input_others')(flat_input)
        input_others_reshaped = Reshape((nb_vehicles * nb_states_per_vehicle * window_length, 1,),
                                        input_shape=(nb_vehicles * nb_states_per_vehicle *
                                                     window_length,), name='input_others_reshaped')(input_others)

        ego_net = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                        name='ego_0')(input_ego)
        for i in range(nb_conv_layers - 1):
            ego_net = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                            name='ego_' + str(i + 1))(ego_net)

        conv_net = Conv1D(nb_conv_filters, nb_states_per_vehicle*window_length,
                          strides=nb_states_per_vehicle*window_length, activation=activation,
                          kernel_initializer=kernel_initializer, name='conv_0')(input_others_reshaped)
        for i in range(nb_conv_layers-1):
            conv_net = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                              kernel_initializer=kernel_initializer, name='conv_'+str(i+1))(conv_net)
        pool = MaxPooling1D(pool_size=nb_vehicles)(conv_net)
        conv_net_out = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,), name='convnet_out')(pool)

        merged_net = concatenate([ego_net, conv_net_out])

        extra_dim_merged_net = Reshape((1, 2*nb_conv_filters,), input_shape=(nb_hidden_neurons,),
                                       name='merged_extra_dim')(merged_net)

        tau_input = Input(shape=(1, nb_quantiles), name='tau_input')
        extra_dim_tau = Reshape((nb_quantiles, 1,), input_shape=(nb_quantiles,))(tau_input)
        cos_embedding = Lambda(lambda tau_: K.concatenate([K.cos(n * np.pi * tau_)
                                                           for n in range(0, nb_cos_embeddings)]),
                               name='cos_tau')(extra_dim_tau)
        tau_net = Conv1D(2*nb_conv_filters, 1, strides=1, activation='relu',
                         kernel_initializer=kernel_initializer, name='fc_tau')(cos_embedding)

        merge = Lambda(lambda x: np.multiply(x[1], x[0]), name='merge')([extra_dim_merged_net, tau_net])

        joint_net = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                           kernel_initializer=kernel_initializer, name='joint_net_0')(merge)
        for i in range(nb_hidden_fc_layers-1):
            joint_net = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                               kernel_initializer=kernel_initializer, name='joint_net_'+str(i+1))(joint_net)

        output = Conv1D(nb_actions, 1, strides=1, activation='linear',
                        kernel_initializer=kernel_initializer, name='output')(joint_net)

        self.model = Model(inputs=[state_input, tau_input], outputs=output)

    def build_cnn_dueling(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                          nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings,
                          activation='relu', kernel_initializer='glorot_normal', window_length=1,
                          dueling_type='avg'):
        """
        See the description of build_mlp_dueling, the same architecture is used here, except for that a
        convolutional structure is applied to the input that describes the surrounding vehicles.
        """
        self. build_cnn(nb_ego_states=nb_ego_states, nb_states_per_vehicle=nb_states_per_vehicle,
                        nb_vehicles=nb_vehicles, nb_actions=nb_actions, nb_conv_layers=nb_conv_layers,
                        nb_conv_filters=nb_conv_filters, nb_hidden_fc_layers=nb_hidden_fc_layers,
                        nb_hidden_neurons=nb_hidden_neurons, nb_quantiles=nb_quantiles,
                        nb_cos_embeddings=nb_cos_embeddings, activation=activation,
                        kernel_initializer=kernel_initializer, window_length=window_length)
        layer = self.model.layers[-2]
        y = Conv1D(nb_actions + 1, 1, strides=1, activation='linear', name='dueling_output')(layer.output)
        if dueling_type == 'avg':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                 K.mean(a[:, :, 1:], axis=-1, keepdims=True),
                                 output_shape=(nb_quantiles, nb_actions,),
                                 name='output')(y)
        elif dueling_type == 'max':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:] -
                                 K.max(a[:, 1:], axis=-1, keepdims=True),
                                 output_shape=(nb_quantiles, nb_actions,), name='output')(y)
        elif dueling_type == 'naive':
            outputlayer = Lambda(lambda a: K.expand_dims(a[:, 0], -1) + a[:, 1:],
                                 output_shape=(nb_quantiles, nb_actions,), name='output')(y)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"
        self.model = Model(inputs=self.model.input, outputs=outputlayer)

    def build_cnn_dueling_prior(self, nb_ego_states, nb_states_per_vehicle, nb_vehicles, nb_actions, nb_conv_layers,
                                nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, nb_quantiles, nb_cos_embeddings,
                                activation='relu', kernel_initializer='glorot_normal',  prior_scale_factor=1.,
                                window_length=1, dueling_type='avg'):
        """
        See the description of build_mlp_dueling_prior, the same architecture is used here, except for that a
        convolutional structure is applied to the input that describes the surrounding vehicles.
        """
        nb_inputs = nb_ego_states + nb_states_per_vehicle * nb_vehicles

        state_input = Input(shape=(window_length, nb_inputs), name='state_input')
        flat_input = Flatten(data_format='channels_first')(state_input)

        input_ego = Lambda(lambda state: state[:, :nb_ego_states * window_length], name='input_ego')(flat_input)
        input_others = Lambda(lambda state: state[:, nb_ego_states * window_length:], name='input_others')(flat_input)
        input_others_reshaped = Reshape((nb_vehicles * nb_states_per_vehicle * window_length, 1,),
                                        input_shape=(nb_vehicles * nb_states_per_vehicle *
                                                     window_length,), name='input_others_reshaped')(input_others)

        ego_net_trainable = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                                  name='ego_trainable_0')(input_ego)
        for i in range(nb_conv_layers - 1):
            ego_net_trainable = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                                      name='ego_trainable_' + str(i + 1))(ego_net_trainable)

        conv_net_trainable = Conv1D(nb_conv_filters, nb_states_per_vehicle * window_length,
                                    strides=nb_states_per_vehicle * window_length, activation=activation,
                                    kernel_initializer=kernel_initializer, name='conv_trainable_0')(input_others_reshaped)
        for i in range(nb_conv_layers - 1):
            conv_net_trainable = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                                        kernel_initializer=kernel_initializer,
                                        name='conv_trainable_' + str(i + 1))(conv_net_trainable)
        pool_trainable = MaxPooling1D(pool_size=nb_vehicles)(conv_net_trainable)
        conv_net_out_trainable = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,),
                                         name='convnet_out_trainable')(pool_trainable)

        merged_trainable = concatenate([ego_net_trainable, conv_net_out_trainable])

        extra_dim_merged_net_trainable = Reshape((1, 2*nb_conv_filters,), input_shape=(nb_hidden_neurons,),
                                                 name='merged_extra_dim_trainable')(merged_trainable)

        tau_input = Input(shape=(1, nb_quantiles), name='tau_input')
        extra_dim_tau = Reshape((nb_quantiles, 1,), input_shape=(nb_quantiles,))(tau_input)
        cos_embedding = Lambda(lambda tau_: K.concatenate([K.cos(n * np.pi * tau_)
                                                           for n in range(0, nb_cos_embeddings)]),
                               name='cos_tau')(extra_dim_tau)
        tau_net_trainable = Conv1D(2*nb_conv_filters, 1, strides=1, activation='relu',
                                   kernel_initializer=kernel_initializer, name='fc_tau_trainable')(cos_embedding)

        merge_trainable = Lambda(lambda x: np.multiply(x[1], x[0]),
                                 name='merge_trainable')([extra_dim_merged_net_trainable, tau_net_trainable])

        joint_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                     kernel_initializer=kernel_initializer, name='joint_net_trainable_0')(merge_trainable)
        for i in range(nb_hidden_fc_layers-1):
            joint_net_trainable = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                         kernel_initializer=kernel_initializer,
                                         name='joint_net_trainable_'+str(i+1))(joint_net_trainable)

        output_trainable_wo_dueling = Conv1D(nb_actions+1, 1, strides=1, activation='linear',
                                             kernel_initializer=kernel_initializer,
                                             name='output_trainable_wo_dueling')(joint_net_trainable)

        if dueling_type == 'avg':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                      K.mean(a[:, :, 1:], axis=-1, keepdims=True), output_shape=(nb_quantiles, nb_actions,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        elif dueling_type == 'max':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                      K.max(a[:, :, 1:], axis=-1, keepdims=True),
                                      output_shape=(nb_quantiles, nb_actions,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        elif dueling_type == 'naive':
            output_trainable = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:],
                                      output_shape=(nb_quantiles, nb_actions,),
                                      name='output_trainable')(output_trainable_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        ego_net_prior = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                              trainable=False, name='ego_prior_0')(input_ego)
        for i in range(nb_conv_layers - 1):
            ego_net_prior = Dense(nb_conv_filters, activation=activation, kernel_initializer=kernel_initializer,
                                  trainable=False, name='ego_prior_' + str(i + 1))(ego_net_prior)

        conv_net_prior = Conv1D(nb_conv_filters, nb_states_per_vehicle * window_length,
                                strides=nb_states_per_vehicle * window_length, activation=activation,
                                kernel_initializer=kernel_initializer, trainable=False,
                                name='conv_prior_0')(input_others_reshaped)
        for i in range(nb_conv_layers - 1):
            conv_net_prior = Conv1D(nb_conv_filters, 1, strides=1, activation=activation,
                                    kernel_initializer=kernel_initializer, trainable=False,
                                    name='conv_prior_' + str(i + 1))(conv_net_prior)
        pool_prior = MaxPooling1D(pool_size=nb_vehicles)(conv_net_prior)
        conv_net_out_prior = Reshape((nb_conv_filters,), input_shape=(1, nb_conv_filters,),
                                     name='convnet_out_prior')(pool_prior)

        merged_prior = concatenate([ego_net_prior, conv_net_out_prior])

        extra_dim_merged_net_prior = Reshape((1, 2*nb_conv_filters,), input_shape=(nb_hidden_neurons,),
                                             name='merged_extra_dim_prior')(merged_prior)

        tau_net_prior = Conv1D(2*nb_conv_filters, 1, strides=1, activation='relu',
                               kernel_initializer=kernel_initializer, trainable=False, name='fc_tau_prior')(cos_embedding)

        merge_prior = Lambda(lambda x: np.multiply(x[1], x[0]),
                             name='merge_prior')([extra_dim_merged_net_prior, tau_net_prior])

        joint_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                 kernel_initializer=kernel_initializer, trainable=False,
                                 name='joint_net_prior_0')(merge_prior)
        for i in range(nb_hidden_fc_layers-1):
            joint_net_prior = Conv1D(nb_hidden_neurons, 1, strides=1, activation=activation,
                                     kernel_initializer=kernel_initializer, trainable=False,
                                     name='joint_net_prior'+str(i+1))(joint_net_prior)

        output_prior_wo_dueling = Conv1D(nb_actions+1, 1, strides=1, activation='linear',
                                         kernel_initializer=kernel_initializer, trainable=False,
                                         name='output_prior_wo_dueling')(joint_net_prior)

        if dueling_type == 'avg':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                  K.mean(a[:, :, 1:], axis=-1, keepdims=True), output_shape=(nb_quantiles, nb_actions,),
                                  name='output_prior')(output_prior_wo_dueling)
        elif dueling_type == 'max':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:] -
                                  K.max(a[:, :, 1:], axis=-1, keepdims=True),
                                  output_shape=(nb_quantiles, nb_actions,), name='output_prior')(output_prior_wo_dueling)
        elif dueling_type == 'naive':
            output_prior = Lambda(lambda a: K.expand_dims(a[:, :, 0], -1) + a[:, :, 1:],
                                  output_shape=(nb_quantiles, nb_actions,), name='output_prior')(output_prior_wo_dueling)
        else:
            assert False, "dueling_type must be one of {'avg','max','naive'}"

        prior_scale = Lambda(lambda x: x * prior_scale_factor, name='prior_scale')(output_prior)
        add_output = add([output_trainable, prior_scale], name='final_output')

        self.model = Model(inputs=[state_input, tau_input], outputs=add_output)
