import warnings
import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input
from rl.util import clone_model, get_object_config
from keras.optimizers import Adam
import numpy as np
import multiprocessing as mp
from core import Agent
from network_architecture_distributional import NetworkMLPDistributional, NetworkCNNDistributional


class IqnRpfAgent(Agent):
    """
    Ensemble Quantile Networks (EQN) agent class.


    # Arguments
        model: A Keras model.
        nb_actions (int): Number of possible actions.
        memory: Replay memory.
        gamma (float): MDP discount factor.
        batch_size (int): Batch size for stochastic gradient descent.
        nb_steps_warmup (int): Steps before training starts.
        train_interval (int): Steps between backpropagation calls.
        memory_interval (int): Steps between samples stored to memory.
        target_model_update (int): Steps between copying the parameters of the trained network to the target network.
        delta_clip (float): Huber loss parameter.
        policy: Policy used during the training phase.
        test_policy: Policy used during testing episodes.
        enable_double_dqn: A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        nb_samples_policy (int): Number of samples used by the policy, parameter K.
        nb_sampled_quantiles (int): Number of sampled quantiles, N and N', in the loss function.
        cvar_eta (float): Parameter for risk-sensitive training, only considering quantiles up to parameter value
        custom_model_objects (dict): Not currently used.
    """
    def __init__(self, models, nb_actions, memory,  gamma, batch_size, nb_steps_warmup, train_interval, memory_interval,
                 target_model_update, delta_clip, policy, test_policy, enable_double_dqn, nb_samples_policy,
                 nb_sampled_quantiles, cvar_eta, custom_model_objects=None, *args, **kwargs):
        super(IqnRpfAgent, self).__init__(*args, **kwargs)

        # Parameters.
        self.nb_actions = nb_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.nb_steps_warmup = nb_steps_warmup
        self.train_interval = train_interval
        self.memory_interval = memory_interval
        self.target_model_update = int(target_model_update)
        self.delta_clip = delta_clip
        self.custom_model_objects = {} if custom_model_objects is None else custom_model_objects
        self.enable_double_dqn = enable_double_dqn
        self.nb_samples_policy = nb_samples_policy
        self.nb_sampled_quantiles = nb_sampled_quantiles
        self.cvar_eta = cvar_eta

        # Related objects.
        self.memory = memory
        self.models = models
        self.nb_models = len(self.models)
        self.active_model = np.random.randint(self.nb_models)
        self.policy = policy
        self.test_policy = test_policy
        self.trainable_models = []
        self.target_models = None

        # State.
        self.compiled = False
        self.recent_action = None
        self.recent_observation = None
        self.reset_states()
        self.parallel = False

        # Validate (important) input.
        if models[0] is not None:   # For parallel case
            if hasattr(models[0].output, '__len__') and len(models[0].output) > 1:
                raise ValueError(
                    'Model "{}" has more than one output. EQN expects a model that has a single output.'.format(models[0]))
            if models[0].output._keras_shape != (None, self.nb_sampled_quantiles, self.nb_actions):
                raise ValueError(
                    'Model output "{}" has invalid shape.')
        if not self.nb_samples_policy == self.nb_sampled_quantiles:
            raise ValueError(
                'For practical reasons, for now, nb_samples_policy and nb_sampled_quantiles have to be equal')

    def quantile_huber_loss(self, y_true, y_pred, tau, clip_value):
        """ Calculate the quantile huber loss, see the paper for details. """
        assert K.backend() == 'tensorflow'  # Only works with tensorflow for now. Minor changes are required to support other backends.
        import tensorflow as tf
        assert clip_value > 0.

        # x = y_true - y_pred
        x = K.concatenate([y_true - tf.roll(y_pred, axis=1, shift=i) for i in range(K.int_shape(y_pred)[-2])], axis=1)
        tau_expanded = K.concatenate([tf.roll(tau, axis=2, shift=i) for i in range(K.int_shape(y_pred)[-2])], axis=2)

        if np.isinf(clip_value):
            # Special case for infinity since Tensorflow does have problems
            # if we compare `K.abs(x) < np.inf`.
            huber_loss = .5 * K.square(x)
        else:
            condition = K.abs(x) < clip_value
            squared_loss = .5 * K.square(x)
            linear_loss = clip_value * (K.abs(x) - .5 * clip_value)
            if hasattr(tf, 'select'):
                huber_loss = tf.select(condition, squared_loss, linear_loss)  # condition, true, false
            else:
                huber_loss = tf.where(condition, squared_loss, linear_loss)  # condition, true, false

        quantile_regression_condition = x < 0
        tau_expanded = K.repeat(K.squeeze(tau_expanded, axis=1), K.int_shape(y_pred)[-1])
        tau_expanded = K.permute_dimensions(tau_expanded, (0, 2, 1))
        quantile_minus_loss = (1 - tau_expanded) * huber_loss / clip_value
        quantile_plus_loss = tau_expanded * huber_loss / clip_value

        if hasattr(tf, 'select'):
            return tf.select(quantile_regression_condition, quantile_minus_loss,
                             quantile_plus_loss)  # condition, true, false
        else:
            return tf.where(quantile_regression_condition, quantile_minus_loss,
                            quantile_plus_loss)  # condition, true, false

    def clipped_masked_quantile_error(self, args):
        y_true, y_pred, tau, mask = args
        loss = self.quantile_huber_loss(y_true, y_pred, tau, self.delta_clip)
        mask_expanded = K.repeat(mask, K.int_shape(loss)[-2])
        loss *= mask_expanded  # apply element-wise mask
        return K.sum(K.sum(loss, axis=-1), axis=-1) / K.int_shape(y_true)[-2]  # Divide by N'

    def max_q(self, y_true, y_pred):  # Returns average maximum Q-value of training batch
        return K.mean(K.max(K.mean(y_pred, axis=-2), axis=-1))

    def mean_q(self, y_true, y_pred):  # Returns average Q-value of training batch
        return K.mean(K.mean(K.mean(y_pred, axis=-2), axis=-1))

    def change_active_model(self):
        """ Change which ensemble member that chooses the actions for each training episode."""
        # Note, UpdateActiveModelCallback from dqn_ensemble.py is used to call this function.
        self.active_model = np.random.randint(self.nb_models)

    def compile(self, optimizer, metrics=[]):
        """ Calculate the quantile huber loss, see the paper for details. """
        metrics += [self.mean_q]  # register default metrics
        metrics += [self.max_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_models = [clone_model(model, self.custom_model_objects) for model in self.models]
        for i in range(self.nb_models):
            self.target_models[i].compile(optimizer='sgd', loss='mse')
            self.models[i].compile(optimizer='sgd', loss='mse')

        # Compile model.
        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        for model in self.models:
            y_pred = model.output
            tau = model.input[1]
            y_true = Input(name='y_true', shape=(self.nb_sampled_quantiles, self.nb_actions,))
            mask = Input(name='mask', shape=(self.nb_actions,))
            loss_out = Lambda(self.clipped_masked_quantile_error, output_shape=(1,), name='loss')([y_true, y_pred, tau, mask])
            ins = [model.input] if type(model.input) is not list else model.input
            trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
            assert len(trainable_model.output_names) == 2
            combined_metrics = {trainable_model.output_names[1]: metrics}
            losses = [
                lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
                lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
            ]
            trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
            self.trainable_models.append(trainable_model)

        self.compiled = True

    def load_weights(self, filepath):
        for i, model in enumerate(self.models):
            model.load_weights(filepath+'_'+str(i))
        self.update_target_models_hard()

    def save_weights(self, filepath, overwrite=False):
        for i, model in enumerate(self.models):
            model.save_weights(filepath + "_" + str(i), overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            for i in range(self.nb_models):
                self.models[i].reset_states()
                self.target_models[i].reset_states()

    def update_target_models_hard(self):
        """ Copy current network parameters to the target network. """
        for i, target_model in enumerate(self.target_models):
            target_model.set_weights(self.models[i].get_weights())

    def compute_batch_z_values(self, state_batch, tau_batch, net):
        batch = self.process_state_batch(state_batch)
        z_values = self.models[net].predict_on_batch([batch, tau_batch])
        assert z_values.shape == (len(state_batch), self.nb_sampled_quantiles, self.nb_actions)
        return z_values

    def compute_sampled_z_values(self, state, max_tau, net):
        tau = self.sample_tau_values(max_tau=max_tau)
        z_values = self.compute_batch_z_values([state], tau, net)
        assert z_values.shape == (1, self.nb_sampled_quantiles, self.nb_actions)
        return z_values, tau

    def compute_z_values_all_nets(self, state, max_tau):
        tau = self.sample_tau_values(max_tau=max_tau, uniform=True)
        z_values_all_nets = []
        for net in range(self.nb_models):
            z_values_all_nets.append(np.squeeze(self.compute_batch_z_values([state], tau, net), axis=0))
        z_values_all_nets = np.array(z_values_all_nets)
        assert z_values_all_nets.shape == (self.nb_models, self.nb_sampled_quantiles, self.nb_actions)
        return z_values_all_nets, tau

    def sample_tau_values(self, max_tau, batch_size=1, uniform=False):
        if uniform:
            return np.linspace(0, max_tau, self.nb_sampled_quantiles)[None, None, :]
        else:
            return np.random.rand(batch_size, 1, self.nb_sampled_quantiles) * max_tau

    def forward(self, observation):
        """
        Ask the agent to choose an action based on the current observation.
        Args:
            observation (ndarray): Current observation.

        Returns:
            action (int): Index of chosen action
            info (dict): Information about the Q-values of the chosen action.
        """
        info = {}
        # Select an action.
        state = self.memory.get_recent_state(observation)
        if self.training:
            z_values, tau = self.compute_sampled_z_values(state, max_tau=self.cvar_eta, net=self.active_model)
            tau = np.squeeze(tau)
            z_values = np.squeeze(z_values, axis=0)
            action, policy_info = self.policy.select_action(z_values=z_values)
            info['z_values'] = z_values
            info['quantiles'] = tau
            info['q_values'] = np.mean(z_values, axis=-2)
            info.update(policy_info)
        else:
            z_values_all_nets, tau = self.compute_z_values_all_nets(state, max_tau=self.cvar_eta)
            action, policy_info = self.test_policy.select_action(z_values_all_nets=z_values_all_nets)
            info['z_values_all_nets'] = z_values_all_nets
            info['quantiles'] = tau
            info['q_values_all_nets'] = np.mean(z_values_all_nets, axis=-2)
            info['q_values'] = np.mean(z_values_all_nets, axis=(0, 1))
            info['z_values'] = np.mean(z_values_all_nets, axis=0)
            info['aleatoric_std_dev'] = np.std(info['z_values'], axis=0)
            info['epistemic_std_dev'] = np.std(info['q_values_all_nets'], axis=0)
            info.update(policy_info)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        return action, info

    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = None
        for active_net in range(self.nb_models):
            metrics = self.train_single_net(active_net)

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_models_hard()

        return metrics

    def train_single_net(self, active_net):
        """ Retrieve a batch of experiences from the replay memory of the specified ensemble member and update
        the network weights. """

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(active_net, self.batch_size)
            assert len(experiences) == self.batch_size

            # Start by extracting the necessary parameters (we use a vectorized implementation).
            state0_batch = []
            reward_batch = []
            action_batch = []
            terminal1_batch = []
            state1_batch = []
            for e in experiences:
                state0_batch.append(e.state0)
                state1_batch.append(e.state1)
                reward_batch.append(e.reward)
                action_batch.append(e.action)
                terminal1_batch.append(0. if e.terminal1 else 1.)

            # Prepare and validate parameters.
            state0_batch = self.process_state_batch(state0_batch)
            state1_batch = self.process_state_batch(state1_batch)
            terminal1_batch = np.array(terminal1_batch)
            reward_batch = np.array(reward_batch)
            assert reward_batch.shape == (self.batch_size,)
            assert terminal1_batch.shape == reward_batch.shape
            assert len(action_batch) == len(reward_batch)

            # Compute Z values for mini-batch update.
            tau_values_policy = self.sample_tau_values(max_tau=self.cvar_eta, batch_size=self.batch_size)
            if self.enable_double_dqn:
                z_values = self.models[active_net].predict_on_batch([state1_batch, tau_values_policy])
            else:
                z_values = self.target_models[active_net].predict_on_batch([state1_batch, tau_values_policy])
            assert z_values.shape == (self.batch_size, self.nb_sampled_quantiles, self.nb_actions)
            actions, policy_info = self.policy.select_action(z_values=z_values)
            assert actions.shape == (self.batch_size,)
            tau_values_targets = self.sample_tau_values(max_tau=1, batch_size=self.batch_size)
            target_z_values = self.target_models[active_net].predict_on_batch([state1_batch, tau_values_targets])
            assert target_z_values.shape == (self.batch_size, self.nb_sampled_quantiles, self.nb_actions)
            z_batch = target_z_values[range(self.batch_size), :, actions]
            assert z_batch.shape == (self.batch_size, self.nb_sampled_quantiles)

            targets = np.zeros((self.batch_size, self.nb_sampled_quantiles, self.nb_actions))
            dummy_targets = np.zeros((self.batch_size, self.nb_sampled_quantiles,))
            masks = np.zeros((self.batch_size, self.nb_actions))

            # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
            # but only for the affected output units (as given by action_batch).
            discounted_reward_batch = self.gamma * z_batch
            # Set discounted reward to zero for all states that were terminal.
            discounted_reward_batch *= terminal1_batch[:, None]
            assert discounted_reward_batch.shape[0] == reward_batch.shape[0]
            Rs = reward_batch[:, None] + discounted_reward_batch
            for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
                target[:, action] = R  # update action with estimated accumulated reward
                dummy_targets[idx, :] = R
                mask[action] = 1.  # enable loss for this specific action
            targets = np.array(targets).astype('float32')
            masks = np.array(masks).astype('float32')

            # Finally, perform a single update on the entire batch. We use a dummy target since
            # the actual loss is computed in a Lambda layer that needs more complex input. However,
            # it is still useful to know the actual target to compute metrics properly.
            tau_values = self.sample_tau_values(max_tau=1, batch_size=self.batch_size)
            ins = [state0_batch, tau_values]
            metrics = self.trainable_models[active_net].train_on_batch(ins + [targets, masks], [dummy_targets, targets])
            metrics = [metric for idx, metric in enumerate(metrics) if
                       idx not in (1, 2)]  # throw away individual losses
            metrics += self.policy.metrics
            if self.processor is not None:
                metrics += self.processor.metrics

        return metrics

    def process_state_batch(self, batch):
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)

    def get_config(self):
        config = {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
            'enable_double_dqn': self.enable_double_dqn,
            'nb_samples_policy': self.nb_samples_policy,
            'nb_sampled_quantiles': self.nb_sampled_quantiles,
            'models':  [get_object_config(model) for model in self.models],
            'policy':  get_object_config(self.policy),
            'test_policy': get_object_config(self.test_policy),
        }
        if self.compiled:
            config['target_models'] = [get_object_config(target_model) for target_model in self.target_models]
        return config

    @property
    def layers(self):
        warnings.warn('Not updated for ensemble!')
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_models[0].output_names) == 2
        dummy_output_name = self.trainable_models[0].output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_models[0].metrics_names) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        return names

    @property
    def policy(self):
        return self.__policy

    @policy.setter
    def policy(self, policy):
        self.__policy = policy
        self.__policy._set_agent(self)

    @property
    def test_policy(self):
        return self.__test_policy

    @test_policy.setter
    def test_policy(self, policy):
        self.__test_policy = policy
        self.__test_policy._set_agent(self)


class IqnRpfAgentParallel(IqnRpfAgent):
    """
    Ensemble Quantile Networks (EQN) agent class, with parallel updates of all ensemble members.

    The speed of the backwards pass of the ensemble members is significantly increased if all members are updated
    in parallel.
    For correct handling of random seeds during initalization of the neural networks, the networks need to be
    created within each worker. Therefore, all the hyperparameters of the networks are used as input to this class,
    instead of just using a set of already created models as input.

    # Arguments
        nb_models (int): Number of ensemble members.
        nb_actions (int): Number of possible actions.
        memory: Replay memory.
        cnn_architecture (bool): Use convolutional architecture for surrounding vehicles.
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_states_per_vehicle (int): Number of states that describe each of the surrounding vehicles.
        nb_vehicles (int): Maximum number of surrounding vehicles.
        nb_conv_layers (int): Number of convolutional layers.
        nb_conv_filters (int): Number of convolutional filters.
        nb_hidden_fc_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        nb_cos_embeddings (int): Number of cosine embeddings of tau value.
        network_seed (int): Random seed to initialize the networks
        duel (bool): Use dueling architecture.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
        processor: Not used
        nb_samples_policy (int): Number of samples used by the policy, parameter K.
        nb_sampled_quantiles (int): Number of sampled quantiles, N and N', in the loss function.
        cvar_eta (float): Parameter for risk-sensitive training, only considering quantiles up to parameter value
        gamma (float): MDP discount factor.
        batch_size (int): Batch size for stochastic gradient descent.
        nb_steps_warmup (int): Steps before training starts.
        train_interval (int): Steps between backpropagation calls.
        memory_interval (int): Steps between samples stored to memory.
        target_model_update (int): Steps between copying the parameters of the trained network to the target network.
        delta_clip (float): Huber loss parameter.
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
        policy: Policy used during the training phase.
        test_policy: Policy used during testing episodes.
        enable_double_dqn (bool): A boolean which enable target network as a second network proposed by van Hasselt et al. to decrease overfitting.
        enable_dueling_dqn (bool): Use dueling neural network architecture or not.
        nb_samples_policy (int): Number of samples used by the policy, parameter K.
        nb_sampled_quantiles (int): Number of sampled quantiles, N and N', in the loss function.
        custom_model_objects (dict): Not currently used.
    """
    def __init__(self, nb_models, nb_actions, memory, cnn_architecture, learning_rate, nb_ego_states,
                 nb_states_per_vehicle, nb_vehicles, nb_conv_layers, nb_conv_filters, nb_hidden_fc_layers,
                 nb_hidden_neurons, nb_cos_embeddings, network_seed, prior_scale_factor, nb_samples_policy,
                 nb_sampled_quantiles, cvar_eta, gamma, batch_size, nb_steps_warmup, train_interval, memory_interval,
                 target_model_update, delta_clip, window_length, policy, test_policy, enable_double_dqn,
                 enable_dueling_dqn, custom_model_objects=None, *args, **kwargs):
        super(IqnRpfAgentParallel, self).__init__(models=[None for _ in range(nb_models)],
                                                  nb_actions=nb_actions,
                                                  memory=memory,
                                                  gamma=gamma,
                                                  batch_size=batch_size,
                                                  nb_steps_warmup=nb_steps_warmup,
                                                  train_interval=train_interval,
                                                  memory_interval=memory_interval,
                                                  target_model_update=target_model_update,
                                                  delta_clip=delta_clip,
                                                  custom_model_objects=custom_model_objects,
                                                  policy=policy,
                                                  test_policy=test_policy,
                                                  enable_double_dqn=enable_double_dqn,
                                                  nb_samples_policy=nb_samples_policy,
                                                  nb_sampled_quantiles=nb_sampled_quantiles,
                                                  cvar_eta=cvar_eta,
                                                  )

        # Parameters.
        self.enable_dueling_dqn = enable_dueling_dqn

        # Related objects.
        self.nb_models = nb_models
        self.active_model = np.random.randint(nb_models)
        self.lr = learning_rate

        # Network parameters
        self.cnn_architecture = cnn_architecture
        self.nb_ego_states = nb_ego_states
        self.nb_states_per_vehicle = nb_states_per_vehicle
        self.nb_vehicles = nb_vehicles
        self.nb_conv_layers = nb_conv_layers
        self.nb_conv_filters = nb_conv_filters
        self.nb_hidden_fc_layers = nb_hidden_fc_layers
        self.nb_hidden_neurons = nb_hidden_neurons
        self.network_seed = network_seed
        self.prior_scale_factor = prior_scale_factor
        self.window_length = window_length
        self.nb_cos_embeddings = nb_cos_embeddings

        # State.
        self.reset_states()

        self.parallel = True
        self.input_queues = None
        self.output_queues = None

        self.init_parallel_execution()
        self.compiled = True

    def init_parallel_execution(self):
        """ Initalize one worker for each ensemble member and set up corresponding queues. """
        self.input_queues = [mp.Queue() for _ in range(self.nb_models)]
        self.output_queues = [mp.Queue() for _ in range(self.nb_models)]
        self.workers = []
        for i in range(self.nb_models):
            worker = Worker(self.network_seed + i, self.input_queues[i], self.output_queues[i],
                            cnn_architecture=self.cnn_architecture,
                            nb_ego_states=self.nb_ego_states, nb_states_per_vehicle=self.nb_states_per_vehicle,
                            nb_vehicles=self.nb_vehicles, nb_actions=self.nb_actions,
                            nb_conv_layers=self.nb_conv_layers, nb_conv_filters=self.nb_conv_filters,
                            nb_hidden_fc_layers=self.nb_hidden_fc_layers, nb_hidden_neurons=self.nb_hidden_neurons,
                            nb_sampled_quantiles=self.nb_sampled_quantiles, nb_cos_embeddings=self.nb_cos_embeddings,
                            cvar_eta=self.cvar_eta,
                            duel=self.enable_dueling_dqn, prior_scale_factor=self.prior_scale_factor,
                            window_length=self.window_length,
                            processor=self.processor, batch_size=self.batch_size,
                            enable_double_dqn=self.enable_double_dqn, gamma=self.gamma, lr=self.lr,
                            delta_clip=self.delta_clip, target_model_update=self.target_model_update,
                            policy=self.policy, mean_q=self.mean_q, max_q=self.max_q,
                            clipped_masked_quantile_error=self.clipped_masked_quantile_error,
                            sample_tau_values=self.sample_tau_values)
            self.workers.append(worker)
        for worker in self.workers:
            worker.start()

    def get_config(self):
        config = {
            'nb_actions': self.nb_actions,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'nb_steps_warmup': self.nb_steps_warmup,
            'train_interval': self.train_interval,
            'memory_interval': self.memory_interval,
            'target_model_update': self.target_model_update,
            'delta_clip': self.delta_clip,
            'memory': get_object_config(self.memory),
            'enable_double_dqn': self.enable_double_dqn,
            'enable_dueling_dqn': self.enable_dueling_dqn,
            'nb_samples_policy': self.nb_samples_policy,
            'nb_sampled_quantiles': self.nb_sampled_quantiles,
            # 'models':  [get_object_config(model) for model in self.models],
            'policy':  get_object_config(self.policy),
            'test_policy': get_object_config(self.test_policy),
        }
        return config

    def get_model_as_string(self):
        self.input_queues[0].put(['model_as_string'])   # All models are the same, so enough to get one of them
        return self.output_queues[0].get()

    def load_weights(self, filepath):
        for i in range(self.nb_models):
            self.input_queues[i].put(['load_weights', filepath+"_"+str(i)])
            output = self.output_queues[i].get()
            proc_name = self.workers[i].name
            assert(output == 'weights_loaded_' + proc_name)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        for i in range(self.nb_models):
            self.input_queues[i].put(['save_weights', filepath+"_"+str(i), overwrite])
            output = self.output_queues[i].get()
            proc_name = self.workers[i].name
            assert(output == 'weights_saved_' + proc_name)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            for i in range(self.nb_models):
                self.input_queues[i].put(['reset_states'])
                out = self.output_queues[i].get()
                proc_name = self.workers[i].name
                assert(out == 'reset_states_done_' + proc_name)

    def update_target_model_hard(self):
        """ Copy current network parameters to the target network. """
        for i in range(self.nb_models):
            self.input_queues[i].put(['update_target_model'])
            output = self.output_queues[i].get()
            proc_name = self.workers[i].name
            assert(output == 'target_model_updated_' + proc_name)

    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """
        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if self.training:
            if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
                for net in range(self.nb_models):
                    experiences = self.memory.sample(net, self.batch_size)
                    assert len(experiences) == self.batch_size
                    self.input_queues[net].put(['train', experiences])

                for net in range(self.nb_models):   # Wait for all workers to finish
                    output = self.output_queues[net].get()
                    if net == self.nb_models - 1:   # Store the metrics of the last agent
                        metrics = output[1]
                    proc_name = self.workers[net].name
                    assert(output[0] == 'training_done_' + proc_name)

                    metrics += [self.active_model]

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

        return metrics   # This is only the metrics of the last agent.

    def compute_batch_z_values(self, state_batch, tau_batch, net):
        batch = self.process_state_batch(state_batch)
        self.input_queues[net].put(['predict', [batch, tau_batch]])
        z_values = self.output_queues[net].get()
        assert z_values.shape == (len(state_batch), self.nb_sampled_quantiles, self.nb_actions)
        return z_values

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        self.input_queues[0].put(['output_names'])
        output_names = self.output_queues[0].get()
        assert len(output_names) == 2
        dummy_output_name = output_names[1]
        self.input_queues[0].put(['metrics_names'])
        metrics_names_ = self.output_queues[0].get()
        model_metrics = [name for idx, name in enumerate(metrics_names_) if idx not in (1, 2)]
        model_metrics = [name.replace(dummy_output_name + '_', '') for name in model_metrics]

        names = model_metrics + self.policy.metrics_names[:]
        if self.processor is not None:
            names += self.processor.metrics_names[:]
        names += ["active_model"]
        return names


class Worker(mp.Process):
    """
    Creates a set of workers that maintains each ensemble member.

    Args:
        seed (int): Seed of worker. Needs to be unique, otherwise all network parameters will be initialized equally.
        input_queue (multiprocessing.queues.Queue): Input queue for worker tasks.
        output_queue (multiprocessing.queues.Queue): Output queue for worker tasks.
        cnn_architecture (bool): Use CNN architecture or not.
        nb_ego_states (int): Number of states that describe the ego vehicle.
        nb_states_per_vehicle (int): Number of states that describe each of the surrounding vehicles.
        nb_vehicles (int): Maximum number of surrounding vehicles.
        nb_actions: (int): Number of outputs from the network.
        nb_conv_layers (int): Number of convolutional layers.
        nb_conv_filters (int): Number of convolutional filters.
        nb_hidden_fc_layers (int): Number of hidden layers.
        nb_hidden_neurons (int): Number of neurons in the hidden layers.
        duel (bool): Use dueling architecture.
        prior_scale_factor (float): Scale factor that balances trainable/untrainable contribution to the output.
        nb_sampled_quantiles (int): Number of sampled quantiles, N and N', in the loss function.
        nb_cos_embeddings (int): Number of cosine embeddings of tau value.
        cvar_eta (float): Parameter for risk-sensitive training, only considering quantiles up to parameter value
        window_length (int): How many historic states that are used as input. Set to 1 in this work.
        processor: Not used
        batch_size (int): Batch size for stochastic gradient descent.
        enable_double_dqn (bool): True if double DQN is used, otherwise false
        gamma (float): MDP discount factor.
        lr (float): Learning rate-
        delta_clip (float): Huber loss parameter.
        target_model_update (int): Steps between copying the paramters of the trained network to the target network.
        policy: Policy of the agent
    """
    def __init__(self, seed, input_queue, output_queue, cnn_architecture, nb_ego_states, nb_states_per_vehicle,
                 nb_vehicles, nb_actions, nb_conv_layers, nb_conv_filters, nb_hidden_fc_layers, nb_hidden_neurons, duel,
                 prior_scale_factor, nb_sampled_quantiles, nb_cos_embeddings, cvar_eta, window_length, processor,
                 batch_size, enable_double_dqn, gamma, lr, delta_clip, target_model_update, policy, mean_q, max_q,
                 clipped_masked_quantile_error, sample_tau_values, verbose=0):
        mp.Process.__init__(self)
        self.seed = seed
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.cnn_architecture = cnn_architecture
        self.nb_ego_states = nb_ego_states
        self.nb_states_per_vehicle = nb_states_per_vehicle
        self.nb_vehicles = nb_vehicles
        self.nb_actions = nb_actions
        self.nb_conv_layers = nb_conv_layers
        self.nb_conv_filters = nb_conv_filters
        self.nb_hidden_fc_layers = nb_hidden_fc_layers
        self.nb_hidden_neurons = nb_hidden_neurons
        self.nb_sampled_quantiles = nb_sampled_quantiles
        self.nb_cos_embeddings = nb_cos_embeddings
        self.cvar_eta = cvar_eta
        self.duel = duel
        self.prior_scale_factor = prior_scale_factor
        self.window_length = window_length

        self.processor = processor
        self.batch_size = batch_size
        self.enable_double_dqn = enable_double_dqn
        self.gamma = gamma
        self.target_model_update = target_model_update
        self.delta_clip = delta_clip
        self.lr = lr
        self.policy = policy

        self.verbose = verbose
        self.model = None
        self.target_model = None
        self.trainable_model = None

        self.mean_q = mean_q
        self.max_q = max_q
        self.clipped_masked_quantile_error = clipped_masked_quantile_error
        self.sample_tau_values = sample_tau_values

    def run(self):
        """ Initializes individual networks and starts the workers for each ensemble member. """
        np.random.seed(self.seed)
        proc_name = self.name
        if self.cnn_architecture:
            n = NetworkCNNDistributional(nb_ego_states=self.nb_ego_states, nb_states_per_vehicle=self.nb_states_per_vehicle,
                                         nb_vehicles=self.nb_vehicles, nb_actions=self.nb_actions, nb_conv_layers=self.nb_conv_layers,
                                         nb_conv_filters=self.nb_conv_filters, nb_hidden_fc_layers=self.nb_hidden_fc_layers,
                                         nb_hidden_neurons=self.nb_hidden_neurons, nb_quantiles=self.nb_sampled_quantiles,
                                         nb_cos_embeddings=self.nb_cos_embeddings, duel=self.duel, prior=True,
                                         prior_scale_factor=self.prior_scale_factor, window_length=self.window_length,
                                         activation='relu', duel_type='avg')
        else:
            n = NetworkMLPDistributional(nb_inputs=self.nb_ego_states + self.nb_vehicles * self.nb_states_per_vehicle,
                                         nb_outputs=self.nb_actions,
                                         nb_hidden_layers=self.nb_hidden_fc_layers,
                                         nb_hidden_neurons=self.nb_hidden_neurons, duel=self.duel,
                                         prior=True, nb_quantiles=self.nb_sampled_quantiles,
                                         nb_cos_embeddings=self.nb_cos_embeddings,
                                         activation='relu',
                                         prior_scale_factor=self.prior_scale_factor, duel_type='avg',
                                         window_length=self.window_length)
        self.model = n.model
        self.compile()

        while True:
            input_ = self.input_queue.get()
            if self.verbose:
                print("Read input proc " + proc_name + ' ' + input_[0])
            if input_ is None:  # If sending None, the process is killed
                break

            if input_[0] == 'predict':
                output = self.model.predict_on_batch(input_[1])
            elif input_[0] == 'train':
                metrics = self.train_single_net(experiences=input_[1])
                output = ['training_done_' + proc_name, metrics]
            elif input_[0] == 'reset_states':
                self.model.reset_states()
                self.target_model.reset_states()
                output = 'reset_states_done_' + proc_name
            elif input_[0] == 'update_target_model':
                self.target_model.set_weights(self.model.get_weights())
                output = 'target_model_updated_' + proc_name
            elif input_[0] == 'save_weights':
                self.model.save_weights(input_[1], overwrite=input_[2])
                output = 'weights_saved_' + proc_name
            elif input_[0] == 'load_weights':
                self.model.load_weights(input_[1])
                output = 'weights_loaded_' + proc_name
            elif input_[0] == 'output_names':
                output = self.trainable_model.output_names
            elif input_[0] == 'metrics_names':
                output = self.trainable_model.metrics_names
            elif input_[0] == 'model_as_string':
                output = self.model.to_json()

            else:
                raise Exception('input command not defined')

            self.output_queue.put(output)
        return

    def compile(self, metrics=None):
        """ Set up the training of the neural network."""
        if metrics is None:
            metrics = []
        metrics += [self.mean_q]  # register default metrics
        metrics += [self.max_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
        if self.target_model_update < 1.:
            raise Exception("Soft target model updates not implemented yet")
            # # We use the `AdditionalUpdatesOptimizer` to efficiently soft-update the target model.
            # updates = get_soft_target_model_updates(self.target_model, self.model, self.target_model_update)
            # optimizer = AdditionalUpdatesOptimizer(optimizer, updates)

        # Create trainable model. The problem is that we need to mask the output since we only
        # ever want to update the Q values for a certain action. The way we achieve this is by
        # using a custom Lambda layer that computes the loss. This gives us the necessary flexibility
        # to mask out certain parameters by passing in multiple inputs to the Lambda layer.
        y_pred = self.model.output
        tau = self.model.input[1]
        y_true = Input(name='y_true', shape=(self.nb_sampled_quantiles, self.nb_actions,))
        mask = Input(name='mask', shape=(self.nb_actions,))
        loss_out = Lambda(self.clipped_masked_quantile_error, output_shape=(1,), name='loss')([y_true, y_pred, tau, mask])
        ins = [self.model.input] if type(self.model.input) is not list else self.model.input
        self.trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(self.trainable_model.output_names) == 2
        combined_metrics = {self.trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        self.trainable_model.compile(optimizer=Adam(lr=self.lr), loss=losses, metrics=combined_metrics)

    def train_single_net(self, experiences):
        """ Retrieve a batch of experiences from the replay memory of the ensemble member and update
        the network weights. """
        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state0_batch.append(e.state0)
            state1_batch.append(e.state1)
            reward_batch.append(e.reward)
            action_batch.append(e.action)
            terminal1_batch.append(0. if e.terminal1 else 1.)

        # Prepare and validate parameters.
        state0_batch = self.process_state_batch(state0_batch)
        state1_batch = self.process_state_batch(state1_batch)
        terminal1_batch = np.array(terminal1_batch)
        reward_batch = np.array(reward_batch)
        assert reward_batch.shape == (self.batch_size,)
        assert terminal1_batch.shape == reward_batch.shape
        assert len(action_batch) == len(reward_batch)

        # Compute Z values for mini-batch update.
        tau_values_policy = self.sample_tau_values(max_tau=self.cvar_eta, batch_size=self.batch_size)
        if self.enable_double_dqn:
            z_values = self.model.predict_on_batch([state1_batch, tau_values_policy])
        else:
            z_values = self.target_model.predict_on_batch([state1_batch, tau_values_policy])
        assert z_values.shape == (self.batch_size, self.nb_sampled_quantiles, self.nb_actions)
        actions, policy_info = self.policy.select_action(z_values=z_values)
        assert actions.shape == (self.batch_size,)
        tau_values_targets = self.sample_tau_values(max_tau=1, batch_size=self.batch_size)
        target_z_values = self.target_model.predict_on_batch([state1_batch, tau_values_targets])
        assert target_z_values.shape == (self.batch_size, self.nb_sampled_quantiles, self.nb_actions)
        z_batch = target_z_values[range(self.batch_size), :, actions]
        assert z_batch.shape == (self.batch_size, self.nb_sampled_quantiles)

        targets = np.zeros((self.batch_size, self.nb_sampled_quantiles, self.nb_actions))
        dummy_targets = np.zeros((self.batch_size, self.nb_sampled_quantiles,))
        masks = np.zeros((self.batch_size, self.nb_actions))

        # Compute r_t + gamma * max_a Q(s_t+1, a) and update the target targets accordingly,
        # but only for the affected output units (as given by action_batch).
        discounted_reward_batch = self.gamma * z_batch
        # Set discounted reward to zero for all states that were terminal.
        discounted_reward_batch *= terminal1_batch[:, None]
        assert discounted_reward_batch.shape[0] == reward_batch.shape[0]
        Rs = reward_batch[:, None] + discounted_reward_batch
        for idx, (target, mask, R, action) in enumerate(zip(targets, masks, Rs, action_batch)):
            target[:, action] = R  # update action with estimated accumulated reward
            dummy_targets[idx, :] = R
            mask[action] = 1.  # enable loss for this specific action
        targets = np.array(targets).astype('float32')
        masks = np.array(masks).astype('float32')

        # Finally, perform a single update on the entire batch. We use a dummy target since
        # the actual loss is computed in a Lambda layer that needs more complex input. However,
        # it is still useful to know the actual target to compute metrics properly.
        tau_values = self.sample_tau_values(max_tau=1, batch_size=self.batch_size)
        ins = [state0_batch, tau_values]
        metrics = self.trainable_model.train_on_batch(ins + [targets, masks], [dummy_targets, targets])
        metrics = [metric for idx, metric in enumerate(metrics) if
                   idx not in (1, 2)]  # throw away individual losses
        metrics += self.policy.metrics
        if self.processor is not None:
            metrics += self.processor.metrics

        return metrics

    def process_state_batch(self, batch):
        """ Heritage from keras-rl, not used here. """
        batch = np.array(batch)
        if self.processor is None:
            return batch
        return self.processor.process_state_batch(batch)
