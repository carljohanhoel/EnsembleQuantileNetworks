import keras.backend as K
from keras.models import Model
from keras.layers import Lambda, Input
from rl.util import clone_model, get_object_config
import numpy as np
from core import Agent


class IQNAgent(Agent):
    """
    Implicit Quantile Networks (IQN) agent class.


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
    def __init__(self, model, nb_actions, memory,  gamma, batch_size, nb_steps_warmup, train_interval, memory_interval,
                 target_model_update, delta_clip, policy, test_policy, enable_double_dqn, nb_samples_policy,
                 nb_sampled_quantiles, cvar_eta, custom_model_objects=None, *args, **kwargs):
        super(IQNAgent, self).__init__(*args, **kwargs)

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
        self.model = model
        self.policy = policy
        self.test_policy = test_policy

        # State.
        self.compiled = False
        self.reset_states()

        # Validate (important) input.
        if hasattr(model.output, '__len__') and len(model.output) > 1:
            raise ValueError(
                'Model "{}" has more than one output. IQN expects a model that has a single output.'.format(model))
        if model.output._keras_shape != (None, self.nb_sampled_quantiles, self.nb_actions):
            raise ValueError(
                'Model output "{}" has invalid shape.')
        if not self.nb_samples_policy == self.nb_sampled_quantiles:
            raise ValueError(
                'For practical reasons, for now, nb_samples_policy and nb sampled_quantiles have to be equal')

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

    def compile(self, optimizer, metrics=[]):
        """ Calculate the quantile huber loss, see the paper for details. """
        metrics += [self.mean_q]  # register default metrics
        metrics += [self.max_q]

        # We never train the target model, hence we can set the optimizer and loss arbitrarily.
        self.target_model = clone_model(self.model, self.custom_model_objects)
        self.target_model.compile(optimizer='sgd', loss='mse')
        self.model.compile(optimizer='sgd', loss='mse')

        # Compile model.
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
        trainable_model = Model(inputs=ins + [y_true, mask], outputs=[loss_out, y_pred])
        assert len(trainable_model.output_names) == 2
        combined_metrics = {trainable_model.output_names[1]: metrics}
        losses = [
            lambda y_true, y_pred: y_pred,  # loss is computed in Lambda layer
            lambda y_true, y_pred: K.zeros_like(y_pred),  # we only include this for the metrics
        ]
        trainable_model.compile(optimizer=optimizer, loss=losses, metrics=combined_metrics)
        self.trainable_model = trainable_model

        self.compiled = True

    def load_weights(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model_hard()

    def save_weights(self, filepath, overwrite=False):
        self.model.save_weights(filepath, overwrite=overwrite)

    def reset_states(self):
        self.recent_action = None
        self.recent_observation = None
        if self.compiled:
            self.model.reset_states()
            self.target_model.reset_states()

    def update_target_model_hard(self):
        """ Copy current network parameters to the target network. """
        self.target_model.set_weights(self.model.get_weights())

    def compute_batch_z_values(self, state_batch, tau_batch):
        batch = self.process_state_batch(state_batch)
        z_values = self.model.predict_on_batch([batch, tau_batch])
        assert z_values.shape == (len(state_batch), self.nb_sampled_quantiles, self.nb_actions)
        return z_values

    def compute_sampled_z_values(self, state, max_tau):
        if self.training:
            tau = self.sample_tau_values(max_tau=max_tau)
        else:
            tau = self.sample_tau_values(max_tau=max_tau, uniform=True)
        z_values = self.compute_batch_z_values([state], tau)
        assert z_values.shape == (1, self.nb_sampled_quantiles, self.nb_actions)
        return z_values, tau

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
        state = self.memory.get_recent_state(observation)
        z_values, tau = self.compute_sampled_z_values(state, max_tau=self.cvar_eta)
        tau = np.squeeze(tau)
        z_values = np.squeeze(z_values, axis=0)
        if self.training:
            action, policy_info = self.policy.select_action(z_values=z_values)
        else:
            action, policy_info = self.test_policy.select_action(z_values=z_values)

        # Book-keeping.
        self.recent_observation = observation
        self.recent_action = action

        info = {'z_values': z_values, 'quantiles': tau, 'q_values': np.mean(z_values, axis=-2),
                'aleatoric_std_dev': np.std(z_values, axis=0)}
        info.update(policy_info)
        return action, info

    def backward(self, reward, terminal):
        """ Store the most recent experience in the replay memory and update all ensemble networks. """

        # Store most recent experience in memory.
        if self.step % self.memory_interval == 0:
            self.memory.append(self.recent_observation, self.recent_action, reward, terminal,
                               training=self.training)

        metrics = [np.nan for _ in self.metrics_names]
        if not self.training:
            # We're done here. No need to update the experience memory since we only use the working
            # memory to obtain the state over the most recent observations.
            return metrics

        # Train the network on a single stochastic batch.
        if self.step > self.nb_steps_warmup and self.step % self.train_interval == 0:
            experiences = self.memory.sample(self.batch_size)
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
                z_values = self.model.predict_on_batch([state1_batch, tau_values_policy])
            else:
                z_values = self.target_model.predict_on_batch([state1_batch, tau_values_policy])
            assert z_values.shape == (self.batch_size, self.nb_sampled_quantiles, self.nb_actions)
            actions = np.argmax(np.mean(z_values, axis=-2), axis=-1)
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

        if self.target_model_update >= 1 and self.step % self.target_model_update == 0:
            self.update_target_model_hard()

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
            'model':  get_object_config(self.model),
            'policy':  get_object_config(self.policy),
            'test_policy': get_object_config(self.test_policy),
        }
        if self.compiled:
            config['target_model'] = get_object_config(self.target_model)
        return config

    @property
    def layers(self):
        return self.model.layers[:]

    @property
    def metrics_names(self):
        # Throw away individual losses and replace output name since this is hidden from the user.
        assert len(self.trainable_model.output_names) == 2
        dummy_output_name = self.trainable_model.output_names[1]
        model_metrics = [name for idx, name in enumerate(self.trainable_model.metrics_names) if idx not in (1, 2)]
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
