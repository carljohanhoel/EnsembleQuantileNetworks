from rl.policy import Policy
import numpy as np


class EnsembleTestPolicy(Policy):
    """
    Policy used by the ensemble RPF agent during the testing episodes.

    During testing episodes, the policy chooses the action that has the highest mean Q-value.
    If safety_threshold is set, the allowed standard deviation of the chosen action is limited by this value.
    If the standard deviation is too high, the backup action is selected, which is defined as one step higher
    than the normal action range.

    Args:
        safety_threshold (float): Maximum standard deviation of Q-values that is considered safe.
    """
    def __init__(self, safety_threshold=None):
        self.safety_threshold = safety_threshold

    def select_action(self, q_values_all_nets):
        mean_q_values = np.mean(q_values_all_nets, axis=0)
        action = np.argmax(mean_q_values)
        if self.safety_threshold is None:
            return action, {}
        else:
            safety_level = np.std(q_values_all_nets[:, action])
            if safety_level > self.safety_threshold:
                # Safe action defined as one step higher than normal action range
                return len(mean_q_values), {'safe_action': True, 'original_action': action}
            else:
                return action, {'safe_action': False}

    def get_config(self):
        config = super(EnsembleTestPolicy, self).get_config()
        config['safety_threshold'] = self.safety_threshold
        return config


class DistributionalTestPolicy(Policy):
    """
    Policy used by the IQN agent during the testing episodes.

    During testing episodes, the policy chooses the action that has the highest Q-value, i.e., expected Z-value.
    If safety_threshold is set, the allowed standard deviation of the chosen action is limited by this value.
    If the standard deviation is too high, the backup action is selected, which is defined as one step higher
    than the normal action range.

    Args:
        safety_threshold (float): Maximum standard deviation of Z-values that is considered safe.
    """

    def __init__(self, safety_threshold=None):
        self.safety_threshold = safety_threshold

    def select_action(self, z_values):
        q_values = np.mean(z_values, axis=0)
        action = np.argmax(q_values)

        if self.safety_threshold is None:
            return action, {}
        else:
            safety_level = np.std(z_values[:, action])
            if safety_level > self.safety_threshold:
                # Safe action defined as one step higher than normal action range
                return len(q_values),  {'safe_action': True, 'original_action': action}
            else:
                return action, {'safe_action': False}

    def get_config(self):
        config = super(DistributionalTestPolicy, self).get_config()
        config['safety_threshold'] = self.safety_threshold
        return config


class DistributionalEpsGreedyPolicy(Policy):
    """
    Policy used by the IQN agent during the training episodes.

    Actions are chosen by maximizing the mean Z-values. Exploration is achieved by selecting a random action
    with probability epsilon.

    Args:
        eps (float): epsilon greedy parameter
    """
    def __init__(self, eps):
        self.eps = eps

    def select_action(self, z_values):
        nb_actions = z_values.shape[-1]

        # First argument (self.eps > 0) avoids call to np.random for greedy policy. Needed for repeatability.
        if self.eps > 0 and np.random.uniform() < self.eps:
            action = np.random.randint(0, nb_actions)
        else:
            action = np.argmax(np.sum(z_values, axis=-2), axis=-1)
        return action, {}

    def get_config(self):
        config = super(DistributionalEpsGreedyPolicy, self).get_config()
        config['eps'] = self.eps
        return config


class DistributionalEnsembleTestPolicy(Policy):
    """
    Policy used by the EQN agent during the testing episodes.

    During testing episodes, the policy chooses the action that has the highest mean Q-value.
    If aleatoric_threshold or epistemic_threshold is set, the allowed corresponding standard deviations
    of the chosen action are limited by these values.
    If the standard deviation is too high, the backup action is selected, which is defined as one step higher
    than the normal action range.

    Args:
        aleatoric_threshold (float): Maximum standard deviation of Z-values that is considered safe.
        epistemic_threshold (float): Maximum standard deviation of Q-values that is considered safe.
    """
    def __init__(self, aleatoric_threshold=None, epistemic_threshold=None):
        self.aleatoric_threshold = aleatoric_threshold
        self.epistemic_threshold = epistemic_threshold

    def select_action(self, z_values_all_nets):
        mean_z_values = np.mean(z_values_all_nets, axis=-3)
        q_values_all_nets = np.mean(z_values_all_nets, axis=-2)
        mean_q_values = np.mean(mean_z_values, axis=-2)
        action = np.argmax(mean_q_values, axis=-1)

        if not (self.aleatoric_threshold or self.epistemic_threshold):
            return action, {}
        elif self.aleatoric_threshold and not self.epistemic_threshold:
            if np.std(mean_z_values[:, action]) > self.aleatoric_threshold:
                # Safe action defined as one step higher than normal action range
                return len(mean_q_values), {'safe_action': True, 'original_action': action}
            else:
                return action, {'safe_action': False}
        elif self.epistemic_threshold and not self.aleatoric_threshold:
            if np.std(q_values_all_nets[:, action]) > self.epistemic_threshold:
                # Safe action defined as one step higher than normal action range
                return len(mean_q_values), {'safe_action': True, 'original_action': action}
            else:
                return action, {'safe_action': False}
        else:
            if np.std(mean_z_values[:, action]) > self.aleatoric_threshold \
                    or np.std(q_values_all_nets[:, action]) > self.epistemic_threshold:
                # Safe action defined as one step higher than normal action range
                return len(mean_q_values), {'safe_action': True, 'original_action': action}
            else:
                return action, {'safe_action': False}
            pass

    def get_config(self):
        config = super(DistributionalEnsembleTestPolicy, self).get_config()
        config['aleatoric_threshold'] = self.aleatoric_threshold
        config['epistemic_threshold'] = self.epistemic_threshold
        return config
