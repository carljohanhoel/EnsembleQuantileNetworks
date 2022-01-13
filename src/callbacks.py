import numpy as np
from rl.callbacks import Callback
import tensorflow as tf
import time


class SaveWeights(Callback):
    """
    Callback to regularly save the weights of the neural network.

    The weights are only saved after an episode has ended, so not exactly at the specified saving frequency.

    Args:
        save_freq (int): Training steps between saves
        save_path (str): Path where the weights are saved.
    """
    def __init__(self, save_freq=10000, save_path=None):
        super(SaveWeights, self).__init__()
        self.save_freq = save_freq
        self.save_path = save_path
        self.nb_saves = 0

    def on_episode_end(self, episode_step, logs=None):
        if (self.nb_saves == 0 or self.model.step - (self.nb_saves - 1) * self.save_freq >= self.save_freq) \
                and self.save_path is not None:
            print("Number of steps: ", self.model.step)
            self.model.save_weights(self.save_path + "/"+str(self.model.step))
            self.nb_saves += 1


class EvaluateAgent(Callback):
    """
    Callback to evaluate agent on testing episodes.

    Args:
        eval_freq (int): Training steps between evaluation runs.
        nb_eval_eps (int): Number of evaluation episodes.
        save_path (int): Path where the result is saved.
    """
    def __init__(self, eval_freq=10000, nb_eval_eps=5, save_path=None):
        super(EvaluateAgent, self).__init__()
        self.eval_freq = eval_freq
        self.nb_eval_eps = nb_eval_eps
        self.save_path = save_path
        self.nb_evaluation_runs = 0
        self.store_data_callback = StoreTestEpisodeData(save_path)
        self.env = None
        self.writer = tf.summary.FileWriter(save_path)

    def on_episode_end(self, episode_step, logs=None):   # Necessary to run testing at the end of an episode
        if (self.nb_evaluation_runs == 0 or
            self.model.step - (self.nb_evaluation_runs-1) * self.eval_freq >= self.eval_freq) \
                and self.save_path is not None:
            test_result = self.model.test(self.env, nb_episodes=self.nb_eval_eps,
                                          callbacks=[self.store_data_callback],
                                          visualize=False)
            with open(self.save_path + '/test_rewards.csv', 'ab') as f:
                np.savetxt(f, test_result.history['episode_reward'], newline=' ')
                f.write(b'\n')
            with open(self.save_path + '/test_steps.csv', 'ab') as f:
                np.savetxt(f, test_result.history['nb_steps'], newline=' ')
                f.write(b'\n')
            self.__write_summary(test_result.history)
            self.model.training = True   # training is set to False in test function, so needs to be reset here
            self.nb_evaluation_runs += 1

    def __write_summary(self, history):
        """ Create tensorboard logs of testing episodes. """
        name = 'evaluation'
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=name + '/avg_return', simple_value=np.mean(history['episode_reward'])),
            tf.Summary.Value(tag=name + '/avg_nb_steps', simple_value=np.mean(history['nb_steps'])),
        ])
        avg_nb_near_collision = np.mean(['near_collision' in ep_info for ep_info in history['info']])
        avg_nb_max_steps = np.mean([ep_info['terminal_reason'] == 'Max steps' for ep_info in history['info']])
        avg_nb_collisions = np.mean(['collision' in ep_info['terminal_reason'] or 'Outside of road' in
                                     ep_info['terminal_reason'] for ep_info in history['info']])
        avg_nb_max_dist = np.mean([ep_info['terminal_reason'] == 'Max dist' for ep_info in history['info']])
        non_collision_eps = [ep_info['terminal_reason'] == 'Max dist' or ep_info['terminal_reason'] == 'Max steps'
                             for ep_info in history['info']]
        avg_nb_steps_non_collision_eps = np.sum([nb_steps if non_collision else 0 for nb_steps, non_collision
                                                 in zip(history['nb_steps'], non_collision_eps)]) / \
                                         np.sum(non_collision_eps)
        summary.value.append(tf.Summary.Value(tag=name + '/avg_nb_near_collisions',
                                              simple_value=avg_nb_near_collision))
        summary.value.append(tf.Summary.Value(tag=name + '/avg_nb_collisions',
                                              simple_value=avg_nb_collisions))
        summary.value.append(tf.Summary.Value(tag=name + '/avg_nb_max_steps',
                                              simple_value=avg_nb_max_steps))
        summary.value.append(tf.Summary.Value(tag=name + '/avg_nb_max_dist',
                                              simple_value=avg_nb_max_dist))
        summary.value.append(tf.Summary.Value(tag=name + '/avg_nb_steps_non_collision_eps',
                                              simple_value=avg_nb_steps_non_collision_eps))
        if 'mean_aleatoric_std_dev_chosen_action' in history['info'][0]:
            summary.value.append(tf.Summary.Value(tag=name + '/mean_aleatoric_std_dev_chosen_action',
                                                  simple_value=np.mean([ep_info['mean_aleatoric_std_dev_chosen_action']
                                                                        for ep_info in history['info']])))
        if 'max_aleatoric_std_dev_chosen_action' in history['info'][0]:
            summary.value.append(tf.Summary.Value(tag=name + '/max_aleatoric_std_dev_chosen_action',
                                                  simple_value=np.mean([ep_info['max_aleatoric_std_dev_chosen_action']
                                                                        for ep_info in history['info']])))
        if 'mean_epistemic_std_dev_chosen_action' in history['info'][0]:
            summary.value.append(tf.Summary.Value(tag=name + '/mean_epistemic_std_dev_chosen_action',
                                                  simple_value=np.mean([ep_info['mean_epistemic_std_dev_chosen_action']
                                                                        for ep_info in history['info']])))
        if 'max_epistemic_std_dev_chosen_action' in history['info'][0]:
            summary.value.append(tf.Summary.Value(tag=name + '/max_epistemic_std_dev_chosen_action',
                                                  simple_value=np.mean([ep_info['max_epistemic_std_dev_chosen_action']
                                                                        for ep_info in history['info']])))
        avg_action_prop = [np.mean([ep_info['total_actions'][i] for ep_info in
                                    history['info']])/np.mean(history['nb_steps'])
                           for i in range(self.env.nb_actions)]
        summary_actions = tf.Summary(value=[
            tf.Summary.Value(tag='action_prop' + '/' + str(i), simple_value=avg_action_prop[i])
            for i in range(self.env.nb_actions)
        ])
        time.sleep(1)  # Temporary fix for getting all tensorboard logs
        self.writer.add_summary(summary_actions, self.nb_evaluation_runs)
        time.sleep(1)  # Temporary fix for getting all tensorboard logs
        self.writer.add_summary(summary, self.nb_evaluation_runs)


class StoreTestEpisodeData(Callback):
    """
    Callback to log statistics on the test episodes.

    Args:
        save_path (int): Path where the result is saved.
    """
    def __init__(self, save_path=None):
        super(StoreTestEpisodeData, self).__init__()
        self.save_path = save_path
        self.episode = -1
        self.action_data = []
        self.reward_data = []
        self.q_values_data = None

    def on_step_end(self, episode_step, logs=None):
        assert(self.model.training is False)   # This should only be done in testing mode
        if logs is None:
            logs = {}

        if self.save_path is not None:
            if not logs['episode'] == self.episode:
                if not self.episode == -1:
                    with open(self.save_path + '/test_individual_reward_data.csv', 'ab') as f:
                        np.savetxt(f, self.reward_data, newline=' ')
                        f.write(b'\n')
                    with open(self.save_path + '/test_individual_action_data.csv', 'ab') as f:
                        np.savetxt(f, self.action_data, newline=' ')
                        f.write(b'\n')
                    if 'q_values_of_chosen_action' in logs:
                        with open(self.save_path + '/test_individual_qvalues_data.csv', 'ab') as f:
                            np.savetxt(f, self.q_values_data, newline='\n')
                            f.write(b'\n')
                self.episode = logs['episode']
                self.action_data = []
                self.reward_data = []
                self.action_data.append(logs['action'])
                self.reward_data.append(logs['reward'])
                if 'q_values_of_chosen_action' in logs:
                    self.q_values_data = []
                    self.q_values_data.append(logs['q_values_of_chosen_action'])
            else:
                self.action_data.append(logs['action'])
                self.reward_data.append(logs['reward'])
                if 'q_values_of_chosen_action' in logs:
                    self.q_values_data.append(logs['q_values_of_chosen_action'])
