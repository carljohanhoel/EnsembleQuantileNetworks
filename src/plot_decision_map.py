"""
This code is used to display the output of the agent for a specific traffic situations, but for different positions
of the ego vehicle. The action, the aleatoric uncertainty, and the epistemic uncertainty of the agent for different
positions are displayed.

Note that the interaction between this feature and SUMO creates a rather heavy computational load, which makes the
visaulization slow.
"""
import numpy as np


class DecisionMap(object):
    def __init__(self, agent, traci, intersection_pos, start_pos, end_pos, nb_actions, dy=2, value_range=(-10, 10),
                 uncertainty_range=(0, 5), x_range_action=(0.2, 3), x_range_value=(-3, -0.2),
                 x_range_aleatoric_uncertainty=(-6, -3.2), x_range_epistemic_uncertainty=(-8.8, -6)):
        self.agent = agent
        self.traci = traci
        self.intersection_pos = intersection_pos
        self.end_pos = end_pos
        self.start_pos = start_pos
        self.y_range = np.array([start_pos, end_pos]) + self.intersection_pos[1]
        self.dy = dy
        self.x_range_action = np.array(intersection_pos) + np.array(x_range_action)
        self.x_range_value = np.array(intersection_pos) + np.array(x_range_value)
        self.x_range_aleatoric_uncertainty = np.array(intersection_pos) + np.array(x_range_aleatoric_uncertainty)
        self.x_range_epistemic_uncertainty = np.array(intersection_pos) + np.array(x_range_epistemic_uncertainty)
        self.value_range = value_range
        self.uncertainty_range = uncertainty_range
        c_max = 150
        self.action_colors = [(c_max, c_max, 0, 255), (0, c_max, 0, 255), (c_max, 0, 0, 255), (0, 0, c_max, 255)]
        self.nb_actions = nb_actions
        self.polygon_names = []

    def plot(self, obs):
        for item in self.polygon_names:
            self.traci.polygon.remove(item)
        self.polygon_names = []
        for idx, y in enumerate(range(self.y_range[0], self.y_range[1], self.dy)):
            # This code should match the sensor model in intersection_env.py
            obs[0] = 2 * (y - (self.intersection_pos[1] + self.end_pos))/(self.end_pos - self.start_pos) + 1
            action, action_info = self.agent.forward(obs)
            if 'q_values' in action_info:
                if action < self.nb_actions:
                    value = action_info['q_values'][action]
                else:
                    value = np.max(action_info['q_values'])
            else:
                raise Exception('Error in decision map plot.')
            if value > 0:
                color_scale = value / self.value_range[1]
                color_scale = np.clip(color_scale, 0, 1)
                color_value = (0, color_scale*255, 0, 255)
            else:
                color_scale = (value - self.value_range[0]) / (0 - self.value_range[0])
                color_scale = np.clip(color_scale, 0, 1)
                color_value = ((1-color_scale)*255, 0, 0, 255)
            self.polygon_names.append(' '*(idx*5+3))
            self.traci.polygon.add(' '*(idx*5+3), [(self.x_range_action[0], y-self.dy/2),
                                                   (self.x_range_action[1], y-self.dy/2),
                                                   (self.x_range_action[1], y+self.dy/2),
                                                   (self.x_range_action[0], y+self.dy/2)],
                                   self.action_colors[action], fill=True, layer=5)
            self.polygon_names.append(' ' * (idx*5+4))
            self.traci.polygon.add(' ' * (idx*5+4), [(self.x_range_value[0], y - self.dy / 2),
                                                     (self.x_range_value[1], y - self.dy / 2),
                                                     (self.x_range_value[1], y + self.dy / 2),
                                                     (self.x_range_value[0], y + self.dy / 2)],
                                   color_value, fill=True, layer=5)
            if 'aleatoric_std_dev' in action_info:
                if action < self.nb_actions:
                    color_scale = action_info['aleatoric_std_dev'][action] / self.uncertainty_range[1]
                else:
                    color_scale = action_info['aleatoric_std_dev'][np.argmax(action_info['q_values'])] / \
                                  self.uncertainty_range[1]
                color_scale = np.clip(color_scale, 0, 1)
                color_value = (color_scale * 255, 0, 0, 255)
                self.polygon_names.append(' ' * (idx*5+5))
                self.traci.polygon.add(' ' * (idx*5+5), [(self.x_range_aleatoric_uncertainty[0], y - self.dy / 2),
                                                         (self.x_range_aleatoric_uncertainty[1], y - self.dy / 2),
                                                         (self.x_range_aleatoric_uncertainty[1], y + self.dy / 2),
                                                         (self.x_range_aleatoric_uncertainty[0], y + self.dy / 2)],
                                       color_value, fill=True, layer=5)
            if 'epistemic_std_dev' in action_info:
                if action < self.nb_actions:
                    color_scale = action_info['epistemic_std_dev'][action] / self.uncertainty_range[1]
                else:
                    color_scale = action_info['epistemic_std_dev'][np.argmax(action_info['q_values'])] / \
                                  self.uncertainty_range[1]
                color_scale = np.clip(color_scale, 0, 1)
                color_value = (color_scale * 255, 0, 0, 255)
                self.polygon_names.append(' ' * (idx*5+6))
                self.traci.polygon.add(' ' * (idx*5+6), [(self.x_range_epistemic_uncertainty[0], y - self.dy / 2),
                                                         (self.x_range_epistemic_uncertainty[1], y - self.dy / 2),
                                                         (self.x_range_epistemic_uncertainty[1], y + self.dy / 2),
                                                         (self.x_range_epistemic_uncertainty[0], y + self.dy / 2)],
                                       color_value, fill=True, layer=5)
