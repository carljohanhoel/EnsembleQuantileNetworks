"""
This script simply runs the intersection driving environment.

In this example, the ego vehicle first stops at the intersection and the continues to drive after 35 s.
"""

import numpy as np
import sys
sys.path.append('../src')
import parameters_intersection as p
from intersection_env import IntersectionEnv

p.sim_params['safety_check'] = False
ego_at_intersection = False

gui_params = {'use_gui': True, 'print_gui_info': True, 'draw_sensor_range': True, 'zoom_level': 3000}

np.random.seed(13)
env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)

episode_rewards = []
episode_steps = []
for i in range(0, 100):
    np.random.seed(i)
    env.reset(ego_at_intersection=ego_at_intersection)
    done = False
    episode_reward = 0
    step = 0
    while done is False:
        if step < 35:
            action = 2
        else:
            action = 1
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        step += 1

    episode_rewards.append(episode_reward)
    episode_steps.append(step)
    print("Episode: " + str(i))
    print("Episode steps: " + str(step))
    print("Episode reward: " + str(episode_reward))

print(episode_rewards)
print(episode_steps)
