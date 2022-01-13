"""
Run trained agent on test episodes or special cases.

This script should be called from the folder src, where the script is located.

The options for what to run are set below, after the import statements. The options are:
- Which agent to run, defined by:
   - filepath (str): '../logs/DATE_TIME_NAME/', which is the folder name of log file for a specific run
   - agent name (str): 'STEP', which chooses the agent that was saved after STEP training steps.
                       In case of ensemble, only use STEP and do not include _N, where N is the ensemble index.
- Which case to run:
   - case (str): 'CASE_NAME', options 'rerun_test_scenarios', 'high_speed'
         - rerun_test_scenarios: This option will run the trained agent on the same test episodes that are used
                                 to evaluate the performance of the agent during the training process.
          - high_speed: This option lets the ego vehicle start close to the intersection and allow the speed of
                        the crossing vehicles to be increased outside of the training distribution, in order to
                        demonstrate the detection of epistemic uncertainty.
   - speed_increase (float): Only applies for case=='high_speed'. Added to speed range of other vehicles.
- Limit allowed uncertainty:
   - safety_threshold_aleatoric (float): Allowed standard deviation of Z-values
   - safety_threshold_epistemic (float): Allowed standard deviation of Q-values
- Number of reruns and saving of data:
   - save_rerun_data (bool)
   - save_all_uncertainty_data (bool)
   - nb_reruns (int)
- GUI options:
   - use_gui (bool): Display simulation or not. Useful to understand what the agent does, but slows the simulation down.
   - print_gui_info (bool): Print additional information in GUI.
   - draw_sensor_range (bool)
   - plot_decision_map (bool): Note, induces heavy overhead, makes visualization slow.
   - save_video (bool): Note, due to SUMO/traci handling of screenshots, printout and vehicle colors are updated one frame after vehicle is moved.
   - zoom_level (int)
"""

import os
import sys
import numpy as np
import pickle
from keras.optimizers import Adam
from keras.models import model_from_json, clone_model
from rl.policy import GreedyQPolicy
from rl.memory import Memory
from dqn_standard import DQNAgent
from dqn_ensemble import DQNAgentEnsemble, UpdateActiveModelCallback
from iqn_ensemble import IqnRpfAgent
from iqn import IQNAgent
from iqn_visualization import IQNVisualization
from time_series_visualization import TimeSeriesVisualization
from plot_decision_map import DecisionMap
from policy import EnsembleTestPolicy, DistributionalEpsGreedyPolicy, DistributionalEnsembleTestPolicy, DistributionalTestPolicy
from intersection_env import IntersectionEnv
import traci
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42   # To avoid Type 3 fonts in figures
rcParams['ps.fonttype'] = 42


""" Options: """
# Choose which agent version to run, either by setting filepath and agent_name here, or when calling the script.
filepath = '../logs/DATE_TIME_NAME/'
agent_name = 'STEP'
if len(sys.argv) > 1:
    filepath = sys.argv[1]
    agent_name = sys.argv[2]

case = 'rerun_test_scenarios'   # 'rerun_test_scenarios' or 'high_speed', see description above
safety_threshold_aleatoric = None
safety_threshold_epistemic = None
speed_increase = 10.0   # Only applies for case=='high_speed'. Added to speed range of other vehicles.
save_rerun_data = False
save_all_uncertainty_data = False
nb_reruns = 1000
""" End options """

""" GUI options """
use_gui = True
print_gui_info = True
draw_sensor_range = True
plot_decision_map = False   # Note, induces heavy overhead, makes visualization slow.
save_video = False   # Note: due to SUMO/traci handling of screenshots, printout and vehicle colors are updated one frame after vehicle is moved.
zoom_level = 5000
""" End GUI options"""

# These import statements need to come after the choice of which agent that should be used.
sys.path.insert(0, filepath + 'src/')
import parameters_stored as p   # Selects parameters that were used during specific run, stored in log folder
import parameters_intersection_stored as ps

with open(filepath + 'model.txt') as text_file:
    saved_model = model_from_json(text_file.read())
if p.agent_par["ensemble"]:
    models = []
    for i in range(p.agent_par["number_of_networks"]):
        models.append(clone_model(saved_model))
    print(models[0].summary())
else:
    model = saved_model
    print(model.summary())

# These two lines are simply needed to create the agent, and are not used further.
nb_actions = 3
memory = Memory(window_length=p.agent_par['window_length'])

# Initialize agent
if p.agent_par['distributional'] and p.agent_par['ensemble']:
    greedy_policy = DistributionalEpsGreedyPolicy(eps=0)
    test_policy = DistributionalEnsembleTestPolicy(aleatoric_threshold=safety_threshold_aleatoric,
                                                   epistemic_threshold=safety_threshold_epistemic)
    agent = IqnRpfAgent(models=models, policy=greedy_policy, test_policy=test_policy,
                        enable_double_dqn=p.agent_par['double_q'],
                        nb_samples_policy=p.agent_par['nb_samples_iqn_policy'],
                        nb_sampled_quantiles=p.agent_par['nb_quantiles'],
                        cvar_eta=p.agent_par['cvar_eta'],
                        nb_actions=nb_actions, memory=memory,
                        gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                        nb_steps_warmup=p.agent_par['learning_starts'],
                        train_interval=p.agent_par['train_interval'],
                        memory_interval=p.agent_par['memory_interval'],
                        target_model_update=p.agent_par['target_network_update_freq'],
                        delta_clip=p.agent_par['delta_clip'])
elif p.agent_par['distributional']:
    policy = DistributionalEpsGreedyPolicy(eps=0)
    test_policy = DistributionalTestPolicy(safety_threshold=safety_threshold_aleatoric)
    agent = IQNAgent(model=model, policy=policy, test_policy=test_policy, enable_double_dqn=p.agent_par['double_q'],
                     nb_samples_policy=p.agent_par['nb_samples_iqn_policy'],
                     nb_sampled_quantiles=p.agent_par['nb_quantiles'],
                     cvar_eta=p.agent_par['cvar_eta'],
                     nb_actions=nb_actions, memory=memory,
                     gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                     nb_steps_warmup=p.agent_par['learning_starts'], train_interval=p.agent_par['train_interval'],
                     memory_interval=p.agent_par['memory_interval'],
                     target_model_update=p.agent_par['target_network_update_freq'],
                     delta_clip=p.agent_par['delta_clip'])
elif p.agent_par["ensemble"]:
    policy = GreedyQPolicy()
    test_policy = EnsembleTestPolicy(safety_threshold=safety_threshold_epistemic)
    agent = DQNAgentEnsemble(models=models, policy=policy, test_policy=test_policy,
                             enable_double_dqn=p.agent_par['double_q'],
                             nb_actions=nb_actions, memory=memory, gamma=p.agent_par['gamma'],
                             batch_size=p.agent_par['batch_size'], nb_steps_warmup=p.agent_par['learning_starts'],
                             train_interval=p.agent_par['train_interval'],
                             memory_interval=p.agent_par['memory_interval'],
                             target_model_update=p.agent_par['target_network_update_freq'],
                             delta_clip=p.agent_par['delta_clip'])
else:
    policy = GreedyQPolicy()
    test_policy = GreedyQPolicy()
    agent = DQNAgent(model=model, policy=policy, test_policy=test_policy, enable_double_dqn=p.agent_par['double_q'],
                     nb_actions=nb_actions, memory=memory,
                     gamma=p.agent_par['gamma'], batch_size=p.agent_par['batch_size'],
                     nb_steps_warmup=p.agent_par['learning_starts'], train_interval=p.agent_par['train_interval'],
                     memory_interval=p.agent_par['memory_interval'],
                     target_model_update=p.agent_par['target_network_update_freq'],
                     delta_clip=p.agent_par['delta_clip'])
agent.compile(Adam(lr=p.agent_par['learning_rate']))

agent.load_weights(filepath+agent_name)
agent.training = False

if save_video:
    if not os.path.isdir("../videos"):
        os.mkdir('../videos')
    agent_video_folder = '../videos/' + filepath[8:-1] + '___' + agent_name
    if safety_threshold_aleatoric:
        agent_video_folder += '_sigma_a_' + str(safety_threshold_aleatoric)
    if safety_threshold_epistemic:
        agent_video_folder += '_sigma_e_' + str(safety_threshold_epistemic)
    if not os.path.isdir(agent_video_folder):
        os.mkdir(agent_video_folder)

# Set up visualization
gui_params = {'use_gui': use_gui, 'print_gui_info': print_gui_info, 'draw_sensor_range': draw_sensor_range,
              'zoom_level': zoom_level}
if case == 'high_speed':
    original_max_speed = ps.road_params['speed_range'][1]
    ps.road_params['speed_range'] = [speed + speed_increase for speed in ps.road_params['speed_range']]
    ps.sim_params['init_steps'] = 20   # More steps needed, since ego vehicle starts closer to intersection
env = IntersectionEnv(sim_params=ps.sim_params, road_params=ps.road_params, gui_params=gui_params)
if case == 'high_speed':
    env.sensor_max_speed_scale = original_max_speed
if use_gui:
    visualizer = IQNVisualization(nb_actions=nb_actions, nb_quantiles=p.agent_par['nb_quantiles'],
                                  iqn=p.agent_par['distributional'], cvar_eta=p.agent_par['cvar_eta'])
    nb_plots = 2
    nb_lines_per_plot = [1, nb_actions+1]
    labels = [['speed'], ['chosen', 'cruise', 'go', 'stop']]
    titles = ('Speed vs position', 'Epistemic uncertainty vs position')
    time_series_visualizer = TimeSeriesVisualization(nb_plots=nb_plots, nb_lines_per_plot=nb_lines_per_plot,
                                                     y_range=[[0, env.max_ego_speed+1], [0, 5]],
                                                     x_range=[0, env.max_steps], labels=labels, titles=titles)
if plot_decision_map:
    decision_map = DecisionMap(agent, traci, ps.road_params['intersection_position'],
                               ps.sim_params['ego_start_position'], ps.sim_params['ego_end_position'], nb_actions)
env.reset()

# Lists for saving data of reruns
episode_rewards = []
episode_steps = []
nb_safe_actions_per_episode = []
episode_collision = []
episode_near_collision = []
episode_max_steps = []
episode_max_std_dev_a = []
episode_mean_std_dev_a = []
episode_max_std_dev_e = []
episode_mean_std_dev_e = []
episode_epistemic_uncertainty_data = []
episode_aleatoric_uncertainty_data = []
# Main loop, testing the agent in nb_reruns different scenarios
for i in range(0, nb_reruns):
    # Set up the specific scenario
    np.random.seed(i)
    obs = env.reset(ego_at_intersection=True if case == 'high_speed' else False)
    if save_video:
        video_folder = agent_video_folder + '/rerun_' + str(i) + '/'
        if not os.path.isdir(video_folder):
            os.mkdir(video_folder)
        traci.gui.screenshot("View #0", video_folder + str(0) + ".png")
    if use_gui:
        time_series_visualizer.clear_plots()
    done = False
    episode_reward = 0
    step = 0
    nb_safe_actions = 0
    max_std_dev_aleatoric = 0
    mean_std_dev_aleatoric = 0
    max_std_dev_epistemic = 0
    mean_std_dev_epistemic = 0
    std_dev_aleatoric = []
    std_dev_epistemic = []
    near_collision = False
    speed_log = [[env.positions[0, 1], env.speeds[0, 0]]]
    # Run the specific scenario
    # NOTE: The main lines of code for running the simulation are "agent.forward(obs)" and "env.step(action, action_info).
    # The rest of the code just handles visualization and storing the results.
    while done is False:
        action, action_info = agent.forward(obs)
        if use_gui:
            if p.agent_par['distributional']:
                nb_quantile_batches = 3
                quantiles_detailed = np.array([])
                z_values_detailed = np.empty((0, nb_actions))
                for batch in range(nb_quantile_batches):
                    tau = np.linspace(batch / nb_quantile_batches, (batch + 1) / nb_quantile_batches, 32)
                    if p.agent_par['ensemble']:
                        z = np.array([np.squeeze(model.predict_on_batch([obs[None, None, :], tau[None, None, :]]), axis=0)
                                      for model in agent.models])
                        z = np.mean(z, axis=0)
                    else:
                        z = np.squeeze(agent.model.predict_on_batch([obs[None, None, :], tau[None, None, :]]), axis=0)
                    quantiles_detailed = np.concatenate((quantiles_detailed, tau))
                    z_values_detailed = np.concatenate((z_values_detailed, z))
                visualizer.update_plots(action, action_info['z_values'], action_info['quantiles'],
                                        z_values_detailed, quantiles_detailed)
            else:
                visualizer.update_plots(action, q_values=action_info['q_values'])
        if plot_decision_map:
            internal_random_state = np.random.get_state()   # Remove effects of potential use of np.random
            decision_map.plot(obs)
            np.random.set_state(internal_random_state)
        if use_gui:
            if 'epistemic_std_dev' in action_info:
                uncertainty = action_info['epistemic_std_dev']
                if not action == nb_actions:
                    uncertainty = np.insert(uncertainty, 0, uncertainty[action])
                else:
                    uncertainty = np.insert(uncertainty, 0, time_series_visualizer.y_range[1][1]*0.99)
            else:
                uncertainty = 0
            time_series_visualizer.update_plots([env.speeds[0, 0], uncertainty])
        obs, rewards, done, info = env.step(action, action_info)
        episode_reward += rewards
        near_collision = near_collision or 'near_collision' in info
        speed_log.append([env.positions[0, 1], env.speeds[0, 0]])
        step += 1
        if 'safe_action' in action_info:
            if action_info['safe_action']:
                nb_safe_actions += 1
        action_idx = action if action < nb_actions else np.argmax(action_info['q_values'])
        if 'epistemic_std_dev' in action_info:
            if action_info['epistemic_std_dev'][action_idx] > max_std_dev_epistemic:
                max_std_dev_epistemic = action_info['epistemic_std_dev'][action_idx]
            mean_std_dev_epistemic += (action_info['epistemic_std_dev'][action_idx] - mean_std_dev_epistemic) / step
            if save_all_uncertainty_data:
                std_dev_epistemic.append(action_info['epistemic_std_dev'][action_idx])
        if 'aleatoric_std_dev' in action_info:
            if action_info['aleatoric_std_dev'][action_idx] > max_std_dev_aleatoric:
                max_std_dev_aleatoric = action_info['aleatoric_std_dev'][action_idx]
            mean_std_dev_aleatoric += (action_info['aleatoric_std_dev'][action_idx] - mean_std_dev_aleatoric) / step
            if save_all_uncertainty_data:
                std_dev_aleatoric.append(action_info['aleatoric_std_dev'][action_idx])
        if save_video:
            traci.gui.setZoom('View #0', zoom_level)
            traci.gui.screenshot("View #0", video_folder + str(step) + ".png")

    # Store results of specific scenario
    episode_rewards.append(episode_reward)
    episode_steps.append(step)
    nb_safe_actions_per_episode.append(nb_safe_actions)
    episode_collision.append('collision' in info['terminal_reason'])
    episode_near_collision.append(near_collision)
    episode_max_steps.append('Max steps' in info['terminal_reason'])
    episode_max_std_dev_a.append(max_std_dev_aleatoric)
    episode_mean_std_dev_a.append(mean_std_dev_aleatoric)
    episode_max_std_dev_e.append(max_std_dev_epistemic)
    episode_mean_std_dev_e.append(mean_std_dev_epistemic)
    episode_aleatoric_uncertainty_data.append(std_dev_aleatoric)
    episode_epistemic_uncertainty_data.append(std_dev_epistemic)
    print("Episode: " + str(i))
    print("Episode steps: " + str(step))
    print("Episode reward: " + str(episode_reward))
    print("Number of safety actions: " + str(nb_safe_actions))
    print('Max std dev: ' + str(max_std_dev_epistemic))
    if 'terminal_reason' in info:
        print('Terminal reason: ' + info['terminal_reason'])

# Save data from reruns
if save_rerun_data:
    save_folder = '../logs/reruns/' + filepath[8:]
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    filename = agent_name + '_nb_reruns_' + str(nb_reruns)
    if case == 'high_speed':
        filename += '_speed_' + str(speed_increase)
    if safety_threshold_aleatoric:
        filename += '_safe_a_' + str(safety_threshold_aleatoric)
    if safety_threshold_epistemic:
        filename += '_safe_e_' + str(safety_threshold_epistemic)
    with open(save_folder + filename + '.pckl', 'wb') as f:
        save_data = [episode_steps, episode_rewards, nb_safe_actions_per_episode, episode_collision,
                     episode_near_collision, episode_max_steps, episode_max_std_dev_a, episode_mean_std_dev_a,
                     episode_max_std_dev_e, episode_mean_std_dev_e]
        if save_all_uncertainty_data:
            save_data.append(episode_epistemic_uncertainty_data)
            save_data.append(episode_aleatoric_uncertainty_data)
        pickle.dump(save_data, f)

# Print results
print(episode_rewards)
print(episode_steps)
print(nb_safe_actions_per_episode)
print('Collision proportion ' + str(np.sum(episode_collision)/(i+1)))
print('Near collision proportion: ' + str(np.sum(episode_near_collision) / (i + 1)))
print('Max steps proportion ' + str(np.sum(episode_max_steps)/(i+1)))
