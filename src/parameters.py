"""
Parameters of the agent.

The meaning of the different parameters are described below.
Setting debug_run = True allows a shorter run, only for debugging purposes.

The different agent versions are obtained by setting the parameters according to the following:
DQN:
agent_par["distributional"] = False
agent_par["ensemble"] = False
IQN:
agent_par["distributional"] = True
agent_par["ensemble"] = False
RPF:
agent_par["distributional"] = False
agent_par["ensemble"] = True
EQN:
agent_par["distributional"] = True
agent_par["ensemble"] = True
"""

debug_run = False   # Set True to run shorted training run, for debugging purposes.
env_seed = 13   # Random seed of environment.
random_seed = env_seed+1   # Random seed of agent.

nb_training_steps = int(3e6) if not debug_run else int(3e5)   # Number of training steps
save_freq = 50000 if not debug_run else 5000   # Number of training steps between saving the network weights
eval_freq = 50000 if not debug_run else 5000   # Number of training steps between evaluating the agent on test episodes
nb_eval_eps = 100 if not debug_run else 10   # Number of test episodes

agent_par = {}
agent_par["distributional"] = True
agent_par['nb_quantiles'] = 32   # Number of sampled quantiles, N and N', in the loss function.
agent_par['cvar_eta'] = 1
agent_par['nb_cos_embeddings'] = 64
agent_par['nb_samples_iqn_policy'] = agent_par['nb_quantiles']   # Number of sampled quantiles for the policy, K.
                                                                 # For now, has to be equal to N and N'.
agent_par["ensemble"] = True   # Ensemble RPF or standard DQN agent
agent_par["parallel"] = True   # Parallel execution of backpropagation for ensemble RPF
agent_par["number_of_networks"] = 10 if not debug_run else 5   # Number of ensemble members
agent_par["prior_scale_factor"] = 300.   # Prior scale factor, beta
agent_par["adding_prob"] = 0.5   # Probability of adding an experience to each individual ensemble replay memory
agent_par["cnn"] = True   # Set True for CNN structure applied to surrounding vehicles, otherwise MLP structure
agent_par["nb_conv_layers"] = 2
agent_par["nb_conv_filters"] = 256 if not debug_run else 32
agent_par["nb_hidden_fc_layers"] = 2
agent_par["nb_hidden_neurons"] = 256 if not debug_run else 32

agent_par["gamma"] = 0.95   # Discount factor
agent_par["learning_rate"] = 0.00005
agent_par["buffer_size"] = 500000 if not debug_run else 50000   # Replay memory
agent_par["exploration_steps"] = 500000 if not debug_run else 100000  # Steps to anneal the exploration rate to minimum.
agent_par["exploration_final_eps"] = 0.05
agent_par["train_interval"] = 1
agent_par["memory_interval"] = 1
agent_par["batch_size"] = 32
agent_par["double_q"] = True   # Use double DQN
agent_par["learning_starts"] = 50000 if not debug_run else 5000   # No training during initial steps
agent_par["target_network_update_freq"] = 20000 if not debug_run else 2000
agent_par['duel_q'] = True   # Dueling neural network architecture
agent_par['delta_clip'] = 10.   # Huber loss parameter
agent_par["window_length"] = 1   # How many historic states to include (1 uses only current state)
agent_par["tensorboard_log"] = "../logs/"
