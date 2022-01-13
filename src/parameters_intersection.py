"""
Parameters of the occluded intersection driving environment.

The meaning of the different parameters are described below.
"""

import numpy as np

# Simulation parameters
sim_params = {}
sim_params['max_steps'] = 100   # Episode number of steps
sim_params['ego_start_position'] = -200   # Episode start position
sim_params['ego_end_position'] = 30   # Episode end position
sim_params['ego_start_route'] = 'ego_route4'   # Ego start route
sim_params['ego_start_lane'] = 'ego_L31_0'   # Ego start lane
sim_params['init_steps'] = 10   # Initial steps before episode starts. At least one to get sumo subscriptions right.
sim_params['nb_vehicles'] = 99   # Number of inserted vehicles.
sim_params['adding_prob'] = 0.5   # Probability of adding a new vehicle for each time step
sim_params['remove_sumo_warnings'] = True
sim_params['safety_check'] = False   # Should be False. Otherwise, SUMO checks if agent's decisions are safe.
sim_params['sensor_range'] = 200.
sim_params['sensor_noise'] = {'pos': 0, 'speed': 0, 'heading': 0}
sim_params['occlusion_dist'] = 7   # Distance from side of road to occlusion
sim_params['sensor_nb_vehicles'] = 30   # Maximum number of vehicles the sensor can represent.
sim_params['idm_params'] = {'a': 1., 'b': 1.67, 's0': 0.5, 'T': 1.5, 'a_min': -3, 'a_max': 1}

# Reward parameters
sim_params['goal_reward'] = 10
sim_params['collision_penalty'] = -10
sim_params['near_collision_penalty'] = -10
sim_params['near_collision_margin'] = [2.5, 1]   # Distance that triggers a near collision penalty

# Vehicle type parameters
# Vehicle 0 is the ego vehicle
vehicles = []
vehicles.append({})
vehicles[0]['id'] = 'truck'
vehicles[0]['vClass'] = 'truck'
vehicles[0]['length'] = 12.0   # default 16.5
vehicles[0]['width'] = 2.55   # default 2.55
vehicles[0]['maxSpeed'] = 15.0
vehicles[0]['speedFactor'] = 1.
vehicles[0]['speedDev'] = 0
vehicles[0]['carFollowModel'] = 'Krauss'
vehicles[0]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap. A closer distance will trigger a collision.
vehicles[0]['accel'] = 1.1   # default 1.1.
vehicles[0]['decel'] = 4.0   # default 4.0.
vehicles[0]['emergencyDecel'] = 9.0   # default 4.0
vehicles[0]['sigma'] = 0.0   # default 0.5. Driver imperfection (0 = perfect driver)
vehicles[0]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[0]['color'] = '1,0,0'
vehicles[0]['laneChangModel'] = 'LC2013'
vehicles[0]['lcStrategic'] = 0
vehicles[0]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[0]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[0]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[0]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[0]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[0]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[0]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[0]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
vehicles[0]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[0]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Vehicle 1 is the type of the surrounding vehicles
vehicles.append({})
vehicles[1]['id'] = 'car'
vehicles[1]['vClass'] = 'passenger'
vehicles[1]['length'] = 4.8   # default 5.0. 4.8 used in previous paper.
vehicles[1]['width'] = 1.8   # default 1.8.
vehicles[1]['maxSpeed'] = 100.0   # Obsolete, since will be randomly set later
vehicles[1]['speedFactor'] = 1.   # Factor times the speed limit. Obsolete, since the speed is set.
vehicles[1]['speedDev'] = 0   # Randomness in speed factor. Obsolete, since speed is set.
vehicles[1]['carFollowModel'] = 'Krauss'
vehicles[1]['minGap'] = 2.5   # default 2.5. Minimum longitudinal gap.
vehicles[1]['accel'] = 2.6   # default 2.6
vehicles[1]['decel'] = 4.5   # default 4.6
vehicles[1]['emergencyDecel'] = 9.0   # default 9.0
vehicles[1]['sigma'] = 0.0   # default 0.5. Driver imperfection.
vehicles[1]['tau'] = 1.0   # default 1.0. Time headway to leading vehicle.
vehicles[1]['laneChangModel'] = 'LC2013'
vehicles[1]['lcStrategic'] = 0
vehicles[1]['lcCooperative'] = 0   # default 1.0. 0 - no cooperation
vehicles[1]['lcSpeedGain'] = 1.0   # default 1.0. Eagerness for tactical lane changes.
vehicles[1]['lcKeepRight'] = 0   # default 1.0. 0 - no incentive to move to the rightmost lane
vehicles[1]['lcOvertakeRight'] = 0   # default 0. Obsolete since overtaking on the right is allowed.
vehicles[1]['lcOpposite'] = 1.0   # default 1.0. Obsolete for freeway.
vehicles[1]['lcLookaheadLeft'] = 2.0   # default 2.0. Probably obsolete.
vehicles[1]['lcSpeedGainRight'] = 1.0   # default 0.1. 1.0 - symmetric desire to change left/right
vehicles[1]['lcAssertive'] = 1.0   # default 1.0. 1.0 - no effect
vehicles[1]['lcMaxSpeedLatFactor'] = 1.0   # default 1.0. Obsolete.
vehicles[1]['lcSigma'] = 0.0   # default 0.0. Lateral imperfection.

# Road parameters
road_params = {}
road_params['road_type'] = 'intersection'
road_params['name'] = 'intersection'
road_params['nb_lanes'] = 1
road_params['lane_width'] = 3.2   # default 3.2
road_params['max_road_speed'] = 100.   # Set very high, the actual max speed is set by the vehicle type parameters.
road_params['lane_change_duration'] = 4   # Number of time steps for a lane change
road_params['speed_range'] = np.array([10, 15])   # Speed range of surrounding vehicles.
road_params['min_start_dist'] = 30   # Minimum vehicle separation when the surrounding vehicles are added.
road_params['overtake_right'] = 'true'   # Allow overtaking on the right side.
road_params['intersection_position'] = [4000, 4000]
road_params['nodes'] = np.array(
    [[-200., 0.], [0., 0.], [200., 0.], [0., sim_params['ego_start_position'] - vehicles[0]['maxSpeed']], [0., 200.],
     [-road_params['intersection_position'][0], 0], [road_params['intersection_position'][0], 0],
     [0, -road_params['intersection_position'][1]], [0, road_params['intersection_position'][1]]])
road_params['priority'] = np.array([[0, 5, 0, 0, 0, 5, 0, 0, 0], [5, 0, 5, 3, 3, 0, 0, 0, 0],
                                    [0, 5, 0, 0, 0, 0, 5, 0, 0], [0, 3, 0, 0, 0, 0, 0, 3, 0],
                                    [0, 3, 0, 0, 0, 0, 0, 0, 3], [5, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 5, 0, 0, 0, 0, 0, 0], [0, 0, 0, 3, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 3, 0, 0, 0, 0]])
road_params['edges'] = np.array(road_params['priority'] > 0, dtype=int)
road_params['routes'] = np.array([[0, 1, 2, 6], [0, 1, 3, 7], [2, 1, 0, 5], [2, 1, 4, 8], [7, 3, 1, 4, 8]])
road_params['vehicles'] = vehicles
road_params['collision_action'] = 'warn'   # 'none', 'warn' (if none, sumo totally ignores collisions)
road_params['stop_line'] = road_params['lane_width']/2 + vehicles[1]['width']/2 + \
                           sim_params['near_collision_margin'][1] + 0.5   # Default stop position ahead of intersection

# Terminal output
road_params['emergency_decel_warn_threshold'] = 10   # A high value disables the warnings
road_params['no_display_step'] = 'true'

# Gui settings
road_params['view_position'] = np.array(road_params['intersection_position'])
road_params['zoom'] = 1400
road_params['view_delay'] = 200
road_params['info_pos'] = np.array([road_params['intersection_position'][0]-50,
                                    road_params['intersection_position'][1]-25])
road_params['action_info_pos'] = np.array([road_params['intersection_position'][0]+50,
                                           road_params['intersection_position'][1]-20])
