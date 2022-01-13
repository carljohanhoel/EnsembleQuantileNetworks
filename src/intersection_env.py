import os
import sys
import numpy as np
import copy
import warnings
warnings.simplefilter('always', UserWarning)

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")
from sumolib import checkBinary
import traci
from road import Road

# Sumo subscription constants
POSITION = 66
LONG_SPEED = 64
LAT_SPEED = 50
LONG_ACC = 114
ANGLE = 67


class IntersectionEnv(object):
    """
    This class creates a gym-like intersection driving environment.

    The ego vehicle starts in the south and aims to cross the intersection to the north. Surrounding traffic
    is initialized at random in the west and east, with intentions to cross the intersection, turn to the north,
    or the south. The surrounding traffic follows the SUMO driver model, except for that the presence of the
    ego vehicle is ignored. See the paper for more details.

    The parameters of the environment are defined in parameters_intersection.py.
    The environment is built in a gym-like structure, with the methods 'reset' and 'step'

    Args:
        sim_params: Parameters that describe the simulation setup, the action space, and the reward function
        road_params: Parameters that describe the road geometry, rules, and properties of different vehicles
        gui_params (bool): GUI options.
        start_time (str): Optional label
    """
    def __init__(self, sim_params, road_params, gui_params=None, start_time=''):
        self.step_ = 0

        # Parameters road
        self.intersection_pos = road_params['intersection_position']
        self.nb_lanes = road_params['nb_lanes']
        self.lane_width = road_params['lane_width']
        self.ego_length = road_params['vehicles'][0]['length']
        self.ego_width = road_params['vehicles'][0]['width']
        self.car_length = road_params['vehicles'][1]['length']
        self.car_width = road_params['vehicles'][1]['width']
        self.stop_line = road_params['stop_line']

        # Parameters scenario
        self.max_steps = sim_params['max_steps']
        self.start_pos = sim_params['ego_start_position']
        self.end_pos = sim_params['ego_end_position']
        self.start_route = sim_params['ego_start_route']
        self.start_lane = sim_params['ego_start_lane']
        self.init_steps = sim_params['init_steps']
        self.adding_prob = sim_params['adding_prob']
        self.max_nb_vehicles = sim_params['nb_vehicles']
        self.idm_params = sim_params['idm_params']
        self.max_speed = road_params['speed_range'][1]
        self.min_speed = road_params['speed_range'][0]
        self.max_ego_speed = road_params['vehicles'][0]['maxSpeed']
        self.safety_check = sim_params['safety_check']

        # Parameters sensing
        self.nb_ego_states = 3
        self.nb_states_per_vehicle = 4
        self.sensor_range = sim_params['sensor_range']
        self.occlusion_dist = sim_params['occlusion_dist']
        self.sensor_nb_vehicles = sim_params['sensor_nb_vehicles']
        self.sensor_noise = sim_params['sensor_noise']
        self.sensor_max_speed_scale = self.max_speed

        # Parameters reward
        self.goal_reward = sim_params['goal_reward']
        self.collision_penalty = sim_params['collision_penalty']
        self.near_collision_penalty = sim_params['near_collision_penalty']
        self.near_collision_margin = sim_params['near_collision_margin']

        # GUI parameters
        self.use_gui = gui_params['use_gui'] if gui_params else False
        self.print_gui_info = gui_params['print_gui_info'] if gui_params else False
        self.draw_sensor_range = gui_params['draw_sensor_range'] if gui_params else False
        self.zoom_level = gui_params['zoom_level'] if gui_params else False
        self.fix_vehicle_colors = False
        self.gui_state_info = []
        self.gui_action_info = []
        self.gui_occlusions = []

        # Initialize state
        self.vehicles = []
        self.positions = np.zeros([self.max_nb_vehicles, 2])  # Defined as center of vehicle
        self.speeds = np.zeros([self.max_nb_vehicles, 2])
        self.accs = np.zeros([self.max_nb_vehicles])
        self.headings = np.zeros([self.max_nb_vehicles])
        self.ego_id = None
        self.previous_adding_node = None
        self.state_t0 = None
        self.state_t1 = None

        self.road = Road(road_params, start_time=start_time)
        self.road.create_road()

        if self.use_gui:
            sumo_binary = checkBinary('sumo-gui')
        else:
            sumo_binary = checkBinary('sumo')

        if sim_params['remove_sumo_warnings']:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start",
                         "--no-warnings"])
        else:
            traci.start([sumo_binary, "-c", self.road.road_path + self.road.name + ".sumocfg", "--start"])

    def reset(self, ego_at_intersection=False, sumo_ctrl=False):
        """
        Resets the intersection driving environment to a new random initial state.

        The ego vehicle starts in the south. A number of surrounding vehicles are added to random positions
        in the east and west, with randomly selected driver model parameters, e.g., desired speed.

        Args:
            ego_at_intersection (bool): If true, the ego vehicle starts close to the intersection (see the paper
                                        for description of specific tests)
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle.

        Returns:
            observation (ndarray): The observation of the traffic situation, according to the sensor model.
        """
        # Remove all vehicles
        # In some cases, the last vehicle may not yet have been inserted, and therefore cannot be deleted.
        # Then, run a few extra simulation steps.
        i = 0
        while not tuple(self.vehicles) == traci.vehicle.getIDList():
            if i > 5:
                raise Exception("All vehicles could not be inserted, and therefore not reset.")
            if len(self.vehicles) - len(traci.vehicle.getIDList()) > 1:
                warnings.warn("More than one vehicle missing during reset")
            traci.simulationStep()
            i += 1
        for veh in self.vehicles:
            traci.vehicle.remove(veh)
        traci.simulationStep()

        # Reset state
        self.vehicles = []
        self.positions = np.zeros([self.max_nb_vehicles, 2])
        self.speeds = np.zeros([self.max_nb_vehicles, 2])
        self.accs = np.zeros([self.max_nb_vehicles])
        self.headings = np.zeros([self.max_nb_vehicles])
        self.previous_adding_node = None
        self.state_t0 = None
        self.state_t1 = None

        # Add ego vehicle
        self.ego_id = 'veh' + '0'.zfill(int(np.ceil(np.log10(self.max_nb_vehicles))))   # Add leading zeros to number
        traci.vehicle.add(self.ego_id, self.start_route, typeID='truck', depart=None, departLane=0,
                          departPos='base', departSpeed=self.road.road_params['vehicles'][0]['maxSpeed'],
                          arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                          line='', personCapacity=0, personNumber=0)
        traci.vehicle.subscribe(self.ego_id, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, ANGLE])
        traci.simulationStep()
        if self.draw_sensor_range:
            traci.vehicle.highlight(self.ego_id, size=self.sensor_range)
        assert (len(traci.vehicle.getIDList()) == 1)
        self.vehicles = [self.ego_id]

        # Random init steps
        for i in range(self.init_steps - 1):
            self.step(action=0, sumo_ctrl=True)
        traci.vehicle.moveTo(self.ego_id, self.start_lane, 0)
        observation, reward, done, info = self.step(action=0, sumo_ctrl=True)

        # Turn off all internal lane changes and all safety checks for ego vehicle
        if not sumo_ctrl:
            if not self.safety_check:
                traci.vehicle.setSpeedMode(self.ego_id, 0)
                traci.vehicle.setLaneChangeMode(self.ego_id, 0)
        else:
            traci.vehicle.setSpeed(self.ego_id, -1)

        # Special case of ego vehicle starting close to the intersection
        if ego_at_intersection:
            traci.vehicle.setSpeed(self.ego_id, 0)
            self.speeds[0, 0] = 7
            traci.vehicle.moveTo(self.ego_id, self.start_lane, -self.start_pos + self.max_ego_speed - self.lane_width
                                 - self.occlusion_dist - self.speeds[0, 0])
            observation, reward, done, info = self.step(action=0)

        self.step_ = 0

        if self.use_gui:
            traci.gui.setZoom('View #0', self.zoom_level)
            self.draw_occlusion()
            if self.print_gui_info:
                self.print_state_info_in_gui(info='Start')

        return observation

    def step(self, action, action_info=None, sumo_ctrl=False):
        """
        Transition the environment to the next state with the specified action.

        Args:
            action (int): Action index, which is translated to an acceleration through
                          the Intelligent Driver Model (IDM).
                          0 - cruise, 1 - go, 2 - stop.
            action_info (dict): Used to display information in the GUI.
            sumo_ctrl (bool): For testing purposes, setting this True lets SUMO control the ego vehicle
                              (ignoring the surrounding vehicles).

        Returns:
            tuple, containing:
                observation (ndarray): Observation of the environment, given by the sensor model.
                reward (float): Reward of the current time step.
                done (bool): True if terminal state is reached, otherwise False
                info (dict): Dictionary with simulation information.
        """
        self.state_t0 = np.copy(self.state_t1)

        if self.use_gui and self.print_gui_info:
            self.print_action_info_in_gui(action, action_info)

        # Add more vehicles if possible
        nb_vehicles = len(self.vehicles)
        if nb_vehicles < self.max_nb_vehicles:
            if np.random.rand() < self.adding_prob:
                veh_id = 'veh' + str(nb_vehicles).zfill(int(np.ceil(np.log10(self.max_nb_vehicles))))  # Add leading zeros to number
                route_id = np.random.randint(4)
                node = 0 if route_id in [0, 1] else 2
                while node == self.previous_adding_node:   # To avoid adding a vehicle to an already occupied spot
                    route_id = np.random.randint(4)
                    node = 0 if route_id in [0, 1] else 2
                self.previous_adding_node = node
                speed = self.min_speed + np.random.rand() * (self.max_speed - self.min_speed)
                traci.vehicle.add(veh_id, 'route' + str(route_id), typeID='car', depart=None, departLane=0,
                                  departPos='base', departSpeed=speed,
                                  arrivalLane='current', arrivalPos='max', arrivalSpeed='current', fromTaz='', toTaz='',
                                  line='', personCapacity=0, personNumber=0)
                traci.vehicle.setMaxSpeed(veh_id, speed)
                traci.vehicle.setLaneChangeMode(veh_id, 0)   # Turn off lane changes
                traci.vehicle.subscribe(veh_id, [POSITION, LONG_SPEED, LAT_SPEED, LONG_ACC, ANGLE])  # position, speed
                self.vehicles.append(veh_id)

        # Take one simulation step
        acc = self.action_model(action, action_info)
        if self.use_gui and not self.fix_vehicle_colors:
            if action == 0:
                traci.vehicle.setColor(self.vehicles[0], (255, 255, 0))
            elif action == 1:
                traci.vehicle.setColor(self.vehicles[0], (0, 255, 0))
            elif action == 2:
                traci.vehicle.setColor(self.vehicles[0], (255, 0, 0))
            elif action == 3:
                traci.vehicle.setColor(self.vehicles[0], (0, 0, 255))
        if not sumo_ctrl:
            traci.vehicle.setSpeed(self.ego_id, self.speeds[0, 0] + acc)
        traci.simulationStep()
        self.step_ += 1

        # Get state information
        nb_digits = int(np.floor(np.log10(self.max_nb_vehicles))) + 1   # Number of digits in vehicle name.
        for veh in self.vehicles:
            i = int(veh[-nb_digits:])   # See comment above
            out = traci.vehicle.getSubscriptionResults(veh)
            self.speeds[i, 0] = out[LONG_SPEED]
            self.speeds[i, 1] = out[LAT_SPEED]
            self.accs[i] = out[LONG_ACC]
            self.headings[i] = np.mod(-out[ANGLE]+90, 360)/180*np.pi
            if i == 0:
                vehicle_length = self.ego_length
            else:
                vehicle_length = self.car_length
            self.positions[i, 0] = out[POSITION][0] - vehicle_length / 2 * np.cos(self.headings[i])
            self.positions[i, 1] = out[POSITION][1] - vehicle_length / 2 * np.sin(self.headings[i])
        if self.use_gui:
            non_occluded_vehicles = self.occlusion_model(
                self.positions[0, 1] + self.ego_length / 2 - self.intersection_pos[1])
            for veh in self.vehicles[1:]:
                if self.fix_vehicle_colors:
                    traci.vehicle.setColor(veh, (255, 255, 0))
                else:
                    i = int(veh[-nb_digits:])
                    color_speed_range = [10, 15]
                    speed_factor = (self.speeds[i, 0] - color_speed_range[0])/(color_speed_range[1] - color_speed_range[0])
                    if speed_factor > 1:
                        r = int(255*(1 - (speed_factor - 1)/4))
                        g = 0
                        b = int(255*(speed_factor - 1)/2)
                    else:
                        r = 255
                        g = 255 - max(int(255*speed_factor), 0)
                        b = 0
                    if non_occluded_vehicles[i-1]:
                        traci.vehicle.setColor(veh, (r, g, b))
                    else:
                        traci.vehicle.setColor(veh, (155,) * 3)
                    # traci.vehicle.setColor(veh, (r, g, b))

        if traci.simulation.getCollidingVehiclesNumber() > 0:
            warnings.warn('Collision between surrounding cars. This should normally not happen.')

        # Check terminal state
        info = {}
        done = False
        collision, near_collision, collision_info = self.collision_detection()
        if collision:
            done = True
            info['terminal_reason'] = str(collision_info)
        if self.step_ == self.max_steps:
            done = True
            info['terminal_reason'] = 'Max steps'
        if self.positions[0, 1] - self.intersection_pos[1] >= self.end_pos:
            done = True
            info['terminal_reason'] = 'Max dist'
        goal_reached = done and info['terminal_reason'] == 'Max dist'

        # Create observation and get reward
        self.state_t1 = copy.deepcopy([self.positions, self.speeds, self.headings, done])
        observation = self.sensor_model(self.state_t1)
        reward, reward_info = self.reward_model(goal_reached=goal_reached, collision=collision,
                                                near_collision=near_collision)
        info.update(reward_info)
        if self.use_gui and self.print_gui_info:
            self.print_state_info_in_gui(reward=reward, info=info)

        return observation, reward, done, info

    def collision_detection(self):
        """ Only works for this specific case with crossing traffic, simplified for speed of execution.
        Ignores vehicles turning to the south. """

        # Remove simple non-collision cases
        if self.intersection_pos[1] - (self.positions[0, 1] + self.ego_length/2) > self.lane_width \
                + self.near_collision_margin[1]:
            return False, False, 'Ego before_intersection'

        # Classification of vehicles
        atol = 3e-2
        east_heading = np.isclose(self.headings, 0, atol=atol, rtol=0) \
                       | np.isclose(self.headings, 2*np.pi, atol=atol, rtol=0)
        west_heading = np.isclose(self.headings, np.pi, atol=atol, rtol=0)
        north_turning = (np.pi/2 + atol <= self.headings) & (self.headings <= np.pi - atol)
        south_turning = (3*np.pi/2 + atol <= self.headings) & (self.headings <= 2*np.pi - atol)
        north_heading = np.isclose(self.headings, np.pi/2, atol=atol, rtol=0)
        south_heading = np.isclose(self.headings, 3*np.pi/2, atol=atol, rtol=0)
        assert (east_heading.astype(int) + west_heading + north_heading + south_heading + north_turning +
                south_turning == 1).all()

        dx = self.positions[1:, 0] - self.positions[0, 0]
        dx = np.insert(dx, 0, np.inf)

        # Collision
        # Crossing vehicles and turning to the north
        possib_col_idx = (np.abs(dx) < self.ego_width / 2 + self.car_length / 2) \
                         & (east_heading | west_heading | north_turning)
        col_idx = np.abs(self.positions[possib_col_idx, 1] - self.positions[0, 1]) \
                  < self.ego_length / 2 + self.car_width / 2
        id_cross = np.argwhere(possib_col_idx)[col_idx]
        # North bound vehicles
        possib_col_idx = (np.abs(dx) < self.ego_width / 2 + self.car_length / 2) & north_heading
        col_idx = np.abs(self.positions[possib_col_idx, 1] - self.positions[0, 1]) \
                  < self.ego_length / 2 + self.car_length / 2
        id_north = np.argwhere(possib_col_idx)[col_idx]
        ids = np.concatenate((id_cross, id_north))
        if len(ids) > 0:
            return True, False, ['collision', 'id: ' + str(np.squeeze(ids)) + ' pos: ' + str(self.positions[0, 1])]

        # Collision between time steps
        possib_col_idx_east = (dx < self.speeds[:, 0]) & (dx > 0) & east_heading
        if possib_col_idx_east.any():
            time_possib_collision_east = (self.positions[possib_col_idx_east, 0] - self.positions[0, 0])\
                                         / self.speeds[possib_col_idx_east, 0]
            ego_y_pos_at_possib_time_col_east = self.positions[0, 1] - self.speeds[0, 0]*time_possib_collision_east
            col_idx_east = np.abs(self.positions[possib_col_idx_east, 1] - ego_y_pos_at_possib_time_col_east) \
                           < self.ego_length / 2 + self.car_width / 2
            if col_idx_east.any():
                return True, False, ['collision',
                                     'id: ' + str(np.squeeze(np.argwhere(possib_col_idx_east)[col_idx_east])) +
                                     ' pos: ' + str(self.positions[0, 1])]
        possib_col_idx_west = (np.abs(dx) < self.speeds[:, 0]) & (dx < 0) & west_heading
        if possib_col_idx_west.any():
            time_possib_collision_west = np.abs(self.positions[possib_col_idx_west, 0] - self.positions[0, 0]) \
                                         / self.speeds[possib_col_idx_west, 0]
            ego_y_pos_at_possib_time_col_west = self.positions[0, 1] - self.speeds[0, 0] * time_possib_collision_west
            col_idx_west = np.abs(self.positions[possib_col_idx_west, 1] - ego_y_pos_at_possib_time_col_west) \
                           < self.ego_length / 2 + self.car_width / 2
            if col_idx_west.any():
                return True, False, ['collision',
                                     'id: ' + str(np.squeeze(np.argwhere(possib_col_idx_west)[col_idx_west]))  +
                                     ' pos: ' + str(self.positions[0, 1])]
        possib_near_col_idx_east = (dx < self.speeds[:, 0] - (self.near_collision_margin[0]+self.ego_width/2+self.car_length/2) ) \
                                   & (dx > - (self.near_collision_margin[0]+self.ego_width/2+self.car_length/2) ) \
                                   & east_heading
        if possib_near_col_idx_east.any():
            # x margin
            time_possib_near_collision_east = (dx[possib_near_col_idx_east] + self.ego_width/2 + self.car_length/2 +
                                               self.near_collision_margin[0]) / self.speeds[possib_near_col_idx_east, 0]
            ego_y_pos_at_possib_time_col_east = self.positions[0, 1] \
                                                - self.speeds[0, 0] * time_possib_near_collision_east
            near_col_idx_east_x = np.abs(self.positions[possib_near_col_idx_east, 1] - ego_y_pos_at_possib_time_col_east) \
                                  < self.ego_length / 2 + self.car_width / 2 + self.near_collision_margin[1]
            id_x = np.argwhere(possib_near_col_idx_east)[near_col_idx_east_x]
            # y margin
            time_possib_near_collision_east = dx[possib_near_col_idx_east] / self.speeds[possib_near_col_idx_east, 0]
            ego_y_pos_at_possib_time_col_east = self.positions[0, 1] \
                                                - self.speeds[0, 0] * time_possib_near_collision_east
            near_col_idx_east_y = np.abs(self.positions[possib_near_col_idx_east, 1] - ego_y_pos_at_possib_time_col_east) \
                                  < self.ego_length / 2 + self.car_width / 2 + self.near_collision_margin[1]
            id_y = np.argwhere(possib_near_col_idx_east)[near_col_idx_east_y]
            ids = np.concatenate((id_x, id_y))
            if len(ids):
                return False, True, ['near_collision', 'id: ' + str(np.squeeze(ids))
                                     + ' pos: ' + str(self.positions[0, 1])]
        possib_near_col_idx_west = (dx < self.near_collision_margin[0] + self.ego_width/2 + self.car_length/2) \
                                   & (dx > -self.speeds[:, 0] + self.near_collision_margin[0] + self.ego_width/2 + self.car_length/2) \
                                   & west_heading
        if possib_near_col_idx_west.any():
            # x margin
            time_possib_near_collision_west = (- dx[possib_near_col_idx_west] + self.ego_width/2 + self.car_length/2 +
                                               self.near_collision_margin[0]) / self.speeds[possib_near_col_idx_west, 0]
            ego_y_pos_at_possib_time_col_west = self.positions[0, 1] \
                                                - self.speeds[0, 0] * time_possib_near_collision_west
            near_col_idx_west_x = np.abs(self.positions[possib_near_col_idx_west, 1] - ego_y_pos_at_possib_time_col_west) \
                                  < self.ego_length / 2 + self.car_width / 2 + self.near_collision_margin[1]
            id_x = np.argwhere(possib_near_col_idx_west)[near_col_idx_west_x]
            # y margin
            time_possib_near_collision_west = np.abs(dx[possib_near_col_idx_west]) / self.speeds[possib_near_col_idx_west, 0]
            ego_y_pos_at_possib_time_col_west = self.positions[0, 1] \
                                                - self.speeds[0, 0] * time_possib_near_collision_west
            near_col_idx_west_y = np.abs(self.positions[possib_near_col_idx_west, 1] - ego_y_pos_at_possib_time_col_west) \
                                  < self.ego_length / 2 + self.car_width / 2 + self.near_collision_margin[1]
            id_y = np.argwhere(possib_near_col_idx_west)[near_col_idx_west_y]
            ids = np.concatenate((id_x, id_y))
            if len(ids):
                return False, True, ['near_collision', 'id: ' + str(np.squeeze(ids)) +
                                     ' pos: ' + str(self.positions[0, 1])]

        # Near collision
        # Crossing vehicles and turning to the north
        possib_near_col_idx = (np.abs(dx) < self.ego_width / 2 + self.car_length / 2 + self.near_collision_margin[0]) \
                              & (east_heading | west_heading | north_turning)
        near_col_idx = np.abs(self.positions[possib_near_col_idx, 1] - self.positions[0, 1]) \
                       < self.ego_length/2 + self.car_width/2 + self.near_collision_margin[1]
        id_cross = np.argwhere(possib_near_col_idx)[near_col_idx]
        # North bound vehicles
        possib_near_col_idx = (np.abs(dx) < self.ego_width / 2 + self.car_length / 2 + self.near_collision_margin[0]) \
                              & north_heading
        near_col_idx = np.abs(self.positions[possib_near_col_idx, 1] - self.positions[0, 1]) \
                       < self.ego_length / 2 + self.car_length / 2 + self.near_collision_margin[1]
        id_north = np.argwhere(possib_near_col_idx)[near_col_idx]
        ids = np.concatenate((id_cross, id_north))
        if len(ids) > 0:
            return False, True, ['near_collision', 'id: ' + str(np.squeeze(ids)) + ' pos: ' + str(self.positions[0, 1])]

        return False, False, ''

    def action_model(self, action, action_info=None):
        """
        Translate action into setpoint of IDM model.

        Args:
            action (int): Action index, which is translated to an acceleration through
                          the Intelligent Driver Model (IDM).
                          0 - cruise, 1 - go, 2 - stop, 3 - safety action.
            action_info (dict): Information about currently selected action.

        Returns:
            acc (float): Acceleration of the ego vehicle
        """
        a_min = self.idm_params['a_min']
        # If selected action is above normal range of actions, a "safety action" is selected.
        if action == self.nb_actions:   # Safety policy
            dt = 1
            ego_stopping_steps = int(self.speeds[0][0] // np.abs(a_min))
            ego_stopping_dist = sum((self.speeds[0][0] - (k + 1) * np.abs(a_min)) * dt
                                    for k in range(ego_stopping_steps))
            # if ego vehicle can stop before the intersection
            if self.positions[0][1] < self.intersection_pos[1] - self.stop_line - self.ego_length/2 - ego_stopping_dist:
                acc = self.idm_acc(self.intersection_pos[1] - self.stop_line, 0, self.idm_params)
            # else follow decision by agent
            else:
                action = action_info['original_action']
        if action == 0:   # Maintain speed
            acc = 0
        elif action == 1:   # Continue through intersection
            acc = self.idm_acc(None, None, self.idm_params)
        elif action == 2:   # Stop at intersection
            # If vehicle has already entered the intersection
            if self.positions[0][1] + self.ego_length/2 > self.intersection_pos[1] - self.stop_line:
                acc = a_min
            else:
                acc = self.idm_acc(self.intersection_pos[1] - self.stop_line, 0, self.idm_params)

        # Limit speed to physical limits
        if self.speeds[0][0] + acc > self.max_ego_speed:
            acc = self.max_ego_speed - self.speeds[0][0]
        elif self.speeds[0][0] + acc < 0:
            acc = 0 - self.speeds[0][0]

        return acc

    def idm_acc(self, pos_target, v_target, idm_params):
        """
        IDM model, see e.g. https://en.wikipedia.org/wiki/Intelligent_driver_model

        Used to let the ego vehicle either stop at the intersection or accelerate up to its desired speed.

        Args:
            pos_target (float): Distance to preceding vehicle
            v_target (float): Speed of preceding vehicle
            idm_params (dict): Parameters of the IDM model

        Returns:
            acc (float): Acceleration of the ego vehicle
        """
        a = idm_params['a']
        b = idm_params['b']
        s0 = idm_params['s0']
        T = idm_params['T']
        a_min = idm_params['a_min']
        a_max = idm_params['a_max']
        if pos_target is None:
            out = a * (1 - (self.speeds[0][0] / self.max_ego_speed) ** 4)
        else:
            approach_rate = np.max([self.speeds[0][0] - v_target, 0])   # Only allow positive approach rates
            out = a * (1 - (self.speeds[0][0] / self.max_ego_speed) ** 4 -
                       ((s0 + self.speeds[0][0] * T +
                         self.speeds[0][0] * approach_rate / (2 * np.sqrt(a * b))) /
                        (pos_target - (self.positions[0][1] + self.ego_length / 2))) ** 2)
        out = np.clip(out, a_min, a_max)
        return out

    def reward_model(self, goal_reached=False, collision=False, near_collision=False):
        """
        Reward model of the intersection environment.

        Args:
            goal_reached (bool): True if the ego vehicle reached the goal to the north of the intersection.
            collision (bool): True if a collision occurred.
            near_collision(bool): True if a near collision occurred.

        Returns:
            reward (float): Reward for the current environment step.
            info (dict): Information about what caused the reward.
        """
        info = {}
        reward = 0
        if goal_reached:
            reward = self.goal_reward
        elif collision:
            reward = self.collision_penalty
        elif near_collision:
            reward = self.near_collision_penalty
            info['near_collision'] = True
        return reward, info

    def sensor_model(self, state):
        """
        Sensor model of the ego vehicle.

        Creates an observation vector from the current state of the environment. All observations are normalized.
        Only surrounding vehicles that are not occluded and that are within the sensor range
        are included in the observation.

        Args:
            state (list): Current state of the environment.

        Returns:
            observation( (ndarray): Current observation of the environment.
        """
        # Vehicle in range and not in [0,0]
        vehicles_in_range = np.array([np.linalg.norm(state[0][1:] - state[0][0], axis=1) <= self.sensor_range,
                            np.logical_not(np.array([state[0][1:, 0] == 0, state[0][1:, 1] == 0]).all(0))]).all(0)
        non_occluded_vehicles = self.occlusion_model(self.positions[0, 1] + self.ego_length/2 - self.intersection_pos[1])
        observed_vehicles = vehicles_in_range & non_occluded_vehicles
        if np.sum(observed_vehicles) > self.sensor_nb_vehicles:
            warnings.warn('More vehicles within range than sensor can represent')

        observation = np.zeros(self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles)
        observation[0] = 2 * (state[0][0, 1] - (self.intersection_pos[1] + self.end_pos)) / \
                         (self.end_pos - self.start_pos) + 1   # Ego pos
        observation[1] = 2 * state[1][0, 0] / self.max_ego_speed - 1   # Long speed
        observation[2] = 1 if state[3] else -1   # Terminal state
        assert(self.nb_ego_states == 3)
        idx = 0
        for i, in_range in enumerate(observed_vehicles):
            if not in_range:
                continue
            internal_random_state = np.random.get_state()  # Quick fix for repeatability of episodes, improve later
            observation[3 + idx*4:5 + idx*4] = (state[0][i + 1] + np.random.normal(0, self.sensor_noise['pos'], 2)
                                                - self.intersection_pos) / self.sensor_range   # Pos
            observation[5 + idx*4] = 2 * (state[1][i + 1, 0] + np.random.normal(0, self.sensor_noise['speed'])) \
                                     / self.sensor_max_speed_scale - 1   # Speed
            observation[6 + idx*4] = 2 * (state[2][i + 1] + np.random.normal(0, self.sensor_noise['heading'])) \
                                     / (2*np.pi) - 1   # Heading
            np.random.set_state(internal_random_state)   # Second part of quick fix
            idx += 1
            if idx >= self.sensor_nb_vehicles:
                break
        # Default values for empty slots
        for i in range(idx, self.sensor_nb_vehicles):
            observation[3 + idx * 4] = -1
            observation[4 + idx * 4] = -1
            observation[5 + idx * 4] = -1
            observation[6 + idx * 4] = -1
            idx += 1
        assert(self.nb_states_per_vehicle == 4)

        return observation

    def occlusion_model(self, d):
        """
        Creates mask of observed and occluded vehicles.
        """
        if d > - (self.lane_width + self.occlusion_dist):   # Full visibility
            return np.ones(self.max_nb_vehicles-1, dtype=bool)
        else:
            d = np.abs(d)
            return np.abs(self.positions[1:, 0] + self.car_length/2*np.cos(self.headings[1:]) -
                          self.intersection_pos[0]) \
                   < self.lane_width / 2 + (d + self.lane_width / 2) * (self.lane_width / 2 + self.occlusion_dist)\
                   / (d - (self.lane_width + self.occlusion_dist))

    def print_state_info_in_gui(self, reward=None, info=None):
        """
        Prints information in the GUI.
        """
        for item in self.gui_state_info:
            traci.polygon.remove(item)
        dy = 15
        self.gui_state_info = ['Position: {0:.1f}'.format(self.positions[0, 1] - self.intersection_pos[1]),
                               'Speed: {0:.1f}'.format(self.speeds[0, 0]),
                               'Reward: ' + str(reward),
                                str(info),
                                'Step: ' + str(self.step_)]
        for idx, text in enumerate(self.gui_state_info):
            traci.polygon.add(text, [self.road.road_params['info_pos'],
                                     self.road.road_params['info_pos'] + [1, -idx*dy]], [0, 0, 0, 0])

    def draw_occlusion(self, width=200, height=200):
        """
        Draws the occluding objects in the GUI.
        """
        for item in self.gui_occlusions:
            traci.polygon.remove(item)
        self.gui_occlusions = [' ', '  ']
        rect = lambda x, y : [(x, y), (x + width, y), (x + width, y + height), (x, y + height)]
        left_occlusion_pos = np.array(self.intersection_pos) - [self.lane_width, self.lane_width] - \
                             [self.occlusion_dist, self.occlusion_dist] - [width, height]
        right_occlusion_pos = np.array(self.intersection_pos) - [-self.lane_width, self.lane_width] - \
                              [-self.occlusion_dist, self.occlusion_dist] - [0, height]
        traci.polygon.add(self.gui_occlusions[0], rect(*left_occlusion_pos), (150, 150, 150, 255), fill=True)
        traci.polygon.add(self.gui_occlusions[1], rect(*right_occlusion_pos), (150, 150, 150, 255), fill=True)

    def print_action_info_in_gui(self, action=None, action_info=None):
        """
        Prints information in the GUI.
        """
        if action == 0:
            action_str = 'cruise'
        elif action == 1:
            action_str = 'go'
        elif action == 2:
            action_str = 'stop'
        else:
            action_str = 'backup'
        for item in self.gui_action_info:
            traci.polygon.remove(item)
        dy = 15
        self.gui_action_info = ['Action: ' + action_str]
        traci.polygon.add(self.gui_action_info[0],
                          [self.road.road_params['info_pos'],  self.road.road_params['info_pos'] + [1, dy]],
                          [0, 0, 0, 0])
        if action_info is not None:
            # Uncomment below to print Q-values of all networks in GUI
            # if 'q_values_all_nets' in action_info:
            #     for i, row in enumerate(action_info['q_values_all_nets']):
            #         self.gui_action_info.append('                                   ' +
            #                                     '  | '.join(['{:6.3f}'.format(element) for element in row]))
            #         traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
            #                           self.road.road_params['action_info_pos'] + [1, -(i+4)*dy]], [0, 0, 0, 0])
            if 'q_values' in action_info:
                self.gui_action_info.append('                                  cruise  |    go      |   stop  ')
                traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
                                                             self.road.road_params['action_info_pos'] + [1, 1*dy]],
                                  [0, 0, 0, 0])
                self.gui_action_info.append('Q-values:                 ' +
                                            '  | '.join(['{:6.3f}'.format(element) for element
                                                         in action_info['q_values']]))
                traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
                                  self.road.road_params['action_info_pos'] + [1, 0]], [0, 0, 0, 0])

            if 'aleatoric_std_dev' in action_info:
                self.gui_action_info.append('Aleatoric std dev:   ' +
                                            '  | '.join(['{:6.3f}'.format(element) for element in
                                                         action_info['aleatoric_std_dev']]))
                traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
                                  self.road.road_params['action_info_pos'] + [1, -1*dy]], [0, 0, 0, 0])

            if 'epistemic_std_dev' in action_info:
                self.gui_action_info.append('Epistemic std dev:  ' +
                                            '  | '.join(['{:6.3f}'.format(element) for element in
                                                         action_info['epistemic_std_dev']]))
                traci.polygon.add(self.gui_action_info[-1], [self.road.road_params['action_info_pos'],
                                  self.road.road_params['action_info_pos'] + [1, -2*dy]], [0, 0, 0, 0])

    @property
    def nb_actions(self):
        return 3

    @property
    def nb_observations(self):
        return self.nb_ego_states + self.nb_states_per_vehicle * self.sensor_nb_vehicles
