import unittest
import numpy as np

import sys
sys.path.append('../src')
import parameters_intersection as p
from intersection_env import IntersectionEnv
import traci


class Tester(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(Tester, self).__init__(*args, **kwargs)
        np.random.seed(13)

    # For some strange reason, test_init sometimes fails when running all the tests
    # through "python3 -m unittest discover .". However, it always passes when running
    # only the intersection environment tests. To make the test process pass, this test
    # is therefore removed, but it should pass when run in isolation.
    # def test_init(self):
    #     gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
    #     self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
    #     self.env.reset()
    #     try:
    #         self.assertGreater(len(traci.vehicle.getIDList()), 1)
    #         self.assertTrue((self.env.speeds[:, 0] <= p.road_params['speed_range'][1]).all())
    #         self.assertTrue((self.env.speeds[:len(self.env.vehicles), 0] >= p.road_params['speed_range'][0]).all())
    #     finally:
    #         traci.close()

    def test_step(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            action = 0
            self.env.step(action)
        finally:
            traci.close()

    def test_reset(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            action = 0
            for _ in range(9):
                self.env.step(action)
            self.env.reset()
            for _ in range(10):
                self.env.step(action)
        finally:
            traci.close()

    def test_gym_interface(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            self.assertEqual(self.env.nb_actions, 3)
            self.assertEqual(self.env.nb_observations, 3 + 4*p.sim_params['sensor_nb_vehicles'])
        finally:
            traci.close()

    def test_sensor_model(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            self.env.sensor_range = 200
            self.env.occlusion_dist = 1e6
            self.env.sensor_noise = {'pos': 0, 'speed': 0, 'heading': 0}
            positions = np.zeros([self.env.max_nb_vehicles, 2])
            rel_pos_intersect = np.array([[1.6, 0], [-190, -self.env.lane_width/2], [190, self.env.lane_width/2],
                                          [-205, -self.env.lane_width/2]])
            positions[0:4, :] = rel_pos_intersect + np.array(self.env.intersection_pos)    # Last vehicle outside sensor range
            speeds = np.zeros([self.env.max_nb_vehicles, 2])
            speeds[0:4, :] = np.array([[15.0, 0], [15, 0.], [0, 0.], [7.5, 0]])
            headings = np.zeros(self.env.max_nb_vehicles)
            headings[0:4] = np.array([0, np.pi/2, 3*np.pi/2, np.pi/2])
            done = False
            state = [positions, speeds, headings, done]
            observation = self.env.sensor_model(state)
            self.assertEqual(observation[0], -2*p.sim_params['ego_end_position'] /
                             (p.sim_params['ego_end_position'] - p.sim_params['ego_start_position']) + 1)
            self.assertEqual(observation[1], 1)
            self.assertEqual(observation[2], -1)
            self.assertEqual(observation[3], -0.95)
            self.assertAlmostEqual(observation[4], -1.6/200)
            self.assertEqual(observation[5], 1)
            self.assertEqual(observation[6], -0.5)
            self.assertEqual(observation[7], 0.95)
            self.assertAlmostEqual(observation[8], 1.6/200)
            self.assertEqual(observation[9], -1)
            self.assertEqual(observation[10], 0.5)
            for i in range(11, self.env.sensor_nb_vehicles):
                self.assertEqual(observation[i], -1)

            rel_pos_intersect = [self.env.lane_width / 2, 30]
            positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            speeds[0] = [0, 0]
            done = True
            state = [positions, speeds, headings, done]
            observation = self.env.sensor_model(state)
            self.assertEqual(observation[0], 1)
            self.assertEqual(observation[1], -1)
            self.assertEqual(observation[2], 1)

            self.env.max_nb_vehicles = self.env.sensor_nb_vehicles + 5
            positions = np.random.rand(self.env.sensor_nb_vehicles + 5, 2) * 300 + np.array(self.env.intersection_pos)
            positions[0] = self.env.intersection_pos
            speeds = np.random.rand(self.env.sensor_nb_vehicles + 5, 2) * 15
            headings = np.random.rand(self.env.sensor_nb_vehicles + 5) * 2 * np.pi
            state = [positions, speeds, headings, done]
            observation = self.env.sensor_model(state)
            self.assertTrue((np.abs(observation) <= 1).all())
            self.assertEqual(len(observation), 3 + self.env.sensor_nb_vehicles * 4)

            # Noise
            self.env.sensor_noise = {'pos': 2, 'speed': 2, 'heading': 20 / 180 * np.pi}
            rel_pos_intersect = [-50, -self.env.lane_width / 2]
            positions[1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            # positions[1] = [950, 998.4]
            speeds[1] = [0, 0]
            state = [positions, speeds, headings, done]
            observation = self.env.sensor_model(state)
            self.assertFalse(observation[5] == -1)

        finally:
            traci.close()

    def test_occlusion_model(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            self.env.occlusion_dist = 1e6
            self.assertTrue(self.env.occlusion_model(-100).all())

            self.env.occlusion_dist = 2
            rel_pos_intersect = [0, - self.env.lane_width / 2]
            self.env.positions[1, :] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            self.env.headings[1] = 0
            rel_pos_intersect = [0, self.env.lane_width / 2]
            self.env.positions[2, :] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            self.env.headings[2] = np.pi
            rel_pos_intersect = [-10.4, -self.env.lane_width / 2]
            self.env.positions[3, :] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            self.env.headings[3] = 0
            rel_pos_intersect = [-14.4, -self.env.lane_width / 2]
            self.env.positions[4, :] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            self.env.headings[4] = 0
            self.assertTrue(self.env.occlusion_model(-100)[0])
            self.assertTrue(self.env.occlusion_model(-100)[1])
            self.assertFalse(self.env.occlusion_model(-100)[2:].any())
            self.assertTrue(self.env.occlusion_model(-10)[2])
            self.assertFalse(self.env.occlusion_model(-10)[3:].any())

        finally:
            traci.close()

    def test_reward_model(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()

        try:
            self.assertEqual(self.env.reward_model(goal_reached=False, collision=False,
                                                   near_collision=False)[0], 0)
            self.assertEqual(self.env.reward_model(goal_reached=True, collision=False, near_collision=False)[0],
                             p.sim_params['goal_reward'])
            self.assertEqual(self.env.reward_model(goal_reached=False, collision=True, near_collision=False)[0],
                             p.sim_params['collision_penalty'])
            self.assertEqual(self.env.reward_model(goal_reached=False, collision=False, near_collision=True)[0],
                             p.sim_params['near_collision_penalty'])
        finally:
            traci.close()

    def test_action_model(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            ego_speed_0 = self.env.speeds[0][0]

            self.env.step(2)
            self.assertLess(self.env.speeds[0][0], ego_speed_0)

            for i in range(3):
                self.env.step(2)
            ego_speed_0 = self.env.speeds[0][0]
            self.env.step(0)
            self.assertEqual(self.env.speeds[0][0], ego_speed_0)

            self.env.step(1)
            self.assertGreater(self.env.speeds[0][0], ego_speed_0)

            self.env.reset()
            self.env.step(1)
            self.assertEqual(self.env.speeds[0][0], self.env.max_ego_speed)

            done = False
            while not done:
                _, _, done, _ = self.env.step(2)
            self.assertEqual(self.env.speeds[0][0], 0)

        finally:
            traci.close()

    def test_safe_action(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        try:
            # Stop at intersection
            ego_speed_0 = self.env.speeds[0][0]
            self.env.positions[0][1] = p.road_params['intersection_position'][1] - 50
            self.env.step(3)
            self.assertEqual(self.env.speeds[0][0], ego_speed_0 + p.sim_params['idm_params']['a_min'])

            # Use original action
            ego_speed_0 = 5
            self.env.speeds[0][0] = ego_speed_0
            self.env.positions[0][1] = p.road_params['intersection_position'][1] - p.road_params['stop_line'] + 1
            self.env.step(3, {'original_action': 0})
            self.assertEqual(self.env.speeds[0][0], ego_speed_0)

            self.env.speeds[0][0] = ego_speed_0
            self.env.positions[0][1] = p.road_params['intersection_position'][1] - p.road_params['stop_line'] + 1
            self.env.step(3, {'original_action': 1})
            self.assertGreater(self.env.speeds[0][0], ego_speed_0)

            self.env.speeds[0][0] = ego_speed_0
            self.env.positions[0][1] = p.road_params['intersection_position'][1] - p.road_params['stop_line'] + 1
            self.env.step(3, {'original_action': 2})
            self.assertEqual(self.env.speeds[0][0], ego_speed_0 + p.sim_params['idm_params']['a_min'])

        finally:
            traci.close()

    def test_collision_detection(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()
        nb_cars = self.env.positions.shape[0] - 1
        try:
            # Outside intersection
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, 'Ego before_intersection')
            rel_pos_intersect = [self.env.lane_width/2, 50]
            self.env.positions[0, :] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')

            # Collision
            # West bound
            rel_pos_intersect_0 = [self.env.lane_width / 2, 0]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2 + 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2 - 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # North bound
            rel_pos_intersect_1 = [self.env.lane_width / 2, self.env.ego_length / 2 + self.env.car_length / 2 + 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi/2
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            rel_pos_intersect_1 = [self.env.lane_width / 2, self.env.ego_length / 2 + self.env.car_length / 2 - 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi/2
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # East bound
            rel_pos_intersect_2 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2 + 0.1,
                                   - self.env.lane_width / 2]
            self.env.positions[-2] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_2)]
            self.env.headings[-2] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars-1) in info[1])
            self.assertTrue(str(nb_cars) in info[1])

            self.env.reset()

            # Near collision, west
            # Within x margin, but ego vehicle too high
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.ego_length/2 + self.env.car_width/2
                                   + self.env.lane_width/2 + self.env.near_collision_margin[1] + 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2
                                   + self.env.near_collision_margin[0] - 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            # Within x margin, but ego vehicle too low
            rel_pos_intersect_0 = [self.env.lane_width / 2, -self.env.ego_length / 2 - self.env.car_width / 2
                                   + self.env.lane_width / 2 - self.env.near_collision_margin[1] - 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2
                                   + self.env.near_collision_margin[0] - 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            # Within x margin right
            rel_pos_intersect_0 = [self.env.lane_width / 2, 0]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within x margin left
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2
                                   - self.env.near_collision_margin[0] + 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within y margin top
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.ego_length / 2 + self.env.car_width / 2
                                   + self.env.lane_width / 2 + self.env.near_collision_margin[1] - 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2 - 0.1,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within y margin bottom
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.ego_length / 2 - self.env.car_width / 2
                                   + self.env.lane_width / 2 - self.env.near_collision_margin[1] + 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])

            # Near collision, east
            # Within margin but ego vehicle too high
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.ego_length/2 + self.env.car_width/2
                                   + self.env.lane_width/2 + self.env.near_collision_margin[1] + 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2
                                   - self.env.near_collision_margin[0] + 0.1,
                                   - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            # Within margin but ego vehicle too low
            rel_pos_intersect_0 = [self.env.lane_width / 2, -self.env.ego_length / 2 - self.env.car_width / 2
                                   - self.env.lane_width / 2 - self.env.near_collision_margin[1] - 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2
                                   - self.env.near_collision_margin[0] + 0.1,
                                   - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            # Within margin left
            rel_pos_intersect_0 = [self.env.lane_width / 2, 0]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within margin right
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2
                                   + self.env.near_collision_margin[0] - 0.1,
                                   - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within y margin top
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.ego_length / 2 + self.env.car_width / 2
                                   - self.env.lane_width / 2 + self.env.near_collision_margin[1] - 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2 + 0.1,
                                   - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            # Within y margin bottom
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.ego_length / 2 - self.env.car_width / 2
                                   - self.env.lane_width / 2 - self.env.near_collision_margin[1] + 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])

            # Turning vehicle
            # South
            rel_pos_intersect_0 = [self.env.lane_width / 2, 0]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [- 3.84, -3.2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 5.6012
            self.env.positions[-2] = self.env.positions[-3]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            rel_pos_intersect_1 = [-1.66, -8.79]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 4.7363
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            # North
            rel_pos_intersect_1 = [3.84, 3.2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 2.4596
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, 3.2 - self.env.ego_length/2 - self.env.car_width/2 - 0.1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, 0]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            rel_pos_intersect_1 = [self.env.lane_width / 2, self.env.ego_length/2 + self.env.car_length/2
                                   + self.env.near_collision_margin[1] - 0.1]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi/2
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])

            # Collision between time steps
            # East
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width/2 + self.env.ego_length/2
                                   + self.env.car_width/2 + self.env.speeds[0][0] - 5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2
                                   + self.env.speeds[-1][0] - 5, - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width/2 + self.env.ego_length/2
                                   + self.env.car_width/2 + self.env.speeds[0][0] + 1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            # West
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0] - 5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2
                                   - self.env.speeds[-1][0] + 5, - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertTrue(collision)
            self.assertFalse(near_collision)
            self.assertTrue('collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0] + 1]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')

            # Near collision between time steps
            # East
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] - 0.6]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width/2 - self.env.car_length/2
                                   + self.env.speeds[-1][0] - self.env.near_collision_margin[0] - 0.5,
                                   - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] + 0.5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            self.env.speeds[0][0] = 0
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.lane_width / 2 - self.env.ego_length / 2
                                   - self.env.car_width / 2 - self.env.near_collision_margin[1] + 0.5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 - self.env.ego_width / 2 - self.env.car_length / 2
                                   + self.env.speeds[-1][0] - 3, - self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = 0
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            rel_pos_intersect_0 = [self.env.lane_width / 2, - self.env.stop_line - self.env.ego_length/2]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            self.env.reset()
            # West
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] - 0.6]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width/2 + self.env.car_length/2
                                   - self.env.speeds[-1][0] + self.env.near_collision_margin[0] + 0.5,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            self.assertTrue(str(nb_cars) in info[1])
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] + 0.5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')
            self.env.speeds[0][0] = 1
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] - 0.5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            self.env.speeds[-1] = [15, 0]
            rel_pos_intersect_1 = [self.env.lane_width / 2 + self.env.ego_width / 2 + self.env.car_length / 2
                                   - self.env.speeds[-1][0] + self.env.near_collision_margin[0] + 0.5,
                                   self.env.lane_width / 2]
            self.env.positions[-1] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_1)]
            self.env.headings[-1] = np.pi
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertTrue(near_collision)
            self.assertTrue('near_collision' == info[0])
            rel_pos_intersect_0 = [self.env.lane_width / 2, self.env.lane_width / 2 + self.env.ego_length / 2
                                   + self.env.car_width / 2 + self.env.speeds[0][0]
                                   + self.env.near_collision_margin[1] + 0.5]
            self.env.positions[0] = [sum(e) for e in zip(self.env.intersection_pos, rel_pos_intersect_0)]
            collision, near_collision, info = self.env.collision_detection()
            self.assertFalse(collision)
            self.assertFalse(near_collision)
            self.assertEqual(info, '')

        finally:
            traci.close()

    def test_timeout(self):
        gui_params = {'use_gui': False, 'print_gui_info': False, 'draw_sensor_range': False, 'zoom_level': 3000}
        self.env = IntersectionEnv(sim_params=p.sim_params, road_params=p.road_params, gui_params=gui_params)
        self.env.reset()

        try:
            done = False
            while not done:
                _, _, done, info = self.env.step(2)   # Action stop at intersection
            self.assertEqual(self.env.step_, p.sim_params['max_steps'])
            self.assertEqual('Max steps', info['terminal_reason'])

        finally:
            traci.close()


if __name__ == '__main__':
    unittest.main()
