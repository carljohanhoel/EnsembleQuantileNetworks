import os


class Road(object):
    """
    This class creates the road definition xml files that are used by SUMO.

    A duplicate road is created for the ego vehicle, which makes the driver models of the other
    vehicles ignore the ego vehicle. This is just a hack to avoid that the driver models of the other
    vehicles brake due to the ego vehicle. The motivation for this feature is described in the paper.

    Args:
        road_params (dict): Parameters of the road.
        road_path (str): Path to store the xml files
        start_time (str): Optional identifier string that is added to the file names.:
    """

    def __init__(self, road_params, road_path='../road/', start_time=''):
        self.road_params = road_params
        self.road_path = road_path
        self.name = self.road_params['name'] + '_' + start_time

        if not os.path.isdir(self.road_path):
            os.mkdir(self.road_path)

    def nodes(self):
        with open(self.road_path + self.name + '.nod.xml', 'w') as text_file:
            text_file.writelines(['<nodes>\n'])
            for idx, node in enumerate(self.road_params['nodes']):
                text_file.writelines(['   <node id="' + str(idx) + '" x="' + str(node[0]) + '" y="' + str(node[1]) +
                                      '" />\n'])
            for idx, node in enumerate(self.road_params['nodes']):
                text_file.writelines(['   <node id="ego_' + str(idx) + '" x="' + str(node[0]) + '" y="' + str(node[1]) +
                                      '" />\n'])
            text_file.writelines(['</nodes>\n'])

    def edges(self):
        with open(self.road_path + self.name + '.edg.xml', 'w') as text_file:
            text_file.writelines(['<edges>\n'])
            for start_idx, node_edges in enumerate(self.road_params['edges']):
                for stop_idx, edge in enumerate(node_edges):
                    if edge:
                        text_file.writelines(['   <edge from="' + str(start_idx) +
                                              '" id="L' + str(start_idx) + str(stop_idx) +
                                              '" to="' + str(stop_idx) +
                                              '" numLanes="' + str(self.road_params['nb_lanes']) +
                                              '" width="' + str(self.road_params['lane_width']) +
                                              '" speed="' + str(self.road_params['max_road_speed']) +
                                              '" priority="' + str(
                            self.road_params['priority'][start_idx, stop_idx]) +
                                              '" />\n'])
                        text_file.writelines(['   <edge from="ego_' + str(start_idx) +
                                              '" id="ego_L' + str(start_idx) + str(stop_idx) +
                                              '" to="ego_' + str(stop_idx) +
                                              '" numLanes="' + str(self.road_params['nb_lanes']) +
                                              '" width="' + str(self.road_params['lane_width']) +
                                              '" speed="' + str(self.road_params['max_road_speed']) +
                                              '" priority="' + str(
                            self.road_params['priority'][start_idx, stop_idx]) +
                                              '" />\n'])
            text_file.writelines(['</edges>\n'])

    def routes(self):
        with open(self.road_path + self.name + '.rou.xml', 'w') as text_file:
            text_file.writelines(['<routes>\n'])
            for idx, vehicle in enumerate(self.road_params['vehicles']):
                text_file.writelines(['   <vType '])
                for key, value in vehicle.items():
                    text_file.writelines([key + '="' + str(value) + '" '])
                text_file.writelines(['/>\n'])
            for route_name, route in enumerate(self.road_params['routes']):
                edges_string = ''
                edges_string_ego = ''
                for i in range(len(route) - 1):
                    edges_string += 'L' + str(route[i]) + str(route[i + 1]) + ' '
                    edges_string_ego += 'ego_L' + str(route[i]) + str(route[i + 1]) + ' '
                edges_string = edges_string[:-1]
                edges_string_ego = edges_string_ego[:-1]
                text_file.writelines(
                    ['   <route id="route' + str(route_name) + '" edges="' + edges_string + '"/>\n'])
                text_file.writelines(
                    ['   <route id="ego_route' + str(route_name) + '" edges="' + edges_string_ego + '"/>\n'])
            text_file.writelines(['</routes>\n'])

    def connections(self):
        with open(self.road_path + self.name + '.con.xml', 'w') as text_file:
            text_file.writelines(['<connections>\n'])
            for route_name, route in enumerate(self.road_params['routes']):
                edge_start = 'L' + str(route[0]) + str(route[1])
                edge_stop = 'L' + str(route[1]) + str(route[2])
                text_file.writelines(['   <connection from="' + edge_start + '" to="' + edge_stop +
                                              '" fromLane="0" toLane="0' +
                                              '" />\n'])
                text_file.writelines(['   <connection from="ego_' + edge_start + '" to="ego_' + edge_stop +
                                      '" fromLane="0" toLane="0' +
                                      '" />\n'])
            text_file.writelines(['</connections>\n'])

    def config(self):
        with open(self.road_path + self.name + '.sumocfg', 'w') as text_file:
            text_file.writelines(['<configuration>\n'])
            text_file.writelines(['   <input>\n'])
            text_file.writelines(['      <net-file value="' + self.name + '.net.xml"/>\n'])
            text_file.writelines(['      <route-files value="' + self.name + '.rou.xml"/>\n'])
            text_file.writelines(['      <gui-settings-file value="' + self.name + '.settings.xml"/>\n'])
            text_file.writelines(['   </input>\n'])
            text_file.writelines(['   <time>\n'])
            text_file.writelines(['      <begin value="0"/>\n'])
            text_file.writelines(['      <end value="1e15"/>\n'])
            text_file.writelines(['   </time>\n'])
            text_file.writelines(['   <processing>\n'])
            text_file.writelines(['      <lanechange.duration value="' + str(self.road_params['lane_change_duration']) +
                                  '"/>\n'])
            text_file.writelines(['      <lanechange.overtake-right value="' + str(self.road_params['overtake_right']) +
                                  '"/>\n'])
            text_file.writelines(['      <emergencydecel.warning-threshold value="' +
                                  str(self.road_params['emergency_decel_warn_threshold']) + '"/>\n'])
            text_file.writelines(['      <collision.action value="' + str(self.road_params['collision_action']) +
                                  '"/>\n'])
            text_file.writelines(['      <no-step-log value="' + str(self.road_params['no_display_step']) + '"/>\n'])
            text_file.writelines(['      <collision.check-junctions value="true"/>\n'])
            text_file.writelines(['   </processing>\n'])
            text_file.writelines(['</configuration>\n'])

    def gui_settings(self):
        with open(self.road_path + self.name + '.settings.xml', 'w') as text_file:
            text_file.writelines(['<viewsettings>\n'])
            text_file.writelines(['   <viewport x="' + str(self.road_params['view_position'][0]) + '" y="' +
                                  str(self.road_params['view_position'][1]) + '" zoom="' +
                                  str(self.road_params['zoom']) + '"/>\n'])
            text_file.writelines(['   <delay value="' + str(self.road_params['view_delay']) + '"/>\n'])
            text_file.writelines(['   <scheme name="real world"/>\n'])
            text_file.writelines(['   <scheme>\n'])
            text_file.writelines(['      <polys polyName_show="1" polyName_color="0,0,0" polyName_size="100.00"/>\n'])
            text_file.writelines(['   </scheme>\n'])
            text_file.writelines(['</viewsettings>\n'])

    def create_road(self):
        """ Creates the xml files that are needed by SUMO. """
        self.nodes()
        self.edges()
        self.routes()
        self.connections()
        self.config()
        self.gui_settings()

        os.system('netconvert --node-files=' + self.road_path + self.name + '.nod.xml ' +
                  '--edge-files=' + self.road_path + self.name + '.edg.xml ' +
                  '--output-file=' + self.road_path + self.name + '.net.xml ' +
                  '--opposites.guess=true')
