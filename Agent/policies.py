import random
import numpy as np
from Agent.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.controller import Controller
from Agent.dynamic_map import DynamicMap
from Agent.actions import LaneAction


class Werling_Policy(object):
    
    def __init__(self, env):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        self.env = env

    def act(self, obs):

        self.dynamic_map.update_map_from_obs(obs, self.env)
        rule_trajectory, high_level_action = self.trajectory_planner.trajectory_update(self.dynamic_map)
        # Control
        trajectory = self.trajectory_planner.trajectory_update_CP(high_level_action, rule_trajectory)
        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
        action = np.array([control_action.acc, control_action.steering])

        return action

class Werling_Policy_Random(object):
    
    def __init__(self, env):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 
        self.env = env

    def act(self, obs):

        self.dynamic_map.update_map_from_obs(obs, self.env)
        rule_trajectory, high_level_action = self.trajectory_planner.trajectory_update(self.dynamic_map)
        high_level_action = random.randint(0, 8)
        # Control
        trajectory = self.trajectory_planner.trajectory_update_CP(high_level_action, rule_trajectory)
        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
        action = np.array([control_action.acc, control_action.steering])
        
        return action