import gym
import numpy as np
import gym_routing
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from Bisimulation.Bisimulation import Bisimulation


env = CarEnv_02_Intersection_fixed()
policy = Werling(env)

model = Bisimulation(env)
model.train_bisim_NNs(10000, env, load_step=0, policy=0) # 1 for random policy, 0 for werling

# model.test_Q_bisim(env, load_step)
