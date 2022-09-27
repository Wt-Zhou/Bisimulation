import gym
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from TestScenario import CarEnv_02_Intersection, CarEnv_02_Intersection_fixed
from Bisimulation.Bisimulation import Bisimulation
from Agent.policies import Werling_Policy, Werling_Policy_Random

env = CarEnv_02_Intersection_fixed()
policy = Werling_Policy(env)

model = Bisimulation(env)
model.train_bisim_NNs(10000, env, load_step=0, policy=0) # 1 for random policy, 0 for werling

model.test_Q_bisim(env, 0, policy)
