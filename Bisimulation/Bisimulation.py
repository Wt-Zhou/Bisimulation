import numpy as np


class Bisimulation(object):

    def __init__(self, env):
        self.data_scenario1 = []
        self.data_scenario2 = []
        self.discount = 0.99
        return None
    
    def collect_data_for_scenario(self, obs, reward, next_obs, done, scenario_id):
        if scenario_id == 1:
            self.data_scenario1.append(obs, reward, next_obs, done)


    def calcualte_original_bisimulation(self, obs1, obs2):        
        r_dist = math.abs(self.calculate_reward(obs1) - self.calculate_reward(obs2))
        mu1 = self.calculate_mu_from_data(obs1)
        mu2 = self.calculate_mu_from_data(obs2)
        sigma1 = self.calculate_sigma_from_data(obs1)
        sigma2 = self.calculate_sigma_from_data(obs2)
        
        transition_dist = math.sqrt((mu1 - mu2).pow(2) + (sigma1 - sigma2).pow(2))
        d = r_dist + self.discount * transition_dist
        return d
    
    def calcualte_improved_bisimulation(self, obs1, obs2):        
        
        return d
    
    def calculate_mu_from_data(self, scenario_id):
        
        return mu
    
