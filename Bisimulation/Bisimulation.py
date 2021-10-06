import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from Bisimulation.transition_model import make_transition_model

from Agent.model import dqn_model, bootstrap_model
from Agent.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.controller import Controller
from Agent.dynamic_map import DynamicMap
from Agent.actions import LaneAction

class Bisimulation(object):

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        state_space_dim,
        transition_model_type,
        env,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        decoder_lr=0.0005,
        decoder_weight_lambda=0.0,
        bisim_coef=0.5
    ):
        self.device = device
        self.discount = discount
        self.bisim_coef = bisim_coef
        self.state_space_dim = state_space_dim
        self.action_shape = action_shape
        self.env = env

        # Transition Model
        self.transition_model_type = transition_model_type
        self.transition_model = make_transition_model(
            transition_model_type, state_space_dim, action_shape
        ).to(device)

        # Reward Model
        self.reward_decoder = nn.Sequential(
        nn.Linear(state_space_dim + action_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)).to(device)

        self.reward_decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # Q-value
        self.Rtree = RTree()

        # Data 
        self.replay_buffer = ReplayBuffer()


    def train_bisim_NNs(self, env):


    
    def update_transition_reward_model(self, obs, action, next_obs, reward,  step):
        obs_with_action = torch.cat([obs, action], dim=1)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(obs_with_action)
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        diff = (pred_next_latent_mu - next_obs.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        # L.log('train_ae/transition_loss', loss, step)

        pred_next_reward = self.reward_decoder(obs_with_action)
        reward_loss = F.mse_loss(pred_next_reward, reward)
        # print("pred_next_reward",pred_next_reward)
        # print("reward",reward)
        # print("reward_loss",reward_loss)
        total_loss = loss + reward_loss
        return total_loss,loss,reward_loss

    def update(self, replay_buffer, step):
        obs, action, _, reward, next_obs, not_done = replay_buffer.sample()
        # L.log('train/batch_reward', reward.mean(), step)

        transition_reward_loss,loss,reward_loss = self.update_transition_reward_model(obs, action, next_obs, reward, step)
        total_loss = transition_reward_loss
        self.reward_decoder_optimizer.zero_grad()
        total_loss.backward()
        self.reward_decoder_optimizer.step()

        with open("Reward_loss.txt", 'a') as fw:
            fw.write(str(loss.detach().cpu().numpy())) 
            fw.write(", ")
            fw.write(str(reward_loss.detach().cpu().numpy())) 
            fw.write("\n")
            fw.close()    

        print("[World_Model] : Updated all models! Step:",step)

    def get_reward_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)

        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.reward_decoder(obs_with_action)

    def get_trans_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action)
        action = torch.as_tensor(np_action, device=self.device)
        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.transition_model(obs_with_action)

    def calculate_bisimulation_pess(self, state_corner, state_normal, action_normal):
        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], state_corner)
        state_corner = torch.as_tensor(np_obs, device=self.device).float()
        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], state_normal)
        state_normal = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((1, 1), dtype=np.float32)
        np.copyto(np_action[0], action_normal)
        action_normal = torch.as_tensor(np_action, device=self.device)
        with torch.no_grad():
            bisim_for_corner_action = []
            for action in self.action_shape:
                np_action = np.empty((1, 1), dtype=np.float32)
                np.copyto(np_action[0], action)
                action = torch.as_tensor(np_action, device=self.device)

                obs_with_action = torch.cat([state_normal, action_normal], dim=1)
                normal_reward = self.reward_decoder(obs_with_action)

                obs_with_action = torch.cat([state_corner, action], dim=1)
                corner_reward = self.reward_decoder(obs_with_action)
                r_dist = F.smooth_l1_loss(normal_reward, corner_reward, reduction='none')

                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([state_normal, action_normal], dim=1))
                pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(torch.cat([state_corner, action], dim=1))

                transition_dist = torch.sqrt((pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2))
                bisim_for_corner_action.append(r_dist + self.discount * transition_dist)

        
        max_action = bisim_for_corner_action.index(max(bisim_for_corner_action))
        return bisim_for_corner_action[max_action], max_action, r_dist, transition_dist

    def calculate_bisimulation_optimal(self, state_corner, state_normal, action_normal):
        with torch.no_grad():
            bisim_for_corner_action = []
            for action in self.action_shape:
                obs_with_action = state_normal.append(action_normal)
                normal_reward = self.reward_decoder(obs_with_action)

                obs_with_action = state_corner.append(action)
                corner_reward = self.reward_decoder(obs_with_action)
                r_dist = F.smooth_l1_loss(normal_reward, corner_reward, reduction='none')

                pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([state_normal, action_normal], dim=1))
                pred_next_latent_mu2, pred_next_latent_sigma2 = self.transition_model(torch.cat([state_corner, action], dim=1))

                transition_dist = torch.sqrt((pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) + (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2))
                bisim_for_corner_action.append(r_dist + self.discount * transition_dist)

        
        min_action = bisim_for_corner_action.index(min(bisim_for_corner_action))
        return bisim_for_corner_action[min_action], min_action
            
    def save(self, model_dir, step):
        torch.save(
            self.reward_decoder.state_dict(),
            '%s/reward_decoder_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.transition_model.state_dict(),
            '%s/transition_model%s.pt' % (model_dir, step)
        )

    def load(self, model_dir, step):

        self.reward_decoder.load_state_dict(
            torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
        )
        self.transition_model.load_state_dict(
            torch.load('%s/transition_model%s.pt' % (model_dir, step))
        )


class RTree(object):

    def __init__(self, new_count=True):
        if new_count == True:
            if osp.exists("visited_state_value.txt"):
                os.remove("visited_state_value.txt")
            if osp.exists("state_index.dat"):
                os.remove("state_index.dat")
                os.remove("state_index.idx")

            # _setup_data_saving 
            self.visited_state_counter = 0
        else:
            self.visited_state_value = np.loadtxt("visited_value.txt")
            self.visited_state_value = self.visited_state_value.tolist()
            self.visited_state_counter = len(self.visited_state_value) 
            print("Loaded Save Rtree, len:",self.visited_state_counter)
        obs_dimension = 16
        self.visited_state_tree_prop = rindex.Property()
        self.visited_state_tree_prop.dimension = obs_dimension+1
        # self.visited_state_dist = np.array([[1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5,1, 1, 0.5, 0.5,1, 1, 0.5, 0.5, 0.1]])#, 10, 0.3, 3, 1, 0.1]])
        self.visited_state_dist = np.array([[2, 2, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        # self.visited_state_dist = np.array([[2, 2, 1.0, 1.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        self.visited_state_tree = rindex.Index('state_index',properties=self.visited_state_tree_prop)

        self.visited_value_outfile = open("visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*2)+"\n"
    
    def update_with_replay_buffer(self, replay_buffer):
        print("Start Update Rtree!Len:",len(replay_buffer._storage))
        j=0
        for experience in replay_buffer._storage:
            obs_e, action_e, rew, new_obs, done, masks, train_times = experience
            for i in range(train_times):
                state_to_record = np.append(obs_e, action_e)
                self.visited_state_tree.insert(self.visited_state_counter,
                            tuple((state_to_record-self.visited_state_dist[0]).tolist()+(state_to_record+self.visited_state_dist[0]).tolist()))
                self.visited_state_counter += 1
            j += 1
            print("Updated count:",j)
        print("Rtree using Train Buffer Updated!,Len:",self.visited_state_counter)

    def add_data_to_rtree(self, training_data):
        (obses_t, actions, rewards, obses_tp1, dones, masks, weights, batch_idxes, training_time) = training_data
        for i in range(len(obses_t)):
            state_to_record = np.append(obses_t[i], actions[i])
            self.visited_state_tree.insert(self.visited_state_counter,
                        tuple((state_to_record-self.visited_state_dist[0]).tolist()+(state_to_record+self.visited_state_dist[0]).tolist()))

            self.visited_state_counter += 1
            self.visited_value_outfile.write(self.visited_value_format % tuple([actions[i],rewards[i]]))

    def calculate_visited_times(self, obs, action):
        
        state_to_count = np.append(obs, action)
        visited_times = sum(1 for _ in self.visited_state_tree.intersection(state_to_count.tolist()))

        return visited_times


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device

        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.k_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.curr_rewards = np.empty((capacity, 1), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, curr_reward, reward, next_obs, done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.curr_rewards[self.idx], curr_reward)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, k=False):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        curr_rewards = torch.as_tensor(self.curr_rewards[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            self.next_obses[idxs], device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        if k:
            return obses, actions, rewards, next_obses, not_dones, torch.as_tensor(self.k_obses[idxs], device=self.device)
        return obses, actions, curr_rewards, rewards, next_obses, not_dones

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.curr_rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.curr_rewards[start:end] = payload[4]
            self.not_dones[start:end] = payload[5]
            self.idx = end