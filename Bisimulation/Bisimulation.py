import argparse
import random
import time
import os
import os.path as osp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import deque
from rtree import index as rindex
from Bisimulation.transition_model import make_transition_model
from Agent.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.controller import Controller
from Agent.dynamic_map import DynamicMap
from Agent.actions import LaneAction

class Bisimulation(object):

    def __init__(self, env):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = self.parse_args()

        self.env = env
        self.obs_shape = env.observation_space.shape
        self.state_space_dim = env.state_space_dim
        self.action_shape = env.action_space.shape

        # Transition Model
        self.transition_model = make_transition_model(
            self.args.transition_model_type, self.state_space_dim, self.action_shape
        ).to(self.device)

        # Reward Model - It might calculated from state
        self.reward_decoder = nn.Sequential(
        nn.Linear(self.state_space_dim + self.action_shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, 128), 
            nn.ReLU(),
            nn.Linear(128, 1)).to(self.device)

        self.reward_decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=self.args.decoder_lr,
            weight_decay=self.args.decoder_weight_lambda
        )

        # Q-value
        self.Rtree = RTree(self.args)

        # Data 
        self.replay_buffer = ReplayBuffer(self.obs_shape, self.action_shape, self.args.replay_buffer_capacity, self.args.batch_size, self.device)


    def parse_args(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--decision_count", type=int, default=1, help="how many steps for a decision")

        # environment
        parser.add_argument('--domain_name', default='carla')
        parser.add_argument('--task_name', default='run')
        parser.add_argument('--image_size', default=84, type=int)
        parser.add_argument('--action_repeat', default=1, type=int)
        parser.add_argument('--frame_stack', default=1, type=int) #3
        parser.add_argument('--resource_files', type=str)
        parser.add_argument('--eval_resource_files', type=str)
        parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
        parser.add_argument('--total_frames', default=1000, type=int)
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=1000000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp'])
        parser.add_argument('--init_steps', default=1, type=int)
        parser.add_argument('--num_train_steps', default=1000, type=int)
        parser.add_argument('--batch_size', default=1, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
        parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
        parser.add_argument('--load_encoder', default=None, type=str)
        # eval
        parser.add_argument('--eval_freq', default=1000, type=int)  # TODO: master had 10000
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        # critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_beta', default=0.9, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        # actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_beta', default=0.9, type=float)
        parser.add_argument('--actor_log_std_min', default=-10, type=float)
        parser.add_argument('--actor_log_std_max', default=2, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        # encoder/decoder
        parser.add_argument('--encoder_type', default='pixelCarla098', type=str, choices=['pixel', 'pixelCarla096', 'pixelCarla098', 'identity'])
        parser.add_argument('--encoder_feature_dim', default=50, type=int)
        parser.add_argument('--encoder_lr', default=1e-3, type=float)
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--encoder_stride', default=1, type=int)
        parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
        parser.add_argument('--decoder_lr', default=1e-3, type=float)
        parser.add_argument('--decoder_update_freq', default=1, type=int)
        parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
        parser.add_argument('--num_layers', default=4, type=int)
        parser.add_argument('--num_filters', default=32, type=int)
        # sac
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        parser.add_argument('--alpha_lr', default=1e-3, type=float)
        parser.add_argument('--alpha_beta', default=0.9, type=float)
        # misc
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--work_dir', default='.', type=str)
        parser.add_argument('--save_tb', default=False, action='store_true')
        parser.add_argument('--save_model', default=True, action='store_true')
        parser.add_argument('--save_buffer', default=True, action='store_true')
        parser.add_argument('--save_video', default=False, action='store_true')
        parser.add_argument('--transition_model_type', default='probabilistic', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--render', default=False, action='store_true')
        parser.add_argument('--port', default=2000, type=int)
        args = parser.parse_args()
        return args
    
    def make_dir(self, dir_path):
        try:
            os.mkdir(dir_path)
        except OSError:
            pass
        return dir_path

    def train_bisim_NNs(self, train_step, env, load_step, policy): # 1 for random policy, 0 for werling
        # Init Planner
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 

        args = self.parse_args()
        self.make_dir(args.work_dir)
        model_dir = self.make_dir(os.path.join(args.work_dir, 'model'))
        buffer_dir = self.make_dir(os.path.join(args.work_dir, 'replay_buffer'))

        # Collected data and train
        episode, episode_reward, done = 0, 0, True
        
        try:
            self.load(model_dir, load_step)
            print("[Bisim_Model] : Load learned model successful, step=",load_step)
            self.replay_buffer.load(buffer_dir)
            print("[Bisim_Model] : Load Buffer!")

        except:
            load_step = 0
            print("[Bisim_Model] : No learned model, Creat new model")

        for step in range(train_step + 1):
            if done:

                obs = env.reset()
                done = False
                episode_reward = 0
                episode_step = 0
                episode += 1
                reward = 0   
            
            # save agent periodically
            if step % args.eval_freq == 0:
                if args.save_model:
                    print("[Bisim_Model] : Saved Model! Step:",step + load_step)
                    self.save(model_dir, step + load_step)
                if args.save_buffer:
                    self.replay_buffer.save(buffer_dir)
                    print("[Bisim_Model] : Saved Buffer!")

            # run training update
            if step >= args.init_steps:
                num_updates = args.init_steps if step == args.init_steps else 1
                for _ in range(num_updates):
                    self.update(self.replay_buffer, step) # Updated Transition and Reward Module


            obs = np.array(obs)
            curr_reward = reward
            
            if policy == 1:
                action = np.array([(random.random() - 0.5) * 2, (random.random() - 0.5) * 2])
                new_obs, reward, done, info = env.step(action)
            else:
                # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory, high_level_action = self.trajectory_planner.trajectory_update(self.dynamic_map)
                high_level_action = random.randint(0, 8)
                print("-------------------",high_level_action)
                # Control
                trajectory = self.trajectory_planner.trajectory_update_CP(high_level_action, rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    action = np.array([control_action.acc, control_action.steering])
                    new_obs, reward, done, info = env.step(action)
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)

            print("Predicted Reward:",self.get_reward_prediction(obs, action))
            print("Actual Reward:",reward)
            print("Predicted State:",self.get_trans_prediction(obs, action)[0])
            print("Actual State:",(new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low))
            episode_reward += reward
            normal_new_obs = (new_obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            self.replay_buffer.add(normal_obs, action, curr_reward, reward, normal_new_obs, done)

            obs = new_obs
            episode_step += 1

    def test_Q_bisim(self, env, load_step, test_policy, test_step=10000):
        
        # Load models
        self.make_dir(self.args.work_dir)
        model_dir = self.make_dir(os.path.join(self.args.work_dir, 'model'))
        test_buffer_dir = self.make_dir(os.path.join(self.args.work_dir, 'test_buffer'))
        
        try:
            self.load(model_dir, load_step)
            print("[Bisim_Model] : Load learned model and buffer successful, step=",load_step)

        except:
            load_step = 0
            print("[Bisim_Model] : No learned model, Creat new model")

        # Test Policy to get Q and states
        test_buffer = ReplayBuffer(self.obs_shape, self.action_shape, self.args.replay_buffer_capacity, self.args.batch_size, self.device)
        done = True
        for step in range(test_step + 1):
            print("step",step)
            if done:
                obs = env.reset()
                done = False
                reward = 0   

            obs = np.array(obs)
            action = test_policy.act(obs)
            print("visited_times", self.Rtree.calculate_visited_times(obs, action))
            print("Q_value",self.Rtree.calculate_Q_value(obs, action) )

            new_obs, reward, done, info = env.step(action)
            obs = new_obs  

            test_buffer.add(obs, action, reward, reward, new_obs, done)
            self.Rtree.add_data_to_rtree(obs, action, reward, new_obs, done)

        test_buffer.save(test_buffer_dir)
        print("[Bisim_Model] : Finished Test, Saved Test Buffer")
        # for experience in replay buffer: all other experience
        normal_obs = (obs - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)

        # calculated Q1-Q2,bisim
        
        # record to txt:(s1,s2,Q1-Q2,bisim,reward1,reward2,transition1,transition2)

        return 0        

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

        # print("debug",loss,reward_loss)
        with open("Reward_loss.txt", 'a') as fw:
            fw.write(str(loss.detach().cpu().numpy())) 
            fw.write(", ")
            fw.write(str(reward_loss.detach().cpu().numpy())) 
            fw.write("\n")
            fw.close()    

        print("[Bisim_Model] : Updated all models! Step:",step)

    def get_reward_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((2), dtype=np.float32)
        np.copyto(np_action, action)
        action = torch.as_tensor(np_action, device=self.device).unsqueeze(0)

        with torch.no_grad():
            obs_with_action = torch.cat([obs, action], dim=1)
            return self.reward_decoder(obs_with_action)

    def get_trans_prediction(self, obs, action):
        obs = (obs - self.env.observation_space.low) / (self.env.observation_space.high - self.env.observation_space.low)

        np_obs = np.empty((1, self.state_space_dim), dtype=np.float32)
        np.copyto(np_obs[0], obs)
        obs = torch.as_tensor(np_obs, device=self.device).float()
        np_action = np.empty((2), dtype=np.float32)
        np.copyto(np_action, action)
        action = torch.as_tensor(np_action, device=self.device).unsqueeze(0)

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

    def __init__(self, args, new_file=True, save_new_data=True):
        if new_file:
            if osp.exists("visited_state_value.txt"):
                os.remove("visited_state_value.txt")
            if osp.exists("state_index.dat"):
                os.remove("state_index.dat")
                os.remove("state_index.idx")
            if osp.exists("visited_value.txt"):
                os.remove("visited_value.txt")

            # _setup_data_saving 
            self.visited_state_value = []
            self.visited_state_counter = 0
        else:
            self.visited_state_value = np.loadtxt("visited_value.txt")
            self.visited_state_value = visited_state_value.tolist()
            self.visited_state_counter = len(visited_state_value)
            print("Loaded Save Rtree, len:",self.visited_state_counter)
        
        self.save_new_data = save_new_data
        self.args = args
        self.trajectory_buffer = deque(maxlen=20)

        self.visited_state_outfile = open("visited_state.txt", "a")
        self.visited_state_format = " ".join(("%f",)*22)+"\n"

        self.visited_value_outfile = open("visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*3)+"\n"

        obs_dimension = 20
        self.visited_state_tree_prop = rindex.Property()
        self.visited_state_tree_prop.dimension = obs_dimension+2
        self.visited_state_dist = np.array([[1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5,1, 1, 0.5, 0.5,1, 1, 0.5, 0.5, 1, 1, 0.5, 0.5, 0.1, 0.1]])#, 10, 0.3, 3, 1, 0.1]])
        # self.visited_state_dist = np.array([[2, 2, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        # self.visited_state_dist = np.array([[2, 2, 1.0, 1.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0, 5, 5, 2.0, 2.0,  0.5]])#, 10, 0.3, 3, 1, 0.1]])
        self.visited_state_tree = rindex.Index('state_index',properties=self.visited_state_tree_prop)

        self.visited_value_outfile = open("visited_value.txt", "a")
        self.visited_value_format = " ".join(("%f",)*3)+"\n"
    
    def state_with_action(self,obs,action):
        return np.append(obs, action)
   
    def add_data_to_rtree(self, obs, action, rew, new_obs, done):
        self.trajectory_buffer.append((obs, action, rew, new_obs, done))
        while len(self.trajectory_buffer) > 10:
            obs_left, action_left, rew_left, new_obs_left, done_left = self.trajectory_buffer.popleft()
            # print("debug111",obs_left)
            # print("debug111",action_left)
            # print("debug111",rew_left)
            state_to_record = self.state_with_action(obs_left, action_left)
            action_to_record = action_left
            r_to_record = rew_left
            if self.save_new_data:
                self.visited_state_value.append(list([np.append(action_to_record, r_to_record)]))
                self.visited_state_tree.insert(self.visited_state_counter,
                    tuple((state_to_record-self.visited_state_dist).tolist()[0]+(state_to_record+self.visited_state_dist).tolist()[0]))
                # print("debug33",state_to_record)
                # print("debug33",self.visited_state_format)
                # print("debug44",np.append(action_to_record, r_to_record))
                # print("debug44",self.visited_value_format)
                self.visited_state_outfile.write(self.visited_state_format % tuple(state_to_record))
                self.visited_value_outfile.write(self.visited_value_format % tuple(np.append(action_to_record, r_to_record)))
                self.visited_state_counter += 1
        

        if done:
            _, _, rew_right, _, _ = self.trajectory_buffer[-1]
            while len(self.trajectory_buffer)>0:
                obs_left, action_left, rew_left, new_obs_left, done_left = self.trajectory_buffer.popleft()
                action_to_record = action_left
                r_to_record = rew_right*self.args.discount**len(self.trajectory_buffer)
                state_to_record = self.state_with_action(obs_left, action_left)
                if self.save_new_data:
                    self.visited_state_value.append(np.append(action_to_record, r_to_record))
                    self.visited_state_tree.insert(self.visited_state_counter,
                        tuple((state_to_record-self.visited_state_dist).tolist()[0]+(state_to_record+self.visited_state_dist).tolist()[0]))
                    self.visited_state_outfile.write(self.visited_state_format % tuple(state_to_record))
                    self.visited_value_outfile.write(self.visited_value_format % tuple(np.append(action_to_record, r_to_record)))
                    self.visited_state_counter += 1

    def calculate_visited_times(self, obs, action):
        
        state_to_count = np.append(obs, action)
        visited_times = sum(1 for _ in self.visited_state_tree.intersection(state_to_count.tolist()))

        return visited_times
    
    def calculate_Q_value(self, obs, action):

        if self.calculate_visited_times(obs, action) == 0:
            return -1, -1, -1
        
        state_to_count = np.append(obs, action)
        value_list = [self.visited_state_value[idx] for idx in self.visited_state_tree.intersection(state_to_count.tolist())]
        value_array_av = np.array(value_list)
        value_array = value_array_av[-1] # Not sure, change

        mean = np.mean(value_array)
        var = np.var(value_array)
        sigma = np.sqrt(var)

        return mean,var,sigma


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