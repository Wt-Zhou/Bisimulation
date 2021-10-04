import argparse

import numpy as np
import math
import os
import os.path as osp
import tensorflow as tf
import tempfile
import time
import random
import _thread
import baselines.common.tf_util as U
import random
from rtree import index as rindex
from collections import deque
from scipy import stats
from baselines import logger
from baselines import deepq
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from baselines.common.misc_util import (
    boolean_flag,
    pickle_load,
    pretty_eta,
    relatively_safe_pickle_dump,
    set_global_seeds,
    RunningAvg,
    SimpleMonitor
)
from baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from baselines.common.atari_wrappers_deprecated import wrap_dqn
from Agent.model import dqn_model, bootstrap_model
from Agent.JunctionTrajectoryPlanner import JunctionTrajectoryPlanner
from Agent.controller import Controller
from Agent.dynamic_map import DynamicMap
from Agent.actions import LaneAction

class DQN(object):

    def __init__(self):
        self.trajectory_planner = JunctionTrajectoryPlanner()
        self.controller = Controller()
        self.dynamic_map = DynamicMap()
        self.target_speed = 30/3.6 


    def parse_args(self):
        parser = argparse.ArgumentParser("DQN experiments for Atari games")
        # Environment
        parser.add_argument("--env", type=str, default="DQN", help="name of the game")
        parser.add_argument("--seed", type=int, default=42, help="which seed to use")
        parser.add_argument("--decision_count", type=int, default=5, help="how many steps for a decision")
        # Core DQN parameters
        parser.add_argument("--replay-buffer-size", type=int, default=int(1e6), help="replay buffer size")
        parser.add_argument("--train-buffer-size", type=int, default=int(1e8), help="train buffer size")
        parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for Adam optimizer")
        parser.add_argument("--num-steps", type=int, default=int(4e7), help="total number of steps to run the environment for")
        parser.add_argument("--batch-size", type=int, default=64, help="number of transitions to optimize at the same time")
        parser.add_argument("--learning-freq", type=int, default=20, help="number of iterations between every optimization step")
        parser.add_argument("--target-update-freq", type=int, default=50, help="number of iterations between every target network update") #10000
        parser.add_argument("--learning-starts", type=int, default=50, help="when to start learning") 
        parser.add_argument("--gamma", type=float, default=0.995, help="the gamma of q update") 
        parser.add_argument("--bootstrapped-data-sharing-probability", type=float, default=0.8, help="bootstrapped_data_sharing_probability") 
        parser.add_argument("--bootstrapped-heads-num", type=int, default=10, help="bootstrapped head num of networks") 
        parser.add_argument("--learning-repeat", type=int, default=10, help="learn how many times from one sample of RP") 
        # Bells and whistles
        boolean_flag(parser, "double-q", default=True, help="whether or not to use double q learning")
        boolean_flag(parser, "dueling", default=False, help="whether or not to use dueling model")
        boolean_flag(parser, "bootstrap", default=True, help="whether or not to use bootstrap model")
        boolean_flag(parser, "prioritized", default=True, help="whether or not to use prioritized replay buffer")
        parser.add_argument("--prioritized-alpha", type=float, default=0.9, help="alpha parameter for prioritized replay buffer")
        parser.add_argument("--prioritized-beta0", type=float, default=0.1, help="initial value of beta parameters for prioritized replay")
        parser.add_argument("--prioritized-eps", type=float, default=1e-6, help="eps parameter for prioritized replay buffer")
        # Checkpointing
        parser.add_argument("--save-dir", type=str, default="./logs", help="directory in which training state and model should be saved.")
        parser.add_argument("--save-azure-container", type=str, default=None,
                            help="It present data will saved/loaded from Azure. Should be in format ACCOUNT_NAME:ACCOUNT_KEY:CONTAINER")
        parser.add_argument("--save-freq", type=int, default=10000, help="save model once every time this many iterations are completed")
        boolean_flag(parser, "load-on-start", default=True, help="if true and model was previously saved then training will be resumed")
        return parser.parse_args()

    def maybe_save_model(self, savedir, state):
        """This function checkpoints the model and state of the training algorithm."""
        if savedir is None:
            return
        start_time = time.time()
        model_dir = "model-{}".format(state["num_iters"])
        U.save_state(os.path.join(savedir, model_dir, "saved"))
        state_dir = "training_state.pkl-{}".format(state["num_iters"]) + ".zip"
        relatively_safe_pickle_dump(state, os.path.join(savedir, state_dir), compression=True)
        logger.log("Saved model in {} seconds\n".format(time.time() - start_time))

    def maybe_load_model(self, savedir, model_step):
        """Load model if present at the specified path."""
        if savedir is None:
            return
        model_dir = "training_state.pkl-{}".format(model_step) + ".zip"
        # state_path = os.path.join(os.path.join(savedir, 'training_state.pkl-100028.zip'))
        state_path = os.path.join(os.path.join(savedir, model_dir))
        found_model = os.path.exists(state_path)
        if found_model:
            state = pickle_load(state_path, compression=True)
            model_dir = "model-{}".format(state["num_iters"])
            U.load_state(os.path.join(savedir, model_dir, "saved"))
            logger.log("Loaded models checkpoint at {} iterations".format(state["num_iters"]))
            return state

    def test_dqn(self, model_list, test_steps, env):
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True

        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
            # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters
            
            learning_rate = args.lr # Maybe Segmented

            U.initialize()
            num_iters = 0

            for model_step in model_list:
                # Load the model
                state = self.maybe_load_model(savedir, model_step)
                if state is not None:
                    num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
        
                start_time, start_steps = None, None
                test_iters = 0

                obs = env.reset()
                self.trajectory_planner.clear_buff()

                # Test
                while test_iters < test_steps:
                    num_iters += 1
                    obs = np.array(obs)
                    # Rule-based Planner
                    self.dynamic_map.update_map_from_obs(obs, env)
                    rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                    # DQN Action
                    q_list = q_values_dqn(obs[None])
                    action = np.array(np.where(q_list[0]==np.max(q_list[0]))[0])

                    print("[Bootstrap DQN]: Obs",obs.tolist())
                    print("[Bootstrap DQN]: DQN Action",action)
                    print("[Bootstrap DQN]: DQN value",q_list[0])

                    # Control
                    trajectory = self.trajectory_planner.trajectory_update_UBP(action[0], rule_trajectory)
                    for i in range(args.decision_count):
                        control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                        output_action = [control_action.acc, control_action.steering]
                        new_obs, rew, done, info = env.step(output_action)
                        if done:
                            break
                        self.dynamic_map.update_map_from_obs(new_obs, env)

                    obs = new_obs
                    if done:
                        obs = env.reset()
                        self.trajectory_planner.clear_buff()
                        test_iters += 1

                    # Record Data    
                self.record_test_data(model_step, 2333, 2333, env)

    def learn(self, total_timesteps, env, load_model_step):      
        # Init DRL
        args = self.parse_args()
        savedir = args.save_dir + "_" + args.env
        save_model = True


        if args.seed > 0:
            set_global_seeds(args.seed)
        
        with U.make_session(120) as sess:
        # Create training graph and replay buffer
            act_dqn, train_dqn, update_target_dqn, q_values_dqn = deepq.build_train_dqn(
                make_obs_ph=lambda name: U.CARLAInput(env.observation_space, name=name),
                original_dqn=dqn_model,
                num_actions=env.action_space.n,
                optimizer=tf.compat.v1.train.AdamOptimizer(learning_rate=args.lr, epsilon=1e-4),
                gamma=args.gamma,
                grad_norm_clipping=10,
                double_q=args.double_q
            )
            
            replay_buffer = PrioritizedReplayBuffer(args.replay_buffer_size, args.prioritized_alpha)
            beta_schedule = LinearSchedule(args.num_steps / 50, initial_p=args.prioritized_beta0, final_p=1.0) # Approximately Iters

            learning_rate = args.lr # maybe Segmented

            U.initialize()
            update_target_dqn()
            num_iters = 0

            # Load the model
            state = self.maybe_load_model(savedir, load_model_step)
            if state is not None:
                num_iters, replay_buffer = state["num_iters"], state["replay_buffer"]
            
            start_time, start_steps = None, None

            obs = env.reset()
            self.trajectory_planner.clear_buff()
            decision_count = 0

            while num_iters < total_timesteps:
                obs = np.array(obs)

                # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                # Bootstapped Action
                dqn_q = q_values_dqn(obs[None])
                optimal_action = np.array(np.where(dqn_q[0]==np.max(dqn_q[0]))[0][0])
                random_action = random.randint(0,7)

                if random.uniform(0,1) < 0.2: # epsilon-greddy
                    action = random_action
                else:
                    action = optimal_action

                print("[Bootstrap DQN]: Obs",obs.tolist())
                print("[Bootstrap DQN]: Action",action, random_action)
                print("[Bootstrap DQN]: DQN Q-value",dqn_q)

                # Control
                trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    output_action = [control_action.acc, control_action.steering]
                    new_obs, rew, done, info = env.step(output_action)
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action, rew, new_obs, float(done), mask)
                obs = new_obs
                if done:
                    obs = env.reset()
                    self.trajectory_planner.clear_buff()

                if (num_iters > args.learning_starts and
                        num_iters % args.learning_freq == 0):
                    # Sample a bunch of transitions from replay buffer
                    if args.prioritized:
                        # Update rl
                        if replay_buffer.__len__() > args.batch_size:
                            for i in range(args.learning_repeat):
                                print("[Bootstrap DQN]: Learning")
                                experience = replay_buffer.sample(args.batch_size, beta=beta_schedule.value(num_iters), count_train=True)
                                (obses_t, actions, rewards, obses_tp1, dones, masks, train_time, weights, batch_idxes) = experience
                                td_errors_dqn, q_t_selected_dqn, q_t_selected_target_dqn, qt_dqn = train_dqn(obses_t, actions, rewards, obses_tp1, dones, masks, weights, learning_rate)
                                # Update the priorities in the replay buffer
                                new_priorities = np.abs(td_errors_dqn) + args.prioritized_eps

                                replay_buffer.update_priorities(batch_idxes, new_priorities)
                    
                    else:
                        obses_t, actions, rewards, obses_tp1, dones = replay_buffer.sample(args.batch_size)
                        weights = np.ones_like(rewards)
                    
                    
                # Update target network.
                if num_iters % args.target_update_freq == 0:
                    print("[Bootstrap DQN]: Update target network")
                    update_target_dqn()


               
                start_time, start_steps = time.time(), 0

                # Save the model and training state.
                if num_iters >= 0 and num_iters % args.save_freq == 0:
                    print("[Bootstrap DQN]: Save model")
                    self.maybe_save_model(savedir, {
                        'replay_buffer': replay_buffer,
                        'num_iters': num_iters,
                    })
                    # self.evaluate_policy(env)

                num_iters += 1


            print("[Bootstrap DQN]: Finish Training, Save model")
            self.maybe_save_model(savedir, {
                'replay_buffer': replay_buffer,
                'num_iters': num_iters,
            })

    def evaluate_policy(self, env, task1_num=10, task2_num=100):
        print("[Bisimulation]: Evaluate policy")
        obs.env.reset()
        env.init_case_1()
        while num_iters < task1_num:
                obs = np.array(obs)

                # Rule-based Planner
                self.dynamic_map.update_map_from_obs(obs, env)
                rule_trajectory = self.trajectory_planner.trajectory_update(self.dynamic_map)

                # Bootstapped Action
                dqn_q = q_values_dqn(obs[None])
                optimal_action = np.array(np.where(dqn_q[0]==np.max(dqn_q[0]))[0][0])
                random_action = random.randint(0,7)

                if random.uniform(0,1) < 0.2: # epsilon-greddy
                    action = random_action
                else:
                    action = optimal_action

                print("[Testing Cases1]: Obs",obs.tolist())
                print("[Testing Cases1]: Action",action, random_action)
                print("[Testing Cases1]: DQN Q-value",dqn_q)

                # Control
                trajectory = self.trajectory_planner.trajectory_update_UBP(action, rule_trajectory)
                for i in range(args.decision_count):
                    control_action =  self.controller.get_control(self.dynamic_map,  trajectory.trajectory, trajectory.desired_speed)
                    output_action = [control_action.acc, control_action.steering]
                    new_obs, rew, done, info = env.step(output_action)
                    if done:
                        break
                    self.dynamic_map.update_map_from_obs(new_obs, env)
                    
                mask = np.random.binomial(1, args.bootstrapped_data_sharing_probability, args.bootstrapped_heads_num) # add mask for data
                replay_buffer.add(obs, action, rew, new_obs, float(done), mask)
                obs = new_obs
                if done:
                    obs = env.reset()
                    self.trajectory_planner.clear_buff()


    def record_test_data(self, model_step, uncertainty_thres, visited_time_thres, env):
        with open("Test_data-{}".format(model_step) + "-{}".format(uncertainty_thres) + "-{}".format(visited_time_thres) + ".txt", 'a') as fw:
            fw.write(str(env.task_num - 1))  # The num will be add 1 in reset()
            fw.write(", ")
            fw.write(str(env.stuck_num)) 
            fw.write(", ")
            fw.write(str(env.collision_num)) 
            fw.write("\n")
            fw.close()    
        env.clean_task_nums()




