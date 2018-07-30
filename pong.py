import gym ,roboschool, sys
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import argparse
import copy
from common.replay_buffer import ReplayBuffer
import common.tf_util as U
# from q_model import *
from ddpg_model import * 
import time

"""
2D-pong 
obs_space = (600, 400, 3)
act_space = (2,)
3D-pong
obs_space = (13,)
act_space = (2,)    
"""

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--epoch", type = int, default = 10, help = "training epoch")
    parser.add_argument("--episodes", type = int, default = 20, help = "episodes")
    parser.add_argument("--steps", type = int, default = 50, help ="steps in one episode" )
    parser.add_argument("--batch_size", type = int, default = 64, help = "set the batch_size")
    parser.add_argument("--max_episode_len", type = int, default = 50, help = "max episode length")
    parser.add_argument("--warm_up_steps", type = int, default = 1000, help = "set the warm up steps")
    parser.add_argument("--lr", type = float, default = 0.001, help = "learning rate")
    parser.add_argument("--gamma", type = float, default = 0.99, help = "discount rate")
    return parser.parse_args()


if __name__ == '__main__':
    arglist = parse_args()
    model_path = './tmp/model'
    random.seed(arglist.seed )
    log_path = './logs/{}_{}_{}_{}.csv'.format(arglist.game, arglist.agent, arglist.seed, time.time())
    log = open(log_path, '+w')

    with U.single_threaded_session():
        
        env = gym.make(arglist.game)
        env.unwrapped.multiplayer(env,"server1", player_n=1)

        
        p_loss_1_list = []
        p_loss_2_list = []
        q_loss_1_list = []
        q_loss_2_list = []
        if arglist.game == "RoboschoolPong-v1":
            obs_space = env.observation_space
            action_space = env.action_space
            agent1 = DDPGLearner(obs_space, action_space, "agent1", arglist)
            U.initialize()
            saver = tf.train.Saver()
            for epoch in range(arglist.epoch):
                for ep in range(arglist.episodes):
                    ep_score = []
                    for step in range(arglist.steps):
                        score = 0
                        frame = 0
                        restart_delay = 0
                        obs = env.reset()
                        while 1:
                            action = agent1.get_act ([obs])
                            obs_n, rew, done, info = env.step(action)
                            agent1.experience(obs, action, rew, obs_n, done)
                            obs = obs_n
                            score += rew
                            frame += 1
                            if not done: continue
                            if restart_delay==0:
                                print("score=%0.2f in %i frames" % (score, frame))
                                ep_score.append(score)
                                restart_delay = 60*2  # 2 sec at 60 fps
                            restart_delay -= 1
                            if restart_delay==0: break
                    print("Episode {} Score: {}".format(ep, np.mean(ep_score)))   
                    p_loss, q_loss = agent1.learn(arglist.batch_size, arglist.gamma)
                    log.write('{}\n'.format(','.join(map(str, [np.mean(ep_score), p_loss, q_loss]))))     
                    p_loss_1_list.append(p_loss)      
                    q_loss_1_list.append(q_loss)  
                print("Epoch loss {} {}".format(np.mean(p_loss_1_list), np.mean(q_loss_1_list)))
                U.save_state(model_path, saver)


        if arglist.game =="Pong-2p-v0": 
            #convert openai  space to normal  soace
            action_space = 1
            obs_space = env.observation_space.shape
            agent1 = DDPGLearner(obs_space, action_space, "agent1", arglist)
            agent2 = DDPGLearner(obs_space, action_space,"agent2", arglist)
            
            U.initialize()
            saver = tf.train.Saver()
            for ep in range(arglist.episodes):
                observation = env.reset()
                for steps in range(arglist.steps):
                # env.render('human')
                    action1 =  agent1.get_act([observation])
                    observation1, reward1, done, info = env.step([action1, 0])
                    action2 = agent2.get_act([observation1])
                    observation2, reward2, done, info = env.step([0, action2])
                    agent1.experience(observation, action1, reward1, observation1, done, None )
                    agent2.experience(observation1, action2, reward2, observation2, done, None)
                    observation = observation2
                    if done:
                        observation  = env.reset()
                        if ep > 5:   
                            p_loss1, q_loss1 = agent1.learn(128)
                            p_loss2, q_loss2  = agent2.learn(128)
                            p_loss_1_list.append(p_loss1)
                            p_loss_2_list.append(p_loss2)
                            q_loss_1_list.append(q_loss1)
                            q_loss_2_list.append(q_loss2)
                            
                            print("Episode: {}, p_loss_1: {}, q_loss_2: {}, p_loss_2: {}, q_loss_2: {}".format(
                                ep, np.mean(p_loss_1_list), np.mean(q_loss_1_list), np.mean(p_loss_2_list), np.mean(q_loss_2_list)))



