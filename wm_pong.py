import gym ,sys
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
from two_player.pong import PongGame
# from script_model import ScriptLearner
"""
2D-pong 
obs_space = (600, 400, 3)
act_space = (2,)
"""

"""
Training tricks,
1. clear replay buffer every epoch
"""

def parse_args():
    parser = argparse.ArgumentParser("Reinforcement Learning experiments for stochastic games")
    parser.add_argument("--agent", type =str, default = 'DDPG', help = "agent type")
    parser.add_argument("--adv_agent", type = str, default = "script", help = "adv agent type")
    parser.add_argument("--game", type=str, default="Pong-2p-v0", help="name of the  env")
    parser.add_argument("--timestep", type= int, default= 2, help ="the time step of act_trajectory")
    parser.add_argument("--iteration", type=int, default=60000, help="number of episodes")
    parser.add_argument("--seed", type = int, default = 10, help = "random seed")
    parser.add_argument("--epoch", type = int, default = 10, help = "training epoch")
    parser.add_argument("--episodes", type = int, default = 100, help = "episodes")
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
    log = open(log_path, '+w', 1)

    with U.single_threaded_session():
        
        env = PongGame()
        obs_space = env.observation_space
        act_space = env.action_space
        agent1 = DDPGLearner(obs_space, act_space, "agent1",arglist)
        if arglist.adv_agent == "agent":
            agent2 = DDPGLearner(obs_space, act_space, "agent2", arglist)
        agent2 = agent1
        U.initialize()
        saver = tf.train.Saver()
        for epo in range(arglist.epoch):
            agent1.reset_replay_buffer()
            agent2.reset_replay_buffer()
            agent1_q_loss = []
            agent1_p_loss = []
            agent2_q_loss = []
            agent2_p_loss = []
            agent1_score = []
            agent2_score = []
            for ep in range(arglist.episodes):
                obs = env.reset()
                while 1:
                    # env.render('human')
                    act1 = agent1.get_act([obs])[0]
                    if arglist.adv_agent == "script":
                        act2 = "script"
                    elif arglist.adv_agent == "agent":
                        act2 = agent.get_act([obs])[0]

                    obs_n, rews, goals, win = env.step([act1,act2])
                    agent1.experience(obs, [act1], rews[0], obs_n, goals[0])
                    if arglist.adv_agent =="agent":
                        agent2.experience(obs, [act2], rews[1], obs_n, goals[1])
                    print(env.scores, act1, act2, rews, goals, win)
                    if win:
                        break
                ep_score = env.scores  
                agent1_score.append(ep_score[0])
                agent2_score.append(ep_score[1])
                q_loss1, p_loss1 = agent1.learn(arglist.batch_size, arglist.gamma)
                q_loss2, p_loss2  = 0,0      
                if arglist.adv_agent == "agent":
                    q_loss2, p_loss2 = agent2.learn(arglist.batch_size, arglist.gamma)  
                agent1_q_loss.append(q_loss1)
                agent1_p_loss.append(p_loss1)
                agent2_q_loss.append(q_loss2)
                agent2_p_loss.append(p_loss2)
                print("Episodes {} scores: {}, losses {}".format(ep, ep_score,[q_loss1, p_loss1 , q_loss2, p_loss2 ]))
                log.write('{}\n'.format(','.join(map(str, 
                    [ep_score[0], ep_score[1] , q_loss1, p_loss1 , q_loss2, p_loss2 ]))))   
            print("Epoch {} scores: {}, losses {}".format(
                epo, [np.mean(agent1_score), np.mean(agent2_score)],
                [np.mean(agent1_q_loss), np.mean(agent1_p_loss) , np.mean(agent2_q_loss), np.mean(agent2_p_loss) ]))
            U.save_state(model_path, saver)








