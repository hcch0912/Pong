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
#from gym.envs.classic_control import rendering
import common.tf_util as U


with tf.Session() as sess:
    new_saver = tf.train.import_meta_graph('./tmp/model.meta')
    new_saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
    graph = tf.get_default_graph()
    world_model = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent1/world_model/")
    #<tf.Variable 'agent1/world_model/fully_connected_1/weights:0' shape=(512, 20) dtype=float32_ref>, <tf.Variable 'agent1/world_model/fully_connected_1/biases:0' shape=(20,) dtype=float32_ref> 
    
    world_model_last_weight = graph.get_tensor_by_name('agent1/world_model/fully_connected_1/weights:0')
    world_model_last_bias = graph.get_tensor_by_name('agent1/world_model/fully_connected_1/biases:0')
    #print(world_model_last_weight,world_model_last_bias)
    
    
    #<tf.Operation 'agent1/world_model/fully_connected_1/Relu' type=Relu>
    z = graph.get_operation_by_name('agent1/world_model/fully_connected_1/Relu').outputs[0]
    
    #agent1/Placeholder_1:0   obs_ph
    #agent1/Placeholder:0  act_ph
    obs_ph = graph.get_tensor_by_name('agent1/Placeholder_1:0')
    act_ph = graph.get_tensor_by_name('agent1/Placeholder:0')
    
    env = PongGame()
    obs = env.reset()
    
    print(obs_ph)
    print(z)
    z_output = sess.run(z, feed_dict = {
        obs_ph: [obs]
    })
    
    print(z_output)
	#viewer = rendering.SimpleImageViewer()
	#viewer.imshow(obs)