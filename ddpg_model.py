import numpy as np
import random
import tensorflow as tf
from tensorflow.contrib import rnn
import tensorflow.contrib.layers as layers
import common.tf_util as U
from common.distributions import make_pdtype
from common.replay_buffer import ReplayBuffer

def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]

def make_update_exp(vals, target_vals):
    polyak = 1.0 - 1e-2
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])

def conv_model(X, num_outputs, scope, reuse = False ):
    with tf.variable_scope(scope, reuse = reuse):
        conv1 = tf.contrib.layers.conv2d(
                X, 32, 8, 4, activation_fn=tf.nn.relu)
        conv2 = tf.contrib.layers.conv2d(
                conv1, 64, 4, 2, activation_fn=tf.nn.relu)
        conv3 = tf.contrib.layers.conv2d(
                conv2, 64, 3, 1, activation_fn=tf.nn.relu)
        flattened = tf.contrib.layers.flatten(conv3)
        fc1 = tf.contrib.layers.fully_connected(flattened, 512)
        out = tf.contrib.layers.fully_connected(fc1, num_outputs) 
    return out 
def mlp_model(input, num_outputs, scope, reuse=False, num_units=8, rnn_cell=None):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):

        out1 = layers.fully_connected(input, num_outputs=num_units, activation_fn=tf.nn.relu)
        # out2 = layers.fully_connected(out1, num_outputs=num_units, activation_fn=tf.nn.relu)
        out3 = layers.fully_connected(out1, num_outputs=num_outputs)

        return out3
class DDPGLearner():
    def __init__(self, input_space, act_space,scope, args ):
        self.input_shape = input_space
        self.act_space = act_space
        self.scope = scope
        self.replay_buffer = ReplayBuffer(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.optimizer = tf.train.AdamOptimizer(learning_rate=args.lr)
        self.grad_norm_clipping = 0.5
        with tf.variable_scope(self.scope):
            act_pdtype = make_pdtype(act_space)

            # act_ph = act_pdtype.sample_placeholder([None], name= "action")
            act_ph = tf.placeholder(tf.float32, shape = (None, 1))
            if args.game == "RoboschoolPong-v1":
                obs_ph = tf.placeholder(tf.float32, shape = (None, input_space.shape[0]))
            elif args.game == "Pong-2p-v0":
                obs_ph = tf.placeholder(tf.float32, shape = (None, input_space.shape[0], input_space.shape[1], input_space.shape[2]))    
            q_target = tf.placeholder(tf.float32, shape =( None,))

            #build the world representation z
            z = conv_model(obs_ph, 20, scope = "world_model")
            p_input = z

            p = mlp_model(p_input, 2, scope = "p_func")
            p_func_vars = U.scope_vars(U.absolute_scope_name("p_func"))

            act_pd = act_pdtype.pdfromflat(p)
            act_sample = act_pd.sample()

            p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))
           
            q_input = tf.concat([z , act_sample], -1)
            q = mlp_model(q_input, 1, scope = "q_func")
            q_func_vars = U.scope_vars(U.absolute_scope_name("q_func"))
            pg_loss = -tf.reduce_mean(q)

            q_loss = tf.reduce_mean(tf.square(q - q_target))
            # q_reg = tf.reduce_mean(tf.square(q))
            q_optimize_expr = U.minimize_and_clip(self.optimizer, q_loss, q_func_vars, self.grad_norm_clipping)

            p_loss = pg_loss + p_reg * 1e-3

            p_optimize_expr = U.minimize_and_clip(self.optimizer, p_loss, p_func_vars, self.grad_norm_clipping)

            p_values = U.function([obs_ph], p)


            target_p = mlp_model(z, 2, scope = "target_p_func")
            target_p_func_vars = U.scope_vars(U.absolute_scope_name("target_p_func"))

            target_q = mlp_model(q_input, 1 , scope = "target_q_func")
            target_q_func_vars = U.scope_vars(U.absolute_scope_name("target_q_func"))
            target_act_sample = act_pdtype.pdfromflat(target_p).sample()

            self.update_target_p = make_update_exp(p_func_vars, target_p_func_vars)
            self.update_target_q  = make_update_exp(q_func_vars, target_q_func_vars)

            self.act = U.function(inputs = [obs_ph], outputs = act_sample)
            self.target_act = U.function(inputs = [obs_ph], outputs = target_act_sample)
            self.p_train = U.function(inputs = [obs_ph] + [act_ph], outputs = p_loss, updates = [p_optimize_expr])
            self.q_train = U.function(inputs = [obs_ph] + [act_ph] + [q_target], outputs = q_loss, updates = [q_optimize_expr] )
            self.q_values = U.function([obs_ph] + [act_ph], q)
            self.target_q_values = U.function([obs_ph] + [act_ph], target_q)

    def get_act(self, obs):
        return self.act(*([obs]))[0]     
    def experience(self, obs, act, rew, new_obs, done):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done))

    def learn(self, batch_size, gamma):
        if len(self.replay_buffer) < self.max_replay_buffer_len: # replay buffer is not large enough
            return 0,0
        self.replay_sample_index = self.replay_buffer.make_index(batch_size)
        # collect replay sample from all agents
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(self.replay_sample_index )
        # train q network
        target_q = [0.] * batch_size
        target_act_next = self.target_act(obs_next)
        target_q_next = self.target_q_values(*([obs_next] + [target_act_next]))
        for i in range(batch_size):
            target_q[i] += rew[i] + gamma * (1.0 -  done[i]) * target_q_next[i]
        target_q = np.squeeze(target_q, axis = 1)    
        q_loss = self.q_train(*([obs] + [act] + [target_q]))
        p_loss = self.p_train(*([obs] + [act]))

        self.update_target_p() 
        self.update_target_q()
        return q_loss, p_loss  
    def reset_replay_buffer(self):
        self.replay_buffer = ReplayBuffer(1e6)     
