#!/usr/bin/env python
from __future__ import print_function

import os
from datetime import datetime

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from tensorflow.contrib.framework import get_or_create_global_step

# from: github.com/dennybritz/reinforcement-learning
from cliff_walking import CliffWalkingEnv

class ActorCritic:

    def __init__(self, nA, nS):
        self.state = tf.placeholder(shape=(), dtype=tf.uint8, name='state')
        self.action = tf.placeholder(dtype=tf.int32, name='action')
        self.target = tf.placeholder(dtype=tf.float32, name='target')
        self.adv = tf.placeholder(dtype=tf.float32, name='advantage')

        state_onehot = tf.one_hot(self.state, nS, dtype=tf.float32)
        hidden = tf.expand_dims(state_onehot, 0)

        self.probs = tf.squeeze(tfl.fully_connected(
            hidden, nA, activation_fn=tf.nn.softmax, biases_initializer=None))
        self.value = tf.squeeze(tfl.fully_connected(
            hidden, 1, activation_fn=None, biases_initializer=None))

        action_prob = tf.gather(self.probs, self.action)

        self.policy_loss = -tf.log(action_prob) * self.adv
        self.value_loss = tf.squared_difference(self.value, self.target)
        self.loss = self.policy_loss + self.value_loss

        learning_rate = 0.01
        global_step = get_or_create_global_step()
        self.train_p = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                               .minimize(self.policy_loss, global_step=global_step)
        self.train_v = tf.train.AdamOptimizer(learning_rate=learning_rate)\
                               .minimize(self.value_loss, global_step=global_step)

        self.summary = tf.summary.merge([
            tf.summary.scalar("target", self.target),
            tf.summary.scalar("adv", self.adv),
            tf.summary.histogram("probs", self.probs),
            tf.summary.scalar("value", self.value),
            tf.summary.scalar("policy_loss", self.policy_loss),
            tf.summary.scalar("value_loss", self.value_loss)
        ])

def main():

    env = CliffWalkingEnv()
    sess = tf.Session()
    ac = ActorCritic(env.nA, env.nS)
    sess.run(tf.global_variables_initializer())

    date_str = datetime.now().strftime("%m%d_%H%M%S")
    summaries_dir = os.path.abspath("./summary/ac/" + date_str)
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())

    state = env.reset()
    episode_cnt = 0
    episode_step = 0
    episode_reward = 0.
    while 1:

        probs, value = sess.run([ac.probs, ac.value], feed_dict={ac.state: state})
        action = np.random.choice(env.nA, p=probs)
        next_state, reward, done, _ = env.step(action)

        episode_step += 1
        episode_reward += reward

        value_next = sess.run(ac.value, feed_dict={ac.state: next_state})
        td_target = reward + 0.99 * value_next
        td_adv = td_target - value

        summary, global_step, _, _ = \
            sess.run([ac.summary, get_or_create_global_step(), ac.train_p, ac.train_v],
                     feed_dict={ac.state: state, ac.action: action,
                                ac.adv: td_adv, ac.target: td_target})

        summary_writer.add_summary(summary, global_step)

        if done or episode_step > 1000:
            print('episode cnt:', episode_cnt, 'eoisode step:', episode_step, 'reward:', episode_reward)
            episode_step = 0.
            episode_reward = 0.
            episode_cnt += 1
            state = env.reset()
        else:
            state = next_state

if __name__ == '__main__':
    main()
