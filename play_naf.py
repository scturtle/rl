#!/usr/bin/env python
from __future__ import print_function
from six import print_ as print
import sys
import time
import itertools
import easydict
import numpy as np
import gym
import tensorflow as tf
from naf import NAF

def play(checkpoint):
    env = gym.make('Pendulum-v0')
    env.reset()
    env.render()

    status_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    naf_cfg = easydict.EasyDict(
        dict(
            status_dim=status_dim,
            action_dim=action_dim,
            hidden_dim=200,
            learning_rate=0.0003,
            l2_reg=0.0003,
            is_training=False))

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    # sess.run(tf.global_variables_initializer())

    estimator = NAF(naf_cfg)
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)

    while 1:
        state = env.reset()
        episode_reward = 0
        env.render()
        for i in itertools.count(1):
            action = estimator.predict(np.expand_dims(state, 0))[0]
            action = np.minimum(env.action_space.high, np.maximum(env.action_space.low, action))
            next_state, reward, done, _ = env.step(action)
            env.render()
            time.sleep(0.01)
            episode_reward += reward
            if i > 200 or done:
                print('episode reward:', episode_reward)
                break
            else:
                state = next_state

    sess.close()


if __name__ == '__main__':
    assert len(sys.argv) == 2 and sys.argv[1].endswith('model')
    play(sys.argv[1])
