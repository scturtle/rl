#!/usr/bin/env python
from __future__ import print_function
from six.moves import input
from six import print_ as print

import sys

import gym
import numpy as np
import tensorflow as tf
from dqn import Estimator, ImageProcessor

def play(checkpoint_dir):

    env = gym.make("Breakout-v0")
    env.render()
    nA = 4

    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    q_estimator = Estimator(nA=nA, scope="q")
    image_processor = ImageProcessor()

    checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
    assert checkpoint_path

    saver = tf.train.Saver()

    saver.restore(sess, checkpoint_path)

    while 1:
        total_reward = 0
        env.reset()
        state, _, _, _ = env.step(1)
        state = image_processor.process(state)
        state = np.stack([state] * 4, axis=2)
        while 1:
            fake_batch = np.expand_dims(state, 0)
            fake_batch = np.concatenate([np.expand_dims(state, 0)] * 32, axis=0)
            q_values = q_estimator.predict(fake_batch)
            action = np.argmax(q_values[0])
            if action == 1:
                print("\nFire!")
            next_state, reward, done, _ = env.step(action)
            env.render()
            total_reward += reward
            print("\rAction: {} Total reward: {}".format(action, total_reward), end="", flush=True)
            next_state = image_processor.process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            if done:
                input("\nDied. Enter to retry...")
                break
            # if (state == next_state).all():
            # if np.random.random() < 0.1:
                # print("\nDied? Fire...")
                # env.step(1)
            state = next_state


if __name__ == '__main__':
    play(sys.argv[1])
