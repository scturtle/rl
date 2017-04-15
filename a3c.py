#!/usr/bin/env python
from __future__ import print_function
from six.moves import range, zip

import os
import time
import multiprocessing
import threading
from collections import namedtuple
from datetime import datetime

import easydict
import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from tensorflow.contrib.framework import get_or_create_global_step


class ImageProcessor():

    def __init__(self):
        with tf.variable_scope("image_processor"):
            self.image = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.image)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, sess, image):
        """ [210, 160, 3] -> [84, 84] """
        return sess.run(self.output, feed_dict={self.image: image})

def normalized_columns_initializer(std=1.0):
    def _initializer(shape, dtype=None, partition_info=None):
        out = np.random.randn(*shape).astype(np.float32)
        out *= std / np.sqrt(np.square(out).sum(axis=0, keepdims=True))
        return tf.constant(out)
    return _initializer

class Network:

    def __init__(self, cfg, scope='global'):
        self.cfg = cfg
        self.sess = tf.get_default_session()
        self.scope = scope
        self.summary_writer = cfg.summary_writer
        with tf.variable_scope(scope):
            self._build()

    def _build(self):
        self.X = tf.placeholder(shape=[None, 84, 84], dtype=tf.uint8, name="X")

        X = tf.to_float(self.X) / 255.0
        conv1 = tfl.conv2d(X, 16, 8, 4, activation_fn=tf.nn.elu, padding='VALID')
        conv2 = tfl.conv2d(conv1, 32, 4, 2, activation_fn=tf.nn.elu, padding='VALID')
        hidden = tfl.fully_connected(tfl.flatten(conv2), 256, activation_fn=tf.nn.elu)

        cell = tf.contrib.rnn.BasicLSTMCell(256, state_is_tuple=True)
        self.state_init = (np.zeros((1, cell.state_size.c)),
                           np.zeros((1, cell.state_size.h)))
        c_in = tf.placeholder(shape=(1, cell.state_size.c), dtype=tf.float32)
        h_in = tf.placeholder(shape=(1, cell.state_size.h), dtype=tf.float32)
        self.state_in = (c_in, h_in)
        state_in = tf.contrib.rnn.LSTMStateTuple(c_in, h_in)
        batch_size = tf.shape(self.X)[:1]
        rnn_outputs, rnn_state = tf.nn.dynamic_rnn(
            cell, tf.expand_dims(hidden, 0),
            initial_state=state_in,
            sequence_length=batch_size,
            time_major=False)
        self.state_out = rnn_state
        rnn_out = tf.reshape(rnn_outputs, [-1, 256])

        # action probability
        self.probs = tfl.fully_connected(
            rnn_out, self.cfg.action_dim,
            activation_fn=tf.nn.softmax,
            weights_initializer=normalized_columns_initializer(0.01),
            biases_initializer=None)
        # state value
        self.value = tfl.fully_connected(
            rnn_out, 1, activation_fn=None,
            weights_initializer=normalized_columns_initializer(1.0),
            biases_initializer=None)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        # get gradients for local network
        if self.scope != 'global':
            self.a = tf.placeholder(shape=[None], dtype=tf.int32, name="a")
            self.v = tf.placeholder(shape=[None], dtype=tf.float32, name="v")
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32, name="adv")

            self.action_onehot = tf.one_hot(self.a, self.cfg.action_dim, dtype=tf.float32)
            self.action_probs = tf.reduce_sum(self.probs * self.action_onehot, axis=1)

            # batch_size = tf.shape(self.a)[0]
            # ind = tf.stack([tf.range(batch_size), self.a], axis=1)
            # self.action_probs = tf.gather_nd(self.probs, ind)

            # add entropy to loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs + 1e-6))

            # loss function
            self.value_loss = 0.5 * tf.reduce_sum(tf.square(self.v - tf.reshape(self.value, [-1])))
            self.policy_loss = -tf.reduce_sum(tf.log(self.action_probs + 1e-6) * self.adv)
            self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy

            # local gradients
            self.gradients = tf.gradients(self.loss, self.vars)
            self.gradients, _ = tf.clip_by_global_norm(self.gradients, 40.0)

            self.summaries = tf.summary.merge([
                tf.summary.scalar("entropy", tf.reduce_mean(self.entropy)),
                tf.summary.scalar("loss", self.loss),
                tf.summary.histogram("value", self.v),
                tf.summary.histogram("gradients0", self.gradients[0]),
                tf.summary.histogram("gradients2", self.gradients[2]),
                tf.summary.histogram("gradients4", self.gradients[4])
            ])

        # global optimizer for applying gradients
        else:
            # self.optimizer = tf.train.RMSPropOptimizer(1e-3, 0.99, 0.0, 1e-6)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

    def get_train_global_op(self, global_network):
        assert len(self.gradients) == len(global_network.vars)
        return global_network.optimizer.apply_gradients(
            zip(self.gradients, global_network.vars),
            global_step=get_or_create_global_step())

def copy_graph(_from, _to):
    return [vt.assign(vf) for vf, vt in zip(_from.vars, _to.vars)]


Transition = namedtuple('Transition', 'state rnn_state value action reward next_state done')

class Worker:

    def __init__(self, cfg, scope='worker'):
        self.cfg = cfg
        self.sess = tf.get_default_session()
        self.scope = scope
        self.local_network = Network(cfg, scope=self.scope)
        self.update_local_op = copy_graph(cfg.global_network, self.local_network)
        self.train_global_op = self.local_network.get_train_global_op(cfg.global_network)

    def work(self):

        env = gym.make("PongDeterministic-v3")
        state = env.reset()
        state = self.cfg.imgproc.process(self.sess, state)
        # state = np.stack([state] * 4, axis=2)

        episode_cnt = 0
        while not self.cfg.coord.should_stop():

            self.sess.run(self.update_local_op)

            transitions = []
            done = False
            rnn_state = self.local_network.state_init

            for _ in range(self.cfg.t_max):

                last_rnn_state = rnn_state
                action_probs, value, rnn_state = self.sess.run(
                    [self.local_network.probs,
                     self.local_network.value,
                     self.local_network.state_out],
                    feed_dict={self.local_network.X: np.expand_dims(state, 0),
                               self.local_network.state_in[0]: rnn_state[0],
                               self.local_network.state_in[1]: rnn_state[1]})
                assert not np.any(np.isnan(action_probs + [value]))

                action = np.random.choice(range(self.cfg.action_dim), p=action_probs[0])
                if self.scope == 'worker_demo':
                    # action = action_probs[0].argmax()
                    print('probs:', action_probs, 'action:', action, 'value:', value)
                    env.render()
                next_state, reward, done, _ = env.step(action)
                next_state = self.cfg.imgproc.process(self.sess, next_state)
                # next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                transitions.append(Transition(state=state, rnn_state=last_rnn_state,
                                              value=value, action=action, reward=reward,
                                              next_state=next_state, done=done))

                if done:
                    state = env.reset()
                    state = self.cfg.imgproc.process(self.sess, state)
                    # state = np.stack([state] * 4, axis=2)
                    break
                else:
                    state = next_state

            if done:
                reward = 0.
            else:
                reward = self.sess.run(
                    self.local_network.value,
                    feed_dict={self.local_network.X: np.expand_dims(transitions[-1].next_state, 0),
                               self.local_network.state_in[0]: rnn_state[0],
                               self.local_network.state_in[1]: rnn_state[1]})

            if self.scope == 'worker_demo':
                continue

            # minibatch
            xs, vs, acts, advs = [], [], [], []

            for t in transitions[::-1]:
                reward = t.reward + self.cfg.discount_factor * reward
                advantage = reward - t.value
                xs.append(t.state)
                vs.append(reward)
                acts.append(t.action)
                advs.append(advantage)
            xs = np.array(xs[::-1])
            vs = np.array(vs[::-1]).reshape([-1])
            acts = np.array(acts[::-1]).reshape([-1])
            advs = np.array(advs[::-1]).reshape([-1])

            # train global network
            episode_cnt += 1
            _, summaries = self.sess.run(
                [self.train_global_op, self.local_network.summaries],
                feed_dict={self.local_network.X: xs,
                           self.local_network.state_in[0]: self.local_network.state_init[0],
                           self.local_network.state_in[1]: self.local_network.state_init[1],
                           self.local_network.v: vs,
                           self.local_network.a: acts,
                           self.local_network.adv: advs})
            self.cfg.summary_writer.add_summary(summaries, episode_cnt)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    np.set_printoptions(precision=4, suppress=True)

    cfg = easydict.EasyDict(dict(
        discount_factor=0.999,
        action_dim=6,
        t_max=20,
    ))

    date_str = datetime.now().strftime("%m%d_%H%M%S")
    summaries_dir = os.path.abspath("./summary/a3c/" + date_str)
    if not os.path.exists(summaries_dir):
        os.makedirs(summaries_dir)
    cfg.summary_writer = tf.summary.FileWriter(
        summaries_dir, graph=tf.get_default_graph())

    with tf.device("/cpu:0"), tf.Session() as sess:

        cfg.global_network = Network(cfg)
        cfg.imgproc = ImageProcessor()
        # workers_num = multiprocessing.cpu_count()
        workers_num = 8
        tf.logging.info('#Workers: {}'.format(workers_num))
        workers = [Worker(cfg, scope='worker' + str(i)) for i in ['_demo'] + list(range(workers_num))]

        sess.run(tf.global_variables_initializer())

        cfg.coord = tf.train.Coordinator()
        threads = []
        for w in workers:
            th = threading.Thread(target=lambda: w.work())
            th.start()
            threads.append(th)
            time.sleep(1)
        cfg.coord.join(threads)


if __name__ == '__main__':
    main()
