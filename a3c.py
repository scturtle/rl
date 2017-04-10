#!/usr/bin/env python
from __future__ import print_function
from six.moves import range, zip

import os
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

class Network:

    def __init__(self, cfg, scope='global'):
        self.cfg = cfg
        self.sess = tf.get_default_session()
        self.scope = scope
        self.summary_writer = cfg.summary_writer
        with tf.variable_scope(scope):
            self._build()

    def _build(self):
        self.X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")

        X = tf.to_float(self.X) / 255.0
        conv1 = tfl.conv2d(X, 16, 8, 4)
        conv2 = tfl.conv2d(conv1, 32, 4, 2)
        fc1 = tfl.fully_connected(tfl.flatten(conv2), 256)

        # TODO LSTM

        # action probability
        self.probs = tfl.fully_connected(
            fc1, self.cfg.action_dim,
            activation_fn=tf.nn.softmax, biases_initializer=None)
        # state value
        self.value = tfl.fully_connected(
            fc1, 1, activation_fn=None, biases_initializer=None)

        self.vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

        # get gradients for local network
        if self.scope != 'global':
            self.adv = tf.placeholder(shape=[None], dtype=tf.float32, name="adv")
            self.v = tf.placeholder(shape=[None], dtype=tf.float32, name="v")
            self.a = tf.placeholder(shape=[None], dtype=tf.int32, name="a")

            batch_size = tf.shape(self.a)[0]
            ind = tf.stack([tf.range(batch_size), self.a], axis=1)
            self.action_probs = tf.gather_nd(self.probs, ind)

            # add entropy to loss to encourage exploration
            self.entropy = -tf.reduce_sum(self.probs * tf.log(self.probs))

            # loss function
            self.value_loss = tf.reduce_sum(tf.square(self.v - tf.reshape(self.value, [-1])))
            self.policy_loss = -tf.log(self.action_probs) * self.adv
            self.loss = 0.5 * self.value_loss + self.policy_loss - 0.01 * self.entropy

            # local gradients
            gradients = tf.gradients(self.loss, self.vars)
            self.gradients, _ = tf.clip_by_global_norm(gradients, 40.0)

            self.summaries = tf.summary.merge([
                tf.summary.scalar("loss", tf.reduce_mean(self.loss)),
                tf.summary.histogram("value", self.v),
                tf.summary.histogram("gradients0", self.gradients[0]),
                tf.summary.histogram("gradients2", self.gradients[2]),
                tf.summary.histogram("gradients4", self.gradients[4])
            ])

        # global optimizer for applying gradients
        else:
            self.optimizer = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6)

    def get_train_global_op(self, global_network):
        assert len(self.gradients) == len(global_network.vars)
        return global_network.optimizer.apply_gradients(
            zip(self.gradients, global_network.vars),
            global_step=get_or_create_global_step())

def copy_graph(_from, _to):
    return [vt.assign(vf) for vf, vt in zip(_from.vars, _to.vars)]


Transition = namedtuple('Transition', 'state value action reward next_state done')

class Worker:

    def __init__(self, cfg, scope='worker'):
        self.cfg = cfg
        self.sess = tf.get_default_session()
        self.scope = scope
        self.local_network = Network(cfg, scope=self.scope)
        self.update_local_op = copy_graph(cfg.global_network, self.local_network)
        self.train_global_op = self.local_network.get_train_global_op(cfg.global_network)

    def work(self):

        env = gym.make("Breakout-v0")
        state = env.reset()
        state = self.cfg.imgproc.process(self.sess, state)
        state = np.stack([state] * 4, axis=2)

        episode_cnt = 0
        while not self.cfg.coord.should_stop():

            self.sess.run(self.update_local_op)

            transitions = []
            done = False
            for _ in range(self.cfg.t_max):

                action_probs, value = self.sess.run(
                    [self.local_network.probs, self.local_network.value],
                    feed_dict={self.local_network.X: np.expand_dims(state, 0)})

                action = np.random.choice(range(self.cfg.action_dim), p=action_probs[0])
                next_state, reward, done, _ = env.step(action)
                next_state = self.cfg.imgproc.process(self.sess, next_state)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
                transitions.append(Transition(state=state, value=value, action=action,
                                              reward=reward, next_state=next_state, done=done))

                if done:
                    state = env.reset()
                    state = self.cfg.imgproc.process(self.sess, state)
                    state = np.stack([state] * 4, axis=2)
                    break
                else:
                    state = next_state

            if done:
                reward = 0.
            else:
                reward = self.sess.run(
                    self.local_network.value,
                    feed_dict={self.local_network.X: np.expand_dims(transitions[-1].next_state, 0)})

            # minibatch
            Xs, vs, acts, advs = [], [], [], []

            transitions.reverse()
            for t in transitions:
                reward = t.reward + self.cfg.discount_factor * reward
                advantage = reward - t.value
                Xs.append(t.state)
                vs.append(reward)
                acts.append(t.action)
                advs.append(advantage)
            Xs = np.array(Xs)
            vs = np.array(vs).reshape([-1])
            acts = np.array(acts).reshape([-1])
            advs = np.array(advs).reshape([-1])

            # train global network
            episode_cnt += 1
            _, summaries = self.sess.run([self.train_global_op, self.local_network.summaries],
                                         feed_dict={self.local_network.X: Xs,
                                                    self.local_network.v: vs,
                                                    self.local_network.a: acts,
                                                    self.local_network.adv: advs})
            self.cfg.summary_writer.add_summary(summaries, episode_cnt)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    cfg = easydict.EasyDict(dict(
        discount_factor=0.999,
        action_dim=4,
        t_max=30,
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
        workers_num = multiprocessing.cpu_count()
        tf.logging.info('#Workers: {}'.format(workers_num))
        workers = [Worker(cfg, scope='worker' + str(i)) for i in range(workers_num)]

        sess.run(tf.global_variables_initializer())

        cfg.coord = tf.train.Coordinator()
        threads = []
        for w in workers:
            th = threading.Thread(target=lambda: w.work())
            th.start()
            threads.append(th)
        cfg.coord.join(threads)


if __name__ == '__main__':
    main()
