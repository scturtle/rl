#!/usr/bin/env python
from __future__ import print_function
from six import print_ as print
from six.moves import range, zip

import os
import itertools
from collections import namedtuple
from datetime import datetime

import easydict
import numpy as np
import gym
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from tensorflow.contrib.framework import get_variables, get_or_create_global_step
from replay_queue import ReplayQueue


class NAF:
    def __init__(self, cfg, scope="q", summaries_dir=None):
        self.cfg = cfg  # status_dim, action_dim, hidden_dim, learning_rate, l2_reg, is_training
        self.sess = tf.get_default_session()
        self.scope = scope
        with tf.variable_scope(scope):
            self._build()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, "summaries_" + scope)
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                print("Summary dir:", summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir, graph=self.sess.graph)

    def fc(self, inp, dim, activation_fn=None, use_batch_norm=False):
        weights_initializer = tfl.xavier_initializer()
        weights_regularizer = tfl.l2_regularizer(self.cfg.l2_reg)
        batch_norm_params = {} if not use_batch_norm else {
            'normalizer_fn': tfl.batch_norm,
            'normalizer_params': {
                "is_training": self.cfg.is_training
            }
        }
        return tfl.fully_connected(
            inp,
            dim,
            activation_fn=activation_fn,
            weights_initializer=weights_initializer,
            weights_regularizer=weights_regularizer,
            biases_initializer=tf.constant_initializer(0.0),
            **batch_norm_params)

    @staticmethod
    def to_lower_triangle(ls, n):
        tot = (n * n + n) // 2
        idx = tot * np.ones((n, n), dtype=np.int32)
        i_s, j_s = np.tril_indices(n - 1)
        i_s = [i + 1 for i in i_s]
        idx[(i_s, j_s)] = range(tot - n)
        idx[(range(n), range(n))] = range(tot - n, tot)
        idx = np.expand_dims(idx, 2)
        zero = tf.constant([0.], dtype=tf.float32)

        def tri(l):
            non_diag, diag = tf.split_v(l, [tot - n, n])
            l = tf.concat_v2([non_diag, tf.exp(diag), zero], 0)
            return tf.gather_nd(l, idx)

        return tf.map_fn(tri, ls)

    def _build(self):
        action_size = self.cfg.action_dim[0]
        self.x = tf.placeholder(
            shape=(None, ) + tuple(self.cfg.status_dim), dtype=tf.float32, name='x')
        self.u = tf.placeholder(
            shape=(None, ) + tuple(self.cfg.action_dim), dtype=tf.float32, name='u')
        self.y = tf.placeholder(shape=(None), dtype=tf.float32, name='y')

        h = self.fc(self.x, self.cfg.hidden_dim, activation_fn=tf.nn.relu)
        h = self.fc(h, self.cfg.hidden_dim, activation_fn=tf.nn.relu)
        self.V = tf.squeeze(self.fc(h, 1))
        self.mu = self.fc(h, action_size)
        self.l = self.fc(h, action_size * (action_size + 1) // 2)
        self.L = self.to_lower_triangle(self.l, action_size)
        self.P = tf.matmul(self.L, tf.transpose(self.L, (0, 2, 1)), name="P")

        diff_u = tf.expand_dims(self.u - self.mu, 1)
        self.A = -tf.matmul(diff_u, tf.matmul(self.P, tf.transpose(diff_u, (0, 2, 1))))
        self.A = tf.squeeze(tf.reshape(self.A, (-1, 1)), name="A")
        self.Q = self.A + self.V
        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.Q))
        self.train_op = tf.train.AdamOptimizer(learning_rate=self.cfg.learning_rate).minimize(
            self.loss, global_step=get_or_create_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss), tf.summary.histogram("mu", self.mu),
            tf.summary.histogram("Q", self.Q)
        ])

    def predict(self, x):
        return self.mu.eval({self.x: x})

    def values(self, x):
        return self.V.eval({self.x: x})

    def update(self, x, u, y):
        _, loss, summaries, global_step = self.sess.run(
            [self.train_op, self.loss, self.summaries, get_or_create_global_step()],
            feed_dict={self.x: x, self.u: u, self.y: y})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def copy_params(sess, _from, _to, tau):
    sess.run([
        vt.assign(tau * vf + (1 - tau) * vt)
        for vf, vt in zip(get_variables(_from.scope), get_variables(_to.scope))
    ])


def main():
    num_episodes = 300
    replay_memory_init_size = 500
    replay_memory_size = 50000
    replay_step_limit = 100
    batch_size = 128
    discount_factor = 0.99
    max_step = 200
    replay_repeat = 5
    update_target_per_steps = 100
    tau = 0.0003
    display = True

    Transition = namedtuple("Transition", "state action reward next_state done")
    replay_memory = ReplayQueue(replay_memory_size)

    env = gym.make('Pendulum-v0')
    status_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    if display:
        env.reset()
        env.render()

    naf_cfg = easydict.EasyDict(
        dict(
            status_dim=status_dim,
            action_dim=action_dim,
            hidden_dim=200,
            learning_rate=0.0003,
            l2_reg=0.0003,
            is_training=True))

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    date_str = datetime.now().strftime("%m%d_%H%M%S")
    summaries_dir = os.path.abspath("./summary/naf/" + date_str)

    q_estimator = NAF(naf_cfg, scope="q", summaries_dir=summaries_dir)
    target_estimator = NAF(naf_cfg, sess=sess, scope="target")

    sess.run(tf.global_variables_initializer())
    copy_params(sess, q_estimator, target_estimator, 1)

    saver = tf.train.Saver()
    checkpoint_dir = os.path.abspath("./checkpoints/naf/" + date_str)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model")
    # saver.restore(sess, "checkpoints/naf/XXXX_XXXXXX/model")

    state = env.reset()
    k = 0
    for _ in range(replay_memory_init_size):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        replay_memory.push(Transition(state, action, reward, next_state, done))
        if done or k >= replay_step_limit:
            k = 0
            state = env.reset()
        else:
            k += 1
            state = next_state

    for i_episode in range(1, num_episodes + 1):

        state = env.reset()
        episode_reward = 0
        if display:
            env.render()

        for i_step in itertools.count(1):
            action = q_estimator.predict(np.expand_dims(state, 0))[0]
            action += np.random.randn(action_dim[0]) / i_episode  # TODO
            action = np.minimum(env.action_space.high, np.maximum(env.action_space.low, action))
            next_state, reward, done, _ = env.step(action)
            if display:
                env.render()
            episode_reward += reward
            replay_memory.push(Transition(state, action, reward, next_state, done))

            for _ in range(replay_repeat):
                samples = replay_memory.sample(batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch =\
                    map(np.array, zip(*samples))
                target_values = target_estimator.values(next_states_batch)
                target_batch = reward_batch + (1 - done_batch) * discount_factor * target_values
                loss = q_estimator.update(states_batch, action_batch, target_batch)

            print(
                "\rEpisode {}/{} step {} action {:.3f} reward {:.3f} loss {:.6f}"
                .format(i_episode, num_episodes, i_step, action[0], episode_reward, loss),
                end='',
                flush=True)

            if i_step % update_target_per_steps == 0:
                copy_params(sess, q_estimator, target_estimator, tau * update_target_per_steps)

            if done or i_step >= max_step:
                print()
                break
            state = next_state

        if i_episode % 50 == 0:
            print('Saved!')
            saver.save(sess, checkpoint_path)

        if q_estimator.summary_writer:
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=episode_reward,
                                      node_name="episode_reward",
                                      tag="episode_reward")
            q_estimator.summary_writer.add_summary(episode_summary, i_episode)
            q_estimator.summary_writer.flush()

    saver.save(sess, checkpoint_path)
    sess.close()


if __name__ == '__main__':
    main()
