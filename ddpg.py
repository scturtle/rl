from __future__ import print_function
from six import print_ as print
from six.moves import range, zip

import os
import itertools
import collections
from datetime import datetime

import gym
import easydict
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from tensorflow.contrib.framework import get_variables, get_or_create_global_step
from replay_queue import ReplayQueue


class PolicyEstimator:

    def __init__(self, cfg, sess, scope, summaries_dir=None):
        self.cfg = cfg
        self.sess = sess
        self.scope = scope
        with tf.variable_scope(scope):
            self._build()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, scope)
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir, graph=sess.graph)

    def _build(self):
        self.x = tf.placeholder(
            shape=(None, self.cfg.status_size), dtype=tf.float32, name='x')
        self.action_gradient = tf.placeholder(
            shape=(None, self.cfg.action_size), dtype=tf.float32, name='action_gradient')

        x = self.x

        h = tfl.fully_connected(x, 400)
        h = tfl.fully_connected(h, 300)
        self.a = tfl.fully_connected(
            h, self.cfg.action_size,
            activation_fn=tf.nn.tanh  # [-1, 1]
        )

        self.network_params = get_variables(self.scope)
        # combine the gradients
        gradient = tf.gradients(self.a, self.network_params, -self.action_gradient)
        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).apply_gradients(
            zip(gradient, self.network_params))

        self.summaries = tf.summary.merge([tf.summary.histogram("a", self.a)])

    def update(self, x, action_gradient):
        _, summaries, global_step = self.sess.run(
            [self.train_op, self.summaries, get_or_create_global_step()],
            feed_dict={self.x: x, self.action_gradient: action_gradient})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)

    def predict(self, x):
        return self.a.eval({self.x: x})

class ValueEstimator:

    def __init__(self, cfg, sess, scope, summaries_dir=None):
        self.cfg = cfg
        self.sess = sess
        self.scope = scope
        with tf.variable_scope(scope):
            self._build()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, scope)
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir, graph=sess.graph)

    def _build(self):
        self.x = tf.placeholder(
            shape=(None, self.cfg.status_size), dtype=tf.float32, name='x')
        self.a = tf.placeholder(
            shape=(None, self.cfg.action_size), dtype=tf.float32, name='a')
        self.y = tf.placeholder(shape=(None, ), dtype=tf.float32, name='y')

        x = self.x

        h = tfl.fully_connected(x, 400)
        h = tfl.fully_connected(tf.concat_v2([h, self.a], 1), 300)
        q = tfl.fully_connected(
            h, 1,
            activation_fn=None,
            weights_regularizer=tfl.l2_regularizer(1e-2))
        self.q = tf.squeeze(q)

        self.network_params = get_variables(self.scope)

        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.q))
        self.train_op = tf.train.AdamOptimizer(self.cfg.learning_rate).minimize(
            self.loss, global_step=get_or_create_global_step())

        batch_size = tf.cast(tf.shape(self.a)[0], tf.float32)
        self.action_gradient = tf.div(tf.gradients(self.q, self.a), batch_size)

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("q", self.q),
        ])

    def update(self, x, a, y):
        _, loss, summaries, global_step = self.sess.run(
            [self.train_op, self.loss, self.summaries, get_or_create_global_step()],
            feed_dict={self.x: x, self.a: a, self.y: y})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss

    def gradient(self, x, a):
        return self.action_gradient.eval({self.x: x, self.a: a})

    def predict(self, x, a):
        return self.q.eval({self.x: x, self.a: a})

def copy_params(sess, _from, _to, tau):
    assert len(_from.network_params) == len(_to.network_params)
    sess.run([
        vt.assign(tau * vf + (1 - tau) * vt)
        for vf, vt in zip(_from.network_params, _to.network_params)
    ])

def ou(x, mu, theta, sigma):
    return theta * (mu - x) + sigma * np.random.randn(x.size)

def main():
    num_episodes = 500
    discount_factor = 0.99
    batch_size = 64  # 16 for deep
    actor_learning_rate = 1e-4
    critic_learning_rate = 1e-3
    tau = 0.001
    update_target_step = 100
    replay_memory_init_size = 100000
    replay_memory_size = 1000000
    replay_repeat = 5
    replay_max_step = 100
    max_step = 200
    display = True

    Transition = collections.namedtuple("Transition", "state action reward next_state done")
    replay_memory = ReplayQueue(replay_memory_size)

    env = gym.make('Pendulum-v0')
    if display:
        env.reset()
        env.render()

    status_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    cfg = easydict.EasyDict(dict(
        status_size=status_dim[0],
        action_size=action_dim[0],
        use_batch_norm=True,
        is_training=True
    ))
    actor_cfg = critic_cfg = cfg
    actor_cfg.learning_rate = actor_learning_rate
    critic_cfg.learning_rate = critic_learning_rate

    tf.reset_default_graph()
    sess = tf.InteractiveSession()

    date_str = datetime.now().strftime("%m%d_%H%M%S")
    summaries_dir = os.path.abspath("./summary/ddpg/" + date_str)

    q_actor = PolicyEstimator(actor_cfg, sess, 'q_actor', summaries_dir)
    q_critic = ValueEstimator(critic_cfg, sess, 'q_critic', summaries_dir)

    target_actor = PolicyEstimator(actor_cfg, sess, 'target_actor')
    target_critic = ValueEstimator(critic_cfg, sess, 'target_critic')

    sess.run(tf.global_variables_initializer())
    copy_params(sess, q_actor, target_actor, 1)
    copy_params(sess, q_critic, target_critic, 1)

    saver = tf.train.Saver()
    checkpoint_dir = os.path.abspath("./checkpoints/ddpg/" + date_str)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    # init replay memory
    state = env.reset()
    k = 0
    for _ in range(replay_memory_init_size):
        action = env.action_space.sample()
        next_state, reward, done, _ = env.step(action)
        action /= 2.
        replay_memory.push(Transition(state, action, reward, next_state, done))
        if done or k >= replay_max_step:
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
            action = target_actor.predict(np.expand_dims(state, 0))[0]
            # action += np.random.randn(action_dim[0]) / i_episode
            action += 1 / i_episode * ou(action, 0, 0.15, 0.2)
            next_state, reward, done, _ = env.step(action * 2)
            if display:
                env.render()
            episode_reward += reward
            replay_memory.push(Transition(state, action, reward, next_state, done))

            for _ in range(replay_repeat):
                samples = replay_memory.sample(batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch =\
                    map(np.array, zip(*samples))

                target_actions = target_actor.predict(next_states_batch)
                target_values = target_critic.predict(next_states_batch, target_actions)
                target_batch = reward_batch + (1 - done_batch) * discount_factor * target_values

                # update the critic
                loss = q_critic.update(states_batch, action_batch, target_batch)

                # update the actor policy
                a = q_actor.predict(states_batch)
                g = q_critic.gradient(states_batch, a)[0]
                q_actor.update(states_batch, g)

            print("\rEpisode {}/{} step {} action {:.3f} reward {:.3f} loss {:.3f}"
                  .format(i_episode, num_episodes, i_step, action[0], episode_reward, loss),
                  end='', flush=True)

            if i_step % update_target_step == 0:
                copy_params(sess, q_actor, target_actor, tau * update_target_step)
                copy_params(sess, q_critic, target_critic, tau * update_target_step)

            if done or i_step >= max_step:
                print()
                break
            state = next_state

        if i_episode % 50 == 0:
            print('Saved!')
            saver.save(sess, checkpoint_path)

        if q_critic.summary_writer:
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=episode_reward,
                                      node_name="episode_reward",
                                      tag="episode_reward")
            q_critic.summary_writer.add_summary(episode_summary, i_episode)
            q_critic.summary_writer.flush()


if __name__ == '__main__':
    main()
