#!/usr/bin/env python
from __future__ import print_function
from six import print_ as print
from six.moves import range, zip
from six.moves import cPickle

import os
import itertools
import collections
from datetime import datetime

import gym
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tfl
from tensorflow.contrib.framework import get_variables, get_or_create_global_step
from replay_queue import ReplayQueue

class ImageProcessor():
    def __init__(self):
        with tf.variable_scope("image_processor"):
            self.image = tf.placeholder(shape=[210, 160, 3], dtype=tf.uint8)
            self.output = tf.image.rgb_to_grayscale(self.image)
            self.output = tf.image.crop_to_bounding_box(self.output, 34, 0, 160, 160)
            self.output = tf.image.resize_images(
                self.output, (84, 84), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            self.output = tf.squeeze(self.output)

    def process(self, image):
        """ [210, 160, 3] -> [84, 84] """
        return self.output.eval({self.image: image})


class Estimator():
    def __init__(self, nA=4, scope="estimator", summaries_dir=None):
        self.sess = tf.get_default_session()
        self.nA = nA
        self.scope = scope
        self.summary_writer = None
        with tf.variable_scope(scope):
            self._build_model()
            if summaries_dir:
                summaries_dir = os.path.join(summaries_dir, scope)
                if not os.path.exists(summaries_dir):
                    os.makedirs(summaries_dir)
                self.summary_writer = tf.summary.FileWriter(summaries_dir, graph=self.sess.graph)

    def _build_model(self):
        self.X = tf.placeholder(shape=[None, 84, 84, 4], dtype=tf.uint8, name="X")
        self.y = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        self.a = tf.placeholder(shape=[None], dtype=tf.int32, name="a")

        X = tf.to_float(self.X) / 255.0
        conv1 = tfl.conv2d(X, 32, 8, 4)
        conv2 = tfl.conv2d(conv1, 64, 4, 2)
        conv3 = tfl.conv2d(conv2, 64, 3, 1)
        flattened = tfl.flatten(conv3)
        fc1 = tfl.fully_connected(flattened, 512)
        self.predictions = tfl.fully_connected(
            fc1, self.nA, activation_fn=None)

        batch_size = tf.shape(self.a)[0]
        ind = tf.pack([tf.range(batch_size), self.a], axis=1)
        self.action_predictions = tf.gather_nd(self.predictions, ind)

        self.network_params = get_variables(self.scope)

        self.loss = tf.reduce_mean(tf.squared_difference(self.y, self.action_predictions))
        self.train_op = tf.train.RMSPropOptimizer(0.0025, 0.99, 0.0, 1e-6).minimize(
            self.loss, global_step=get_or_create_global_step())

        self.summaries = tf.summary.merge([
            tf.summary.scalar("loss", self.loss),
            tf.summary.histogram("a", tf.argmax(self.predictions, axis=1)),
            tf.summary.histogram("max_q", tf.reduce_max(self.predictions)),
        ])

    def predict(self, s):
        """ state: [?, 84, 84, 4] -> q_values: [?, nA] """
        return self.predictions.eval({self.X: s})

    def update(self, s, a, y):
        """ ([?, 84, 84, 4], [?], [?]) -> loss """
        _, loss, summaries, global_step = self.sess.run(
            [self.train_op, self.loss, self.summaries,
             get_or_create_global_step()],
            feed_dict={self.X: s, self.a: a, self.y: y})
        if self.summary_writer:
            self.summary_writer.add_summary(summaries, global_step)
        return loss


def copy_params(sess, e1, e2):
    sess.run([v2.assign(v1) for v1, v2 in
              zip(e1.network_params, e2.network_params)])


def make_epsilon_greed_policy(estimator, nA):
    def policy_fn(observation, epsilon):
        action_probs = np.ones(nA, dtype=np.float) * epsilon / nA
        fake_batch = np.expand_dims(observation, 0)
        q_values = estimator.predict(fake_batch)
        best_action = np.argmax(q_values)
        action_probs[best_action] += 1.0 - epsilon
        return action_probs
    return policy_fn


def dqn():

    num_episodes = 10000
    replay_memory_size = 1000000
    replay_memory_init_size = 50000
    replay_repeat = 5
    update_target_estimator_every = 10000
    discount_factor = 0.99
    epsilon_start = 1.0
    epsilon_end = 0.1
    epsilon_decay_steps = 500000
    batch_size = 32
    show_per_i_episode = 100

    env = gym.make("Breakout-v0")
    env.render()
    nA = 4  # or env.action_space.n

    tf.reset_default_graph()

    date_str = datetime.now().strftime("%m%d_%H%M%S")
    summaries_dir = os.path.abspath("./summary/dqn/" + date_str)

    sess = tf.InteractiveSession()

    q_estimator = Estimator(nA=nA, scope="q", summaries_dir=summaries_dir)
    target_estimator = Estimator(nA=nA, scope="target_q")

    image_processor = ImageProcessor()

    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    checkpoint_dir = os.path.abspath("./checkpoints/dqn/" + date_str)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, "model")

    Transition = collections.namedtuple("Transition", "state action reward next_state done")
    replay_memory = ReplayQueue(replay_memory_size)
    # saver.restore(sess, "checkpoints/dqn/XXXX_XXXXXX/model")

    total_t = sess.run(tf.contrib.framework.get_global_step())
    epsilons = np.linspace(epsilon_start, epsilon_end, epsilon_decay_steps)

    def eps_fn():
        return epsilons[min(total_t, epsilon_decay_steps - 1)]

    policy_fn = make_epsilon_greed_policy(q_estimator, nA)

    # human control
    human_control = False  # not useful?
    if human_control:
        print("Replay memory from human control data...")
        lst = cPickle.load(open("human.data", "rb"))
        for l in lst:
            first_image = image_processor.process(l[0])
            state = np.stack([first_image] * 4, axis=2)
            for i in range(1, len(l) - 1):
                next_image, action, reward, done = l[i]
                # if done:
                #     reward -= 10
                next_image = image_processor.process(next_image)
                next_state = np.append(state[:, :, 1:], np.expand_dims(next_image, 2), axis=2)
                t = Transition(state, action, reward, next_state, done)
                # for i in range(5):  # weight
                replay_memory.push(t)
                state = next_state
    else:
        state = env.reset()
        state = image_processor.process(state)
        state = np.stack([state] * 4, axis=2)
        for i in range(replay_memory_init_size):
            action_probs = policy_fn(state, eps_fn())
            action = np.random.choice(range(nA), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            # if done:
            #     reward -= 10
            next_state = image_processor.process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            replay_memory.push(Transition(state, action, reward, next_state, done))
            if done:
                state = env.reset()
                state = image_processor.process(state)
                state = np.stack([state] * 4, axis=2)
            else:
                state = next_state
            if i % 100 == 0:
                print(
                    "\rReplay memory...[{}/{}]".format(i, replay_memory_init_size), end="", flush=True)
        print()

    for i_episode in range(1, num_episodes + 1):

        state = env.reset()
        if i_episode % show_per_i_episode == 0:
            env.render()
        state = image_processor.process(state)
        state = np.stack([state] * 4, axis=2)
        episode_reward = 0
        episode_length = 0

        for t in itertools.count(1):

            if total_t % update_target_estimator_every == 0:
                copy_params(sess, q_estimator, target_estimator)
                print("\nCopied model parameters to target estimator.")

            action_probs = policy_fn(state, eps_fn())
            action = np.random.choice(range(nA), p=action_probs)
            next_state, reward, done, _ = env.step(action)
            # if done:
            #     reward -= 10
            episode_reward += reward
            if i_episode % show_per_i_episode == 0:
                env.render()
            next_state = image_processor.process(next_state)
            next_state = np.append(state[:, :, 1:], np.expand_dims(next_state, 2), axis=2)
            replay_memory.push(Transition(state, action, reward, next_state, done))

            for _ in range(replay_repeat):
                samples = replay_memory.sample(batch_size)
                states_batch, action_batch, reward_batch, next_states_batch, done_batch =\
                    map(np.array, zip(*samples))

                # double dqn
                # q_values_next = q_estimator.predict(next_states_batch)
                # best_actions = np.argmax(q_values_next, axis=1)
                # q_values_next_target = target_estimator.predict(next_states_batch)
                # targets_batch = reward_batch + (1 - done_batch) *\
                #     discount_factor * q_values_next_target[range(batch_size), best_actions]

                q_values_next_target = target_estimator.predict(next_states_batch)
                targets_batch = reward_batch + (1 - done_batch) *\
                    discount_factor * np.amax(q_values_next_target, axis=1)

                loss = q_estimator.update(states_batch, action_batch, targets_batch)

            print(
                "\rStep {}({}) episode {}/{} reward {} loss {:.6f}"
                .format(t, total_t, i_episode, num_episodes, episode_reward, loss),
                end=" " * 10, flush=True)

            if done:
                print()
                episode_length = t
                break

            state = next_state
            total_t += 1

        if i_episode % 100 == 0:
            print('Saved!')
            saver.save(sess, checkpoint_path)

        if q_estimator.summary_writer:
            episode_summary = tf.Summary()
            episode_summary.value.add(simple_value=episode_reward,
                                      node_name="episode_reward",
                                      tag="episode_reward")
            episode_summary.value.add(simple_value=episode_length,
                                      node_name="episode_length",
                                      tag="episode_length")
            q_estimator.summary_writer.add_summary(episode_summary, total_t)
            q_estimator.summary_writer.flush()

    saver.save(sess, checkpoint_path)
    sess.close()


if __name__ == '__main__':
    dqn()
