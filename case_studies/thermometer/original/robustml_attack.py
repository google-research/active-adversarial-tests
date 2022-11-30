# Copyright 2022 The Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from robustml_model import LEVELS
from discretization_utils import discretize_uniform

import numpy as np
from discretization_attacks import adv_lspga


class Attack:
    def __init__(self, sess, model, epsilon, num_steps=30, step_size=1, batch_size=1, n_classes=10):
        self._sess = sess
        self.model = model
        self.num_steps = num_steps
        self.step_size = step_size

        self.xs = tf.Variable(np.zeros((batch_size, 32, 32, 3), dtype=np.float32),
                                    name='modifier')
        self.orig_xs = tf.placeholder(tf.float32, [None, 32, 32, 3])

        self.ys = tf.placeholder(tf.int32, [None])

        self.epsilon = epsilon * 255

        delta = tf.clip_by_value(self.xs, 0, 255) - self.orig_xs
        delta = tf.clip_by_value(delta, -self.epsilon, self.epsilon)

        self.do_clip_xs = tf.assign(self.xs, self.orig_xs+delta)

        compare = tf.constant((256.0/LEVELS)*np.arange(-1,LEVELS-1).reshape((1,1,1,1,LEVELS)),
                              dtype=tf.float32)
        inner = tf.reshape(self.xs,(-1, 32, 32, 3, 1)) - compare
        inner = tf.maximum(tf.minimum(inner/(256.0/LEVELS), 1.0), 0.0)

        self.therm = tf.reshape(inner, (-1, 32, 32, LEVELS*3))

        self.logits = logits = model(self.therm)

        self.uniform = discretize_uniform(self.xs/255.0, levels=LEVELS, thermometer=True)
        self.real_logits = model(self.uniform)

        label_mask = tf.one_hot(self.ys, n_classes)
        correct_logit = tf.reduce_sum(label_mask * logits, axis=1)
        wrong_logit = tf.reduce_max((1-label_mask) * logits - 1e4*label_mask, axis=1)

        self.loss = (correct_logit - wrong_logit)

        start_vars = set(x.name for x in tf.global_variables())
        optimizer = tf.train.AdamOptimizer(step_size*1)
        self.grad = tf.sign(tf.gradients(self.loss, self.xs)[0])

        grad,var = optimizer.compute_gradients(self.loss, [self.xs])[0]
        self.train = optimizer.apply_gradients([(tf.sign(grad),var)])

        end_vars = tf.global_variables()
        self.new_vars = [x for x in end_vars if x.name not in start_vars]

    #@profile
    def perturb(self, x, y, sess, feed_dict={}):
        sess.run(tf.variables_initializer(self.new_vars))
        sess.run(self.xs.initializer)
        sess.run(self.do_clip_xs,
                 {self.orig_xs: x})

        for i in range(self.num_steps):

            t = sess.run(self.uniform)
            sess.run(self.train, feed_dict={self.ys: y,
                                            self.therm: t,
                                            **feed_dict})
            sess.run(self.do_clip_xs,
                     {self.orig_xs: x})

        x_batch_adv = sess.run(self.xs)

        return x_batch_adv

    def run(self, x, y, target, feed_dict={}):
        if len(x.shape) == 3:
            x = np.array([x])
            y = np.array([y])
        if target is not None:
            raise NotImplementedError
        return self.perturb(x * 255.0, y, self._sess, feed_dict) / 255.0


class LSPGDAttack:
    def __init__(self, sess, model, epsilon, num_steps=7, step_size=0.1,
        use_labels=True, n_classes=10):
        # ATTENTION: use_labels is a modification from AUTHOR
        self._sess = sess
        self.model = model

        self.xin = tf.placeholder(tf.float32, (None, 32, 32, 3))
        if use_labels:
            self.yin = tf.placeholder(tf.int64, shape=None)
            self.y_filled = tf.one_hot(
                self.yin,
                n_classes)
        else:
            self.yin = None

        steps = num_steps
        eps = epsilon
        attack_step = step_size

        projection_fn = tf.identity

        self.attack = adv_lspga(self.xin, model, discretize_uniform,
                           projection_fn, 16, tf.constant(False), steps, eps,
                           attack_step, thermometer=True, noisy_grads=False,
                           y=self.y_filled)



    def perturb(self, x, y, sess, feed_dict={}):
        if self.yin is None:
            x_batch_adv = sess.run(self.attack,
                               {self.xin: x/255.0, **feed_dict})
        else:
            x_batch_adv = sess.run(self.attack,
                                   {self.xin: x/255.0, **feed_dict,
                                    self.yin: y})

        return x_batch_adv

    def run(self, x, y, target, feed_dict={}):
        if len(x.shape) == 3:
            x = np.array([x])
            y = np.array([y])
        if target is not None:
            raise NotImplementedError
        return self.perturb(x * 255.0, y, self._sess, feed_dict) / 255.0
