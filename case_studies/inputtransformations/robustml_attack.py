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

import robustml
import sys
import tensorflow as tf
import numpy as np

class BPDA(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=1000, learning_rate=0.1, lam=1e-6, debug=False):
        self._sess = sess

        self._model = model
        self._input = model.input
        self._l2_input = tf.placeholder(tf.float32, self._input.shape, name="l2_input") # using BPDA, so we want this to pass the original adversarial example
        self._original = tf.placeholder(tf.float32, self._input.shape, name="original")
        self._label = tf.placeholder(tf.int32, (None,), name="label")
        one_hot = tf.one_hot(self._label, 1000)
        #ensemble_labels = tf.tile(one_hot, (model.logits.shape[0], 1))
        self._l2 = tf.sqrt(2*tf.nn.l2_loss(self._l2_input - self._original))
        self._xent = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=model.logits, labels=one_hot))
        self._loss = lam * tf.maximum(self._l2 - epsilon, 0) + self._xent
        self._grad, = tf.gradients(self._loss, self._input)

        self._epsilon = epsilon
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        adv = np.copy(x)
        for i in range(self._max_steps):
            adv_def = self._model.defend(adv)
            p, ll2, lxent, g = self._sess.run(
                [self._model.predictions, self._l2, self._xent, self._grad],
                {self._input: adv_def, self._label: y, self._l2_input: adv, self._original: x}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, xent loss = %g, l2 loss = %g (max %g), (true %d, predicted %s)' % (
                        i+1,
                        self._max_steps,
                        lxent,
                        ll2,
                        self._epsilon,
                        y,
                        p
                    ),
                    file=sys.stderr
                )
            is_adv = np.logical_and(y != p, ll2 < self._epsilon)
            print(is_adv.sum())
            if np.all(is_adv):
            #if y not in p and ll2 < self._epsilon:
                # we're done
                #if self._debug:
                print('returning early', file=sys.stderr)
                break
            g *= (~is_adv).astype(int).reshape(-1, 1, 1, 1)
            adv += self._learning_rate * g
            adv = np.clip(adv, 0, 1)

        adv_l2 = np.sqrt(((adv - x)**2).sum((1, 2, 3), keepdims=True))
        factor = self._epsilon / adv_l2
        factor = np.minimum(factor, np.ones_like(factor))
        diff = adv - x
        adv = diff*factor + x

        return adv

