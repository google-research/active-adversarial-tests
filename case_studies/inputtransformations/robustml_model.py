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
from defense import *
from inceptionv3 import model as inceptionv3_model
import tensorflow as tf

class InputTransformations(robustml.model.Model):
    def __init__(self, sess, defense):
        self._sess = sess
        self._input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        #self._input_single = tf.placeholder(tf.float32, (299, 299, 3))
        #input_expanded = tf.expand_dims(self._input, axis=0)

        if defense == 'crop':
            raise NotImplementedError("crop transformation not properly "
                                      "implemented yet")
            cropped_xs = defend_crop(self._input)
            self._logits, _ = inceptionv3_model(sess, cropped_xs)
            self._probs = tf.reduce_mean(tf.nn.softmax(self._logits), axis=0, keepdims=True)
        else:
            self._logits, _ = inceptionv3_model(sess, self._input)
            self._probs = tf.nn.softmax(self._logits)

        self._predictions = tf.argmax(self._probs, 1)

        if defense == 'bitdepth':
            self._defend = batched_defend_reduce
        elif defense == 'jpeg':
            self._defend = batched_defend_jpeg
        elif defense == 'crop':
            raise NotImplementedError("crop transformation not properly "
                                      "implemented yet")
            self._defend = lambda x: x # implemented as part of model so it's differentiable
        elif defense == 'quilt':
            self._defend = batched_make_defend_jpeg(sess)
        elif defense == 'tv':
            self._defend = batched_defend_tv
        elif defense == None:
            self._defend = lambda x: x
        else:
            raise ValueError('invalid defense: %s' % defense)

        #self._dataset = robustml.dataset.ImageNet((299, 299, 3))
        #self._threat_model = robustml.threat_model.L2(epsilon=0.05*299) # 0.05 * sqrt(299*299)
        # TODO: I think there is a factor sqrt(3) missing here

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x):
        x_defended = self.defend(x)
        return self._sess.run(self._predictions, {self._input: x_defended})

    # expose internals for white box attacks

    def defend(self, x):
        return self._defend(x)

    @property
    def input(self):
        return self._input

    @property
    def logits(self):
        return self._logits

    @property
    def predictions(self):
        return self._predictions
