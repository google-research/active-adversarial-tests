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
import tensorflow as tf
from discretization_utils import discretize_uniform
import numpy as np
from cifar_model import Model

LEVELS = 16

class Thermometer(robustml.model.Model):
    def __init__(self, sess, epsilon):
        self._sess = sess

        self._x = tf.placeholder(tf.float32, (None, 32, 32, 3))
        self._encode = discretize_uniform(self._x/255.0, levels=LEVELS, thermometer=True)

        self._model = Model(
            'checkpoints/original_thermometer_wrn/thermometer_advtrain/',
            sess,
            tiny=False,
            mode='eval',
            thermometer=True,
            levels=LEVELS
        )

        self._dataset = robustml.dataset.CIFAR10()
        self._threat_model = robustml.threat_model.Linf(epsilon=epsilon/255.0)

    @property
    def dataset(self):
        return self._dataset

    @property
    def threat_model(self):
        return self._threat_model

    def classify(self, x, skip_encoding=False):
        x = x * 255.0
        if not skip_encoding:
            # first encode the input, then classify it
            x = self.encode(x)
        return self._sess.run(self._model.predictions, {self._model.x_input: x})

    def get_features_and_gradients(self, x):
        x = x * 255.0
        x = self.encode(x)
        grad = tf.gradients(self._model.features, self._model.x_input)[0]
        return self._sess.run((self._model.features, grad),
                              {self._model.x_input: x})

    def get_features(self, x):
        x = x * 255.0
        x = self.encode(x)
        return self._sess.run(self._model.features,
                              {self._model.x_input: x})

    def get_features_and_logits(self, x):
        x = x * 255.0
        x = self.encode(x)
        return self._sess.run((self._model.features, self._model.pre_softmax),
                              {self._model.x_input: x})

    # expose internals for white box attacks

    @property
    def model(self):
        return self._model

    # x should be in [0, 255]
    def encode(self, x):
        return self._sess.run(self._encode, {self._x: x})
