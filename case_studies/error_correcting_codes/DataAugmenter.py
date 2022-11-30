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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
"""

from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
import tensorflow as tf

import numpy as np


class DataAugmenter(Layer):
    """Shifts and scales input
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  batch_size, **kwargs):
        super(DataAugmenter, self).__init__(**kwargs)
        self.supports_masking = True
        self.batch_size = batch_size




    def call(self, inputs, training=None):
        def augmented():
            angles = (15*(2*np.random.rand(self.batch_size)-1))*np.pi/180
            shifts = 4*(2*np.random.rand(self.batch_size, 2)-1) 
            inputs_shifted = tf.contrib.image.translate(inputs, shifts)
            inputs_shifted_rotated = tf.contrib.image.rotate(inputs_shifted,angles)
            
            random_number = tf.random_uniform([self.batch_size])   
            inputs_shifted_rotated_flipped = tf.where(random_number<0.5, tf.image.flip_left_right(inputs_shifted_rotated), inputs_shifted_rotated)

            # modificaiton by zimmerrol to make sure keras shape computation works
            inputs_shifted_rotated_flipped = tf.ensure_shape(inputs_shifted_rotated_flipped, inputs.shape)

            return inputs_shifted_rotated_flipped

            
        
        return K.in_train_phase(augmented, inputs, training=training)

    def get_config(self):
        config = {}
        config['batch_size'] = self.batch_size
        base_config = super(DataAugmenter, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
