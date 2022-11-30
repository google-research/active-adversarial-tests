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


class Grayscaler(Layer):
    """Converts input to grayscale
    Only active at training time since it is a regularization layer.
    # Arguments
        attenuation: how much to attenuate the input
    # Input shape
        Arbitrary.
    # Output shape
        Same as the input shape.
    """

    def __init__(self,  **kwargs):
        super(Grayscaler, self).__init__(**kwargs)
        self.supports_masking = True



    def call(self, inputs, training=None):
        def augmented():            
            return tf.image.rgb_to_grayscale(inputs)
                        
        return K.in_train_phase(augmented, augmented, training=training)
    
    
    

    def get_config(self):
        config = {}
        base_config = super(Grayscaler, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
