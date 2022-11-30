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

from utils import optimistic_restore
import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets
import functools

INCEPTION_CHECKPOINT_PATH = 'checkpoints/inputtransformations_inceptionv3/inception_v3.ckpt'


def _get_model(reuse):
    arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
    func = nets.inception.inception_v3
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)
    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size
    return network_fn

def _preprocess(image, height, width, scope=None):
    with tf.name_scope(scope, 'eval_image', [image, height, width]):
        if image.dtype != tf.float32:
            image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_bilinear(image, [height, width], align_corners=False)
        image = tf.subtract(image, 0.5)
        image = tf.multiply(image, 2.0)
        return image

# input is [batch, ?, ?, 3], pixels in [0, 1]
# it's rescaled to [batch, 299, 299, 3] and shifted to [-1, 1]
# output is [batch, 1000] (imagenet classes)
_inception_initialized = False
def model(sess, image):
    global _inception_initialized
    network_fn = _get_model(reuse=_inception_initialized)
    size = network_fn.default_image_size
    preprocessed = _preprocess(image, size, size)
    logits, _ = network_fn(preprocessed)
    logits = logits[:,1:] # ignore background class
    predictions = tf.argmax(logits, 1)

    if not _inception_initialized:
        optimistic_restore(sess, INCEPTION_CHECKPOINT_PATH)
        _inception_initialized = True

    return logits, predictions
