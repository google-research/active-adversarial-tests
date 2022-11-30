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

# based on https://github.com/tensorflow/models/tree/master/resnet
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import json
from collections import OrderedDict


class Model(object):
    """ResNet model."""

    def __init__(self, mode, dataset, train_batch_size=None, normalize_zero_mean=True, use_pert=False):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.neck = None
        self.y_pred = None
        self.mode = mode
        self.dataset = dataset
        self.pert = True if (mode == 'train' and use_pert) else False
        if dataset == "cifar10":
            self.num_classes = 10
        elif dataset == "cifar100":
            self.num_classes = 100
        elif dataset == "GTSRB":
            self.num_classes = 43
        else:
            self.num_classes = 200
        self.train_batch_size = train_batch_size
        self.activations = []
        self.normalize_zero_mean = normalize_zero_mean
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('classifier'):
            with tf.variable_scope('input'):
                
                if self.dataset == 'cifar10' or self.dataset == 'cifar100' or self.dataset == 'GTSRB':
                    self.x_input = tf.placeholder(
                        tf.float32,
                        shape=[None, 32, 32, 3])
                else:
                    self.x_input = tf.placeholder(
                        tf.float32,
                        shape=[None, 64, 64, 3])

                self.y_input = tf.placeholder(tf.int64, shape=None)

                if self.pert:
                    self.pert = tf.get_variable(name='instance_perturbation', initializer=tf.zeros_initializer,
                                                shape=[self.train_batch_size, 32, 32, 3], dtype=tf.float32,
                                                trainable=True)
                    self.final_input = self.x_input + self.pert
                    self.final_input = tf.clip_by_value(self.final_input, 0., 255.)
                else:
                    self.final_input = self.x_input
                
                if self.normalize_zero_mean:
                    final_input_mean = tf.reduce_mean(self.final_input, axis=[1,2,3])
                    for i in range(3):
                        final_input_mean = tf.expand_dims(final_input_mean, axis=-1)
                    final_input_mean = tf.tile(final_input_mean, [1,32,32,3])
                    zero_mean_final_input = self.final_input - final_input_mean
                    self.input_standardized = tf.math.l2_normalize(zero_mean_final_input, axis=[1,2,3])
                else:
                    self.input_standardized = tf.math.l2_normalize(self.final_input, axis=[1,2,3])

                x = self._conv('init_conv', self.input_standardized, 3, 3, 16, self._stride_arr(1))
                self.activations.append(x)

            strides = [1, 2, 2]
            activate_before_residual = [True, False, False]
            res_func = self._residual

            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 16, 32, 64] # for debugging
            filters = [16, 160, 320, 640]

            # Update hps.num_residual_units to 9

            with tf.variable_scope('unit_1_0'):
                x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                            activate_before_residual[0])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_1_%d' % i):
                    x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_2_0'):
                x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                            activate_before_residual[1])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_2_%d' % i):
                    x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_3_0'):
                x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                            activate_before_residual[2])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_3_%d' % i):
                    x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_last'):
                x = self._batch_norm('final_bn', x)
                x = self._relu(x, 0.1)
                x = self._global_avg_pool(x)
                self.neck = x

            with tf.variable_scope('logit'):
                self.pre_softmax = self._fully_connected(x, self.num_classes)
                self.activations.append(self.pre_softmax)
                self.softmax = tf.nn.softmax(self.pre_softmax)
                
                sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
                sample_indices = tf.expand_dims(sample_indices, axis=-1)
                target_indices = tf.expand_dims(self.y_input, axis=-1)
                self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
                self.target_softmax = tf.gather_nd(self.softmax, self.gather_indices, name="targetsoftmax")
                # target logit is independent of other class logits while target softmax value is
                self.target_logit = tf.gather_nd(self.pre_softmax, self.gather_indices, name="targetlogit")

            self.predictions = tf.argmax(self.pre_softmax, 1)
            self.y_pred = self.predictions
            self.correct_prediction = tf.equal(self.predictions, self.y_input)
            self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            self.feature_grad = tf.gradients(self.neck, self.x_input)[0]

            with tf.variable_scope('costs'):
                self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pre_softmax, labels=self.y_input)
                self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
                self.mean_xent = tf.reduce_mean(self.y_xent)
                self.y_xent_dbp = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pre_softmax, labels=self.y_input)
                self.xent_dbp = tf.reduce_sum(self.y_xent_dbp, name='y_xent_dbp')
                self.mean_xent_dbp = tf.reduce_mean(self.y_xent_dbp)
                self.weight_decay_loss = self._decay()
                self.temploss = tf.reduce_sum(-tf.multiply(tf.one_hot(self.y_input, self.num_classes), tf.log(tf.clip_by_value(self.softmax, 1e-10, 1.0))), axis = 1)

                # for top-2 logit diff loss
                self.label_mask = tf.one_hot(self.y_input,
                                        self.num_classes,
                                        on_value=1.0,
                                        off_value=0.0,
                                        dtype=tf.float32)
                self.correct_logit = tf.reduce_sum(self.label_mask * self.pre_softmax, axis=1)
                self.wrong_logit = tf.reduce_max((1-self.label_mask) * self.pre_softmax - 1e4*self.label_mask, axis=1)
                self.top2_logit_diff_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x, decay=.9, center=True, scale=True, activation_fn=None,
                                                updates_collections=None, is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])
    

class ModelTinyImagnet(object):
    """ResNet model."""

    def __init__(self, mode, dataset, train_batch_size=None, normalize_zero_mean=False, use_pert=False):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.neck = None
        self.y_pred = None
        self.mode = mode
        self.pert = True if (mode == 'train' and use_pert) else False
        if dataset == 'tinyimagenet':
            self.num_classes = 200
            self.input_size = 64
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.input_size = 32
        else:
            self.num_classes = 10
            self.input_size = 32
        self.train_batch_size = train_batch_size
        self.activations = []
        self.normalize_zero_mean = normalize_zero_mean
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('classifier'):
            with tf.variable_scope('input'):

                self.x_input = tf.placeholder(
                    tf.float32,
                    shape=[None, self.input_size, self.input_size, 3])

                self.y_input = tf.placeholder(tf.int64, shape=None)

                if self.pert:
                    self.pert = tf.get_variable(name='instance_perturbation', initializer=tf.zeros_initializer,
                                                shape=[self.train_batch_size, self.input_size, self.input_size, 3], dtype=tf.float32,
                                                trainable=True)
                    self.final_input = self.x_input + self.pert
                    self.final_input = tf.clip_by_value(self.final_input, 0., 255.)
                else:
                    self.final_input = self.x_input
                
                if self.normalize_zero_mean:
                    final_input_mean = tf.reduce_mean(self.final_input, axis=[1,2,3])
                    for i in range(3):
                        final_input_mean = tf.expand_dims(final_input_mean, axis=-1)
                    final_input_mean = tf.tile(final_input_mean, [1,self.input_size,self.input_size,3])
                    zero_mean_final_input = self.final_input - final_input_mean
                    self.input_standardized = tf.math.l2_normalize(zero_mean_final_input, axis=[1,2,3])
                else:
                    self.input_standardized = tf.math.l2_normalize(self.final_input, axis=[1,2,3])

                x = self._conv('init_conv', self.input_standardized, 3, 3, 16, self._stride_arr(1))
                self.activations.append(x)

            strides = [1, 2, 2]
            activate_before_residual = [True, False, False]
            res_func = self._residual

            # Uncomment the following codes to use w28-10 wide residual network.
            # It is more memory efficient than very deep residual network and has
            # comparably good performance.
            # https://arxiv.org/pdf/1605.07146v1.pdf
            # filters = [16, 16, 32, 64] # for debugging
            filters = [16, 160, 320, 640]

            # Update hps.num_residual_units to 9

            with tf.variable_scope('unit_1_0'):
                x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                            activate_before_residual[0])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_1_%d' % i):
                    x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_2_0'):
                x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                            activate_before_residual[1])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_2_%d' % i):
                    x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_3_0'):
                x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                            activate_before_residual[2])
                self.activations.append(x)
            for i in range(1, 5):
                with tf.variable_scope('unit_3_%d' % i):
                    x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
                    self.activations.append(x)

            with tf.variable_scope('unit_last'):
                x = self._batch_norm('final_bn', x)
                x = self._relu(x, 0.1)
                x = self._global_avg_pool(x)
                self.neck = x

            with tf.variable_scope('logit'):
                self.pre_softmax = self._fully_connected(x, self.num_classes)
                self.activations.append(self.pre_softmax)
                self.softmax = tf.nn.softmax(self.pre_softmax)
                # y_one_hot = tf.one_hot(self.y_input, self.num_classes)
                # self.target_softmax = self.softmax * y_one_hot
                
                sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
                sample_indices = tf.expand_dims(sample_indices, axis=-1)
                target_indices = tf.expand_dims(self.y_input, axis=-1)
                self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
                self.target_softmax = tf.gather_nd(self.softmax, self.gather_indices, name="targetsoftmax")
                # target logit is independent of other class logits while target softmax value is
                self.target_logit = tf.gather_nd(self.pre_softmax, self.gather_indices, name="targetlogit")

            self.predictions = tf.argmax(self.pre_softmax, 1)
            self.y_pred = self.predictions
            self.correct_prediction = tf.equal(self.predictions, self.y_input)
            self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            with tf.variable_scope('costs'):
                self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pre_softmax, labels=self.y_input)
                self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
                self.mean_xent = tf.reduce_mean(self.y_xent)
                self.weight_decay_loss = self._decay()
                self.temploss = tf.reduce_sum(-tf.multiply(tf.one_hot(self.y_input, self.num_classes), tf.log(tf.clip_by_value(self.softmax, 1e-10, 1.0))), axis = 1)

                # for top-2 logit diff loss
                self.label_mask = tf.one_hot(self.y_input,
                                        self.num_classes,
                                        on_value=1.0,
                                        off_value=0.0,
                                        dtype=tf.float32)
                self.correct_logit = tf.reduce_sum(self.label_mask * self.pre_softmax, axis=1)
                self.wrong_logit = tf.reduce_max((1-self.label_mask) * self.pre_softmax - 1e4*self.label_mask, axis=1)
                self.top2_logit_diff_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x, decay=.9, center=True, scale=True, activation_fn=None,
                                                updates_collections=None, is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

class ModelTinyImagenetSource(object):
    """ResNet model."""

    def __init__(self, mode, dataset, train_batch_size=None, normalize_zero_mean=False, use_pert=False):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.neck = None
        self.y_pred = None
        self.mode = mode
        self.pert = True if (mode == 'train' and use_pert) else False
        if dataset == 'tinyimagenet':
            self.num_classes = 200
            self.input_size = 64
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.input_size = 32
        else:
            self.num_classes = 10
            self.input_size = 32
        self.train_batch_size = train_batch_size
        self.activations = []
        self.normalize_zero_mean = normalize_zero_mean
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):

            self.x_input = tf.placeholder(
                tf.float32,
                shape=[None, self.input_size, self.input_size, 3])

            self.y_input = tf.placeholder(tf.int64, shape=None)

            if self.pert:
                self.pert = tf.get_variable(name='instance_perturbation', initializer=tf.zeros_initializer,
                                            shape=[self.train_batch_size, self.input_size, self.input_size, 3], dtype=tf.float32,
                                            trainable=True)
                self.final_input = self.x_input + self.pert
                self.final_input = tf.clip_by_value(self.final_input, 0., 255.)
            else:
                self.final_input = self.x_input
            
            if self.normalize_zero_mean:
                final_input_mean = tf.reduce_mean(self.final_input, axis=[1,2,3])
                for i in range(3):
                    final_input_mean = tf.expand_dims(final_input_mean, axis=-1)
                final_input_mean = tf.tile(final_input_mean, [1,self.input_size,self.input_size,3])
                zero_mean_final_input = self.final_input - final_input_mean
                self.input_standardized = tf.math.l2_normalize(zero_mean_final_input, axis=[1,2,3])
            else:
                self.input_standardized = tf.math.l2_normalize(self.final_input, axis=[1,2,3])

            x = self._conv('init_conv', self.input_standardized, 3, 3, 16, self._stride_arr(1))
            self.activations.append(x)

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        # filters = [16, 16, 32, 64] # for debugging
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                        activate_before_residual[0])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                        activate_before_residual[1])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                        activate_before_residual[2])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)
            self.neck = x

        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, self.num_classes)
            self.activations.append(self.pre_softmax)
            self.softmax = tf.nn.softmax(self.pre_softmax)
            
            sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
            sample_indices = tf.expand_dims(sample_indices, axis=-1)
            target_indices = tf.expand_dims(self.y_input, axis=-1)
            self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
            self.target_softmax = tf.gather_nd(self.softmax, self.gather_indices, name="targetsoftmax")
            # target logit is independent of other class logits while target softmax value is
            self.target_logit = tf.gather_nd(self.pre_softmax, self.gather_indices, name="targetlogit")

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.y_pred = self.predictions
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        with tf.variable_scope('costs'):
            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
            self.weight_decay_loss = self._decay()

            # for top-2 logit diff loss
            self.label_mask = tf.one_hot(self.y_input,
                                    self.num_classes,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            self.correct_logit = tf.reduce_sum(self.label_mask * self.pre_softmax, axis=1)
            self.wrong_logit = tf.reduce_max((1-self.label_mask) * self.pre_softmax - 1e4*self.label_mask, axis=1)
            self.top2_logit_diff_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x, decay=.9, center=True, scale=True, activation_fn=None,
                                                updates_collections=None, is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

class ModelTinyImagenetSourceExtendedLogits(object):
    """ResNet model."""

    def __init__(self, mode, dataset, train_batch_size=None, normalize_zero_mean=False, use_pert=False, target_task_class_num=10, input_tensor=None):
        """ResNet constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.neck = None
        self.y_pred = None
        self.mode = mode
        self.pert = True if (mode == 'train' and use_pert) else False
        if dataset == 'tinyimagenet':
            self.num_classes = 200
            self.input_size = 64
        elif dataset == 'cifar100':
            self.num_classes = 100
            self.input_size = 32
        else:
            self.num_classes = 10
            self.input_size = 32
        self.train_batch_size = train_batch_size
        self.activations = []
        self.normalize_zero_mean = normalize_zero_mean
        self.input_tensor = input_tensor
        self.target_task_class_num = target_task_class_num
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('input'):
            if self.input_tensor == None:
                self.x_input = tf.placeholder(
                    tf.float32,
                    shape=[None, self.input_size, self.input_size, 3])
            else:
                self.x_input = self.input_tensor

            self.y_input = tf.placeholder(tf.int64, shape=None)

            if self.pert:
                self.pert = tf.get_variable(name='instance_perturbation', initializer=tf.zeros_initializer,
                                            shape=[self.train_batch_size, self.input_size, self.input_size, 3], dtype=tf.float32,
                                            trainable=True)
                self.final_input = self.x_input + self.pert
                self.final_input = tf.clip_by_value(self.final_input, 0., 255.)
            else:
                self.final_input = self.x_input
            
            if self.normalize_zero_mean:
                final_input_mean = tf.reduce_mean(self.final_input, axis=[1,2,3])
                for i in range(3):
                    final_input_mean = tf.expand_dims(final_input_mean, axis=-1)
                final_input_mean = tf.tile(final_input_mean, [1,self.input_size,self.input_size,3])
                zero_mean_final_input = self.final_input - final_input_mean
                self.input_standardized = tf.math.l2_normalize(zero_mean_final_input, axis=[1,2,3])
            else:
                self.input_standardized = tf.math.l2_normalize(self.final_input, axis=[1,2,3])

            x = self._conv('init_conv', self.input_standardized, 3, 3, 16, self._stride_arr(1))
            self.activations.append(x)

        strides = [1, 2, 2]
        activate_before_residual = [True, False, False]
        res_func = self._residual

        # Uncomment the following codes to use w28-10 wide residual network.
        # It is more memory efficient than very deep residual network and has
        # comparably good performance.
        # https://arxiv.org/pdf/1605.07146v1.pdf
        # filters = [16, 16, 32, 64] # for debugging
        filters = [16, 160, 320, 640]

        # Update hps.num_residual_units to 9

        with tf.variable_scope('unit_1_0'):
            x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                        activate_before_residual[0])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_1_%d' % i):
                x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_2_0'):
            x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                        activate_before_residual[1])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_2_%d' % i):
                x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_3_0'):
            x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                        activate_before_residual[2])
            self.activations.append(x)
        for i in range(1, 5):
            with tf.variable_scope('unit_3_%d' % i):
                x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
                self.activations.append(x)

        with tf.variable_scope('unit_last'):
            x = self._batch_norm('final_bn', x)
            x = self._relu(x, 0.1)
            x = self._global_avg_pool(x)
            self.neck = x

        with tf.variable_scope('logit'):
            self.pre_softmax = self._fully_connected(x, self.num_classes)
            self.activations.append(self.pre_softmax)
            self.softmax = tf.nn.softmax(self.pre_softmax)
            
            sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
            sample_indices = tf.expand_dims(sample_indices, axis=-1)
            target_indices = tf.expand_dims(self.y_input, axis=-1)
            self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
            self.target_softmax = tf.gather_nd(self.softmax, self.gather_indices, name="targetsoftmax")
            # target logit is independent of other class logits while target softmax value is
            self.target_logit = tf.gather_nd(self.pre_softmax, self.gather_indices, name="targetlogit")

        self.predictions = tf.argmax(self.pre_softmax, 1)
        self.y_pred = self.predictions
        self.correct_prediction = tf.equal(self.predictions, self.y_input)
        self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        with tf.variable_scope('costs'):
            self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.pre_softmax, labels=self.y_input)
            self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
            self.mean_xent = tf.reduce_mean(self.y_xent)
            self.weight_decay_loss = self._decay()

            # for top-2 logit diff loss
            self.label_mask = tf.one_hot(self.y_input,
                                    self.num_classes,
                                    on_value=1.0,
                                    off_value=0.0,
                                    dtype=tf.float32)
            self.correct_logit = tf.reduce_sum(self.label_mask * self.pre_softmax, axis=1)
            self.wrong_logit = tf.reduce_max((1-self.label_mask) * self.pre_softmax - 1e4*self.label_mask, axis=1)
            self.top2_logit_diff_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)

        with tf.variable_scope('target_task_logit'):
            self.target_task_pre_softmax = self._fully_connected(x, self.target_task_class_num)   
            
            self.target_task_softmax = tf.nn.softmax(self.target_task_pre_softmax)             
            sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
            sample_indices = tf.expand_dims(sample_indices, axis=-1)
            target_indices = tf.expand_dims(self.y_input, axis=-1)
            self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
            self.target_softmax = tf.gather_nd(self.target_task_softmax, self.gather_indices, name="targetsoftmax")

        self.target_task_predictions = tf.argmax(self.target_task_pre_softmax, 1)
        self.target_task_correct_prediction = tf.equal(self.target_task_predictions, self.y_input)
        self.target_task_num_correct = tf.reduce_sum(
            tf.cast(self.target_task_correct_prediction, tf.int64))
        self.target_task_accuracy = tf.reduce_mean(
            tf.cast(self.target_task_correct_prediction, tf.float32))

        with tf.variable_scope('target_task_costs'):
            self.target_task_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.target_task_pre_softmax, labels=self.y_input)
            self.target_task_xent = tf.reduce_sum(self.target_task_y_xent, name='target_task_y_xent')
            self.target_task_mean_xent = tf.reduce_mean(self.target_task_y_xent)
            
    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x, decay=.9, center=True, scale=True, activation_fn=None,
                                                updates_collections=None, is_training=(self.mode == 'train'))

    def _residual(self, x, in_filter, out_filter, stride, activate_before_residual=False):
        """Residual unit with 2 sub layers."""
        if activate_before_residual:
            with tf.variable_scope('shared_activation'):
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)
                orig_x = x
        else:
            with tf.variable_scope('residual_only_activation'):
                orig_x = x
                x = self._batch_norm('init_bn', x)
                x = self._relu(x, 0.1)

        with tf.variable_scope('sub1'):
            x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

        with tf.variable_scope('sub2'):
            x = self._batch_norm('bn2', x)
            x = self._relu(x, 0.1)
            x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

        with tf.variable_scope('sub_add'):
            if in_filter != out_filter:
                orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
                orig_x = tf.pad(
                    orig_x, [[0, 0], [0, 0], [0, 0],
                             [(out_filter - in_filter) // 2, (out_filter - in_filter) // 2]])
            x += orig_x

        tf.logging.debug('image after unit %s', x.get_shape())
        return x

    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            return tf.nn.conv2d(x, kernel, strides, padding='SAME')

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim):
        """FullyConnected layer for final output."""
        num_non_batch_dimensions = len(x.shape)
        prod_non_batch_dimensions = 1
        for ii in range(num_non_batch_dimensions - 1):
            prod_non_batch_dimensions *= int(x.shape[ii + 1])
        x = tf.reshape(x, [tf.shape(x)[0], -1])
        w = tf.get_variable(
            'DW', [prod_non_batch_dimensions, out_dim],
            initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
        b = tf.get_variable('biases', [out_dim],
                            initializer=tf.constant_initializer())
        return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])

class ModelExtendedLogitsC2I(object):
  """ResNet model."""
  def __init__(self, mode, target_task_class_num=200, train_batch_size=None, input_tensor=None, source_task="cifar10"):
    """ResNet constructor.

    Args:
      mode: One of 'train' and 'eval'.
    """
    self.mode = mode
    self.activations = []
    self.target_task_class_num = target_task_class_num
    self.train_batch_size = train_batch_size
    self.input_tensor = input_tensor
    self.source_task = source_task
    self._build_model()

  def add_internal_summaries(self):
    pass

  def _stride_arr(self, stride):
    """Map a stride scalar to the stride array for tf.nn.conv2d."""
    return [1, stride, stride, 1]

  def _build_model(self):
    assert self.mode == 'train' or self.mode == 'eval'
    """Build the core model within the graph."""
    with tf.variable_scope('input'):
      
      if self.input_tensor == None:
        self.x_input = tf.placeholder(
          tf.float32,
          shape=[None, 32, 32, 3])
      else:
        self.x_input = self.input_tensor

      self.y_input = tf.placeholder(tf.int64, shape=None)


      input_standardized = tf.map_fn(lambda img: tf.image.per_image_standardization(img),
                               self.x_input)
      x = self._conv('init_conv', input_standardized, 3, 3, 16, self._stride_arr(1))
      self.activations.append(x)

    strides = [1, 2, 2]
    activate_before_residual = [True, False, False]
    res_func = self._residual
    # Uncomment the following codes to use w28-10 wide residual network.
    # It is more memory efficient than very deep residual network and has
    # comparably good performance.
    # https://arxiv.org/pdf/1605.07146v1.pdf
    filters = [16, 160, 320, 640]


    # Update hps.num_residual_units to 9

    with tf.variable_scope('unit_1_0'):
      x = res_func(x, filters[0], filters[1], self._stride_arr(strides[0]),
                   activate_before_residual[0])
      self.activations.append(x)
    for i in range(1, 5):
      with tf.variable_scope('unit_1_%d' % i):
        x = res_func(x, filters[1], filters[1], self._stride_arr(1), False)
        self.activations.append(x)

    with tf.variable_scope('unit_2_0'):
      x = res_func(x, filters[1], filters[2], self._stride_arr(strides[1]),
                   activate_before_residual[1])
      self.activations.append(x)
    for i in range(1, 5):
      with tf.variable_scope('unit_2_%d' % i):
        x = res_func(x, filters[2], filters[2], self._stride_arr(1), False)
        self.activations.append(x)

    with tf.variable_scope('unit_3_0'):
      x = res_func(x, filters[2], filters[3], self._stride_arr(strides[2]),
                   activate_before_residual[2])
      self.activations.append(x)
    for i in range(1, 5):
      with tf.variable_scope('unit_3_%d' % i):
        x = res_func(x, filters[3], filters[3], self._stride_arr(1), False)
        self.activations.append(x)

    with tf.variable_scope('unit_last'):
      x = self._batch_norm('final_bn', x)
      x = self._relu(x, 0.1)
      x = self._global_avg_pool(x)

    with tf.variable_scope('logit'):
      if self.source_task == "cifar10":
        self.pre_softmax = self._fully_connected(x, 10)
      elif self.source_task == "cifar100":
        self.pre_softmax = self._fully_connected(x, 100)
      self.activations.append(self.pre_softmax)

    self.predictions = tf.argmax(self.pre_softmax, 1)
    self.correct_prediction = tf.equal(self.predictions, self.y_input)
    self.num_correct = tf.reduce_sum(
        tf.cast(self.correct_prediction, tf.int64))
    self.accuracy = tf.reduce_mean(
        tf.cast(self.correct_prediction, tf.float32))

    with tf.variable_scope('costs'):
      self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.pre_softmax, labels=self.y_input)
      self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
      self.mean_xent = tf.reduce_mean(self.y_xent)
      self.weight_decay_loss = self._decay()

    with tf.variable_scope('target_task_logit'):
      self.target_task_pre_softmax = self._fully_connected(x, self.target_task_class_num)   
      
      self.target_task_softmax = tf.nn.softmax(self.target_task_pre_softmax)             
      sample_indices = tf.range(self.train_batch_size, dtype=tf.int64)
      sample_indices = tf.expand_dims(sample_indices, axis=-1)
      target_indices = tf.expand_dims(self.y_input, axis=-1)
      self.gather_indices = tf.concat([sample_indices, target_indices], axis=-1)
      self.target_softmax = tf.gather_nd(self.target_task_softmax, self.gather_indices, name="targetsoftmax")

      # self.target_task_pre_softmax = self._named_fully_connected('target_task_logit', x, self.target_task_class_num)

    self.target_task_predictions = tf.argmax(self.target_task_pre_softmax, 1)
    self.target_task_correct_prediction = tf.equal(self.target_task_predictions, self.y_input)
    self.target_task_num_correct = tf.reduce_sum(
        tf.cast(self.target_task_correct_prediction, tf.int64))
    self.target_task_accuracy = tf.reduce_mean(
        tf.cast(self.target_task_correct_prediction, tf.float32))

    with tf.variable_scope('target_task_costs'):
      self.target_task_y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
          logits=self.target_task_pre_softmax, labels=self.y_input)
      self.target_task_xent = tf.reduce_sum(self.target_task_y_xent, name='target_task_y_xent')
      self.target_task_mean_xent = tf.reduce_mean(self.target_task_y_xent)
      # self.weight_decay_loss = self._decay()

  def _batch_norm(self, name, x):
    """Batch normalization."""
    with tf.name_scope(name):
      return tf.contrib.layers.batch_norm(
          inputs=x,
          decay=.9,
          center=True,
          scale=True,
          activation_fn=None,
          updates_collections=None,
          is_training=(self.mode == 'train'))

  def _residual(self, x, in_filter, out_filter, stride,
                activate_before_residual=False):
    """Residual unit with 2 sub layers."""
    if activate_before_residual:
      with tf.variable_scope('shared_activation'):
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)
        orig_x = x
    else:
      with tf.variable_scope('residual_only_activation'):
        orig_x = x
        x = self._batch_norm('init_bn', x)
        x = self._relu(x, 0.1)

    with tf.variable_scope('sub1'):
      x = self._conv('conv1', x, 3, in_filter, out_filter, stride)

    with tf.variable_scope('sub2'):
      x = self._batch_norm('bn2', x)
      x = self._relu(x, 0.1)
      x = self._conv('conv2', x, 3, out_filter, out_filter, [1, 1, 1, 1])

    with tf.variable_scope('sub_add'):
      if in_filter != out_filter:
        orig_x = tf.nn.avg_pool(orig_x, stride, stride, 'VALID')
        orig_x = tf.pad(
            orig_x, [[0, 0], [0, 0], [0, 0],
                     [(out_filter-in_filter)//2, (out_filter-in_filter)//2]])
      x += orig_x

    tf.logging.debug('image after unit %s', x.get_shape())
    return x

  def _decay(self):
    """L2 weight decay loss."""
    costs = []
    for var in tf.trainable_variables():
      if var.op.name.find('DW') > 0:
        costs.append(tf.nn.l2_loss(var))
    return tf.add_n(costs)

  def _conv(self, name, x, filter_size, in_filters, out_filters, strides):
    """Convolution."""
    with tf.variable_scope(name):
      n = filter_size * filter_size * out_filters
      kernel = tf.get_variable(
          'DW', [filter_size, filter_size, in_filters, out_filters],
          tf.float32, initializer=tf.random_normal_initializer(
              stddev=np.sqrt(2.0/n)))
      return tf.nn.conv2d(x, kernel, strides, padding='SAME')

  def _relu(self, x, leakiness=0.0):
    """Relu, with optional leaky support."""
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

  def _fully_connected(self, x, out_dim):
    """FullyConnected layer for final output."""
    num_non_batch_dimensions = len(x.shape)
    prod_non_batch_dimensions = 1
    for ii in range(num_non_batch_dimensions - 1):
      prod_non_batch_dimensions *= int(x.shape[ii + 1])
    x = tf.reshape(x, [tf.shape(x)[0], -1])
    w = tf.get_variable(
        'DW', [prod_non_batch_dimensions, out_dim],
        initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
    b = tf.get_variable('biases', [out_dim],
                        initializer=tf.constant_initializer())
    return tf.nn.xw_plus_b(x, w, b)

  def _named_fully_connected(self, name, x, out_dim):
    """FullyConnected layer for final output."""
    with tf.variable_scope(name):
      num_non_batch_dimensions = len(x.shape)
      prod_non_batch_dimensions = 1
      for ii in range(num_non_batch_dimensions - 1):
        prod_non_batch_dimensions *= int(x.shape[ii + 1])
      x = tf.reshape(x, [tf.shape(x)[0], -1])
      w = tf.get_variable(
          'DW', [prod_non_batch_dimensions, out_dim],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      b = tf.get_variable('biases', [out_dim],
                          initializer=tf.constant_initializer())
      return tf.nn.xw_plus_b(x, w, b)

  def _global_avg_pool(self, x):
    assert x.get_shape().ndims == 4
    return tf.reduce_mean(x, [1, 2])



class IgamConvDiscriminatorModel(object):
    """Simple conv discriminator model."""
    # based on https://github.com/tensorflow/models/blob/d361076952b73706c5c7ddf9c940bf42c27a3213/research/slim/nets/dcgan.py#L41

    def __init__(self, mode, dataset, train_batch_size=None, num_conv_layers=5, base_num_channels=16, x_modelgrad_input_tensor=None, y_modelgrad_input_tensor=None, x_source_modelgrad_input_tensor=None, 
        y_source_modelgrad_input_tensor=None, normalize_zero_mean=False, only_fully_connected=False, num_fc_layers=3, image_size=32, cropped_input_size=None, crop_pad_x_tensor=None, crop_pad_y_tensor=None, avg_pool_hw=False):
        """conv disc constructor.

        Args:
          mode: One of 'train' and 'eval'.
        """
        self.neck = None
        self.y_pred = None
        self.mode = mode
        self.pert = False
        self.num_classes = 2 # grad from model or rand init grad
        self.train_batch_size = train_batch_size
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.base_num_channels = base_num_channels
        self.x_modelgrad_input_tensor = x_modelgrad_input_tensor
        self.y_modelgrad_input_tensor = y_modelgrad_input_tensor
        self.x_source_modelgrad_input_tensor = x_source_modelgrad_input_tensor
        self.y_source_modelgrad_input_tensor = y_source_modelgrad_input_tensor
        self.normalize_zero_mean = normalize_zero_mean
        self.only_fully_connected = only_fully_connected
        self.image_size = image_size
        self.cropped_input_size = cropped_input_size
        self.crop_pad_x_tensor = crop_pad_x_tensor
        self.crop_pad_y_tensor = crop_pad_y_tensor
        self.avg_pool_hw = avg_pool_hw
        self._build_model()

    def add_internal_summaries(self):
        pass

    def _stride_arr(self, stride):
        """Map a stride scalar to the stride array for tf.nn.conv2d."""
        return [1, stride, stride, 1]

    def _build_model(self):
        assert self.mode == 'train' or self.mode == 'eval'
        """Build the core model within the graph."""
        with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('input'):

                if self.x_modelgrad_input_tensor == None:
                    # for assign to work
                    self.x_modelgrad_input = tf.get_variable(name='x_modelgrad_input', initializer=tf.zeros_initializer,
                                                shape=[self.train_batch_size, self.image_size, self.image_size, 3], dtype=tf.float32)
                    
                    self.x_source_modelgrad_input = tf.placeholder(
                        tf.float32,
                        shape=[None, self.image_size, self.image_size, 3])
                else:
                    self.x_modelgrad_input = self.x_modelgrad_input_tensor
                    self.x_source_modelgrad_input = self.x_source_modelgrad_input_tensor
                
                if self.cropped_input_size != None:
                    if self.crop_pad_x_tensor == None:
                        crop_pad = (self.image_size - self.cropped_input_size) // 2
                        cropped_x_modelgrad_input = tf.slice(self.x_modelgrad_input, [0, crop_pad, crop_pad, 0], [-1, self.cropped_input_size, self.cropped_input_size, -1])
                        cropped_x_source_modelgrad_input = tf.slice(self.x_source_modelgrad_input, [0, crop_pad, crop_pad, 0], [-1, self.cropped_input_size, self.cropped_input_size, -1])

                        self.x_input = tf.concat([cropped_x_modelgrad_input, cropped_x_source_modelgrad_input], axis=0)
                    else:
                        cropped_x_modelgrad_input = tf.slice(self.x_modelgrad_input, [0, self.crop_pad_x_tensor, self.crop_pad_y_tensor, 0], [-1, self.cropped_input_size, self.cropped_input_size, -1])
                        cropped_x_source_modelgrad_input = tf.slice(self.x_source_modelgrad_input, [0, self.crop_pad_x_tensor, self.crop_pad_y_tensor, 0], [-1, self.cropped_input_size, self.cropped_input_size, -1])

                        self.x_input = tf.concat([cropped_x_modelgrad_input, cropped_x_source_modelgrad_input], axis=0)
                else:
                    self.x_input = tf.concat([self.x_modelgrad_input, self.x_source_modelgrad_input], axis=0)
                    self.cropped_input_size = self.image_size
                
                if self.y_modelgrad_input_tensor == None:
                    # for assign to work
                    self.y_modelgrad_input = tf.get_variable(name='y_modelgrad_input', initializer=tf.zeros_initializer,
                                                shape=self.train_batch_size, dtype=tf.int64)

                    self.y_source_modelgrad_input = tf.placeholder(tf.int64, shape=None)
                else:
                    self.y_modelgrad_input = self.y_modelgrad_input_tensor
                    self.y_source_modelgrad_input = self.y_source_modelgrad_input_tensor

                self.y_input = tf.concat([self.y_modelgrad_input, self.y_source_modelgrad_input], axis=0)
                

                if self.pert:
                    self.pert = tf.get_variable(name='instance_perturbation', initializer=tf.zeros_initializer,
                                                shape=[self.train_batch_size, self.cropped_input_size, self.cropped_input_size, 3], dtype=tf.float32,
                                                trainable=True)
                    self.final_input = self.x_input + self.pert
                    self.final_input = tf.clip_by_value(self.final_input, 0., 255.)
                else:
                    self.final_input = self.x_input

                if self.normalize_zero_mean:
                    final_input_mean = tf.reduce_mean(self.final_input, axis=[1,2,3])
                    for i in range(3):
                        final_input_mean = tf.expand_dims(final_input_mean, axis=-1)
                    final_input_mean = tf.tile(final_input_mean, [1, self.cropped_input_size, self.cropped_input_size,3])
                    zero_mean_final_input = self.final_input - final_input_mean
                    self.input_standardized = tf.math.l2_normalize(zero_mean_final_input, axis=[1,2,3])
                else:
                    self.input_standardized = tf.math.l2_normalize(self.final_input, axis=[1,2,3])

            x = self.input_standardized
            base_num_channels = self.base_num_channels
            if self.only_fully_connected == False:
                for i in range(self.num_conv_layers):
                    output_num_channels = base_num_channels * 2**i
                    if i == 0:
                        x = self._conv('conv{}'.format(i), x, 4, 3, output_num_channels, self._stride_arr(2), bias=True)            
                        x = self._batch_norm('bn{}'.format(i), x)
                        x = self._relu(x, 0.1)
                    else:
                        x = self._conv('conv{}'.format(i), x, 4, output_num_channels // 2, output_num_channels, self._stride_arr(2), bias=True)
                        x = self._batch_norm('bn{}'.format(i), x)
                        x = self._relu(x, 0.1)
            else:
                for i in range(self.num_fc_layers):
                    if i == self.num_fc_layers -1:
                        x = self._fully_connected(x, base_num_channels//2, name='fc{}'.format(i))
                    else:
                        x = self._fully_connected(x, base_num_channels, name='fc{}'.format(i))
                    x = self._batch_norm('bn{}'.format(i), x)
                    x = self._relu(x, 0.1)

            with tf.variable_scope('logit'):
                if self.avg_pool_hw:
                    x = self._global_avg_pool(x)
                self.pre_softmax = self._fully_connected(x, self.num_classes)

            self.predictions = tf.argmax(self.pre_softmax, 1)
            self.y_pred = self.predictions
            self.correct_prediction = tf.equal(self.predictions, self.y_input)
            self.num_correct = tf.reduce_sum(tf.cast(self.correct_prediction, tf.int64))
            self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

            with tf.variable_scope('costs'):
                self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self.pre_softmax, labels=self.y_input)
                self.xent = tf.reduce_sum(self.y_xent, name='y_xent')
                self.mean_xent = tf.reduce_mean(self.y_xent)
                self.weight_decay_loss = self._decay()

    def _batch_norm(self, name, x):
        """Batch normalization."""
        with tf.name_scope(name):
            return tf.contrib.layers.batch_norm(inputs=x, decay=.9, center=True, scale=True, activation_fn=None,
                                                updates_collections=None, is_training=(self.mode == 'train'))
    def _decay(self):
        """L2 weight decay loss."""
        costs = []
        for var in tf.trainable_variables():
            if var.op.name.find('DW') > 0:
                costs.append(tf.nn.l2_loss(var))
        return tf.add_n(costs)

    def _conv(self, name, x, filter_size, in_filters, out_filters, strides, bias=False, padding='SAME'):
        """Convolution."""
        with tf.variable_scope(name):
            n = filter_size * filter_size * out_filters
            kernel = tf.get_variable(
                'DW', [filter_size, filter_size, in_filters, out_filters],
                tf.float32, initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(2.0 / n)))
            if bias == True:
                b = tf.get_variable('biases', [out_filters],
                                    initializer=tf.constant_initializer())
                conv_out = tf.nn.conv2d(x, kernel, strides, padding=padding)
                conv_out_b = tf.nn.bias_add(conv_out, b)
                return conv_out_b
            else:
                return tf.nn.conv2d(x, kernel, strides, padding=padding)

    def _relu(self, x, leakiness=0.0):
        """Relu, with optional leaky support."""
        return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

    def _fully_connected(self, x, out_dim, name=None):
        """FullyConnected layer for final output."""
        if name == None:
            num_non_batch_dimensions = len(x.shape)
            prod_non_batch_dimensions = 1
            for ii in range(num_non_batch_dimensions - 1):
                prod_non_batch_dimensions *= int(x.shape[ii + 1])
            x = tf.reshape(x, [tf.shape(x)[0], -1])
            w = tf.get_variable(
                'DW', [prod_non_batch_dimensions, out_dim],
                initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
            b = tf.get_variable('biases', [out_dim],
                                initializer=tf.constant_initializer())
            return tf.nn.xw_plus_b(x, w, b)
        else:
            with tf.variable_scope(name):
                num_non_batch_dimensions = len(x.shape)
                prod_non_batch_dimensions = 1
                for ii in range(num_non_batch_dimensions - 1):
                    prod_non_batch_dimensions *= int(x.shape[ii + 1])
                x = tf.reshape(x, [tf.shape(x)[0], -1])
                w = tf.get_variable(
                    'DW', [prod_non_batch_dimensions, out_dim],
                    initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
                b = tf.get_variable('biases', [out_dim],
                                    initializer=tf.constant_initializer())
                return tf.nn.xw_plus_b(x, w, b)

    def _global_avg_pool(self, x):
        assert x.get_shape().ndims == 4
        return tf.reduce_mean(x, [1, 2])


