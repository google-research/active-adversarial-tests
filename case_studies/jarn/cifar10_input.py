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

"""
Utilities for importing the CIFAR10 dataset.
Each image in the dataset is a numpy array of shape (32, 32, 3), with the values
being unsigned integers (i.e., in the range 0,1,...,255).
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import pickle
import sys
import tensorflow as tf
import numpy as np
import re

version = sys.version_info


class CIFAR10Data(object):
    """
    Unpickles the CIFAR10 dataset from a specified folder containing a pickled
    version following the format of Krizhevsky which can be found
    [here](https://www.cs.toronto.edu/~kriz/cifar.html).
    Inputs to constructor
    =====================
        - path: path to the pickled dataset. The training data must be pickled
        into five files named data_batch_i for i = 1, ..., 5, containing 10,000
        examples each, the test data
        must be pickled into a single file called test_batch containing 10,000
        examples, and the 10 class names must be
        pickled into a file called batches.meta. The pickled examples should
        be stored as a tuple of two objects: an array of 10,000 32x32x3-shaped
        arrays, and an array of their 10,000 true labels.
    """

    def __init__(self, path, init_shuffle=True, train_size_ratio=1):
        num_classes = 10
        path = CIFAR10Data.rec_search(path)
        train_filenames = ['data_batch_{}'.format(ii + 1) for ii in range(5)]
        eval_filename = 'test_batch'
        metadata_filename = 'batches.meta'

        train_images = np.zeros((50000, 32, 32, 3), dtype='uint8')
        train_labels = np.zeros(50000, dtype='int32')
        for ii, fname in enumerate(train_filenames):
            cur_images, cur_labels = self._load_datafile(os.path.join(path, fname))
            train_images[ii * 10000: (ii + 1) * 10000, ...] = cur_images
            train_labels[ii * 10000: (ii + 1) * 10000, ...] = cur_labels
        eval_images, eval_labels = self._load_datafile(
            os.path.join(path, eval_filename))

        with open(os.path.join(path, metadata_filename), 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            self.label_names = data_dict[b'label_names']
        for ii in range(len(self.label_names)):
            self.label_names[ii] = self.label_names[ii].decode('utf-8')
        
        if train_size_ratio < 1:
            new_train_images = []
            new_train_labels = []
            for class_ind in range(num_classes):
                current_class_train_images = train_images[train_labels == class_ind]
                num_train_per_class = int(current_class_train_images.shape[0] * train_size_ratio)
                new_train_images.append(current_class_train_images[:num_train_per_class])
                new_train_labels.append(np.full(num_train_per_class, class_ind, dtype='int32'))
            train_images = np.concatenate(new_train_images, axis=0)
            train_labels = np.concatenate(new_train_labels)
        
        self.train_data = DataSubset(train_images, train_labels, init_shuffle=init_shuffle)
        self.eval_data = DataSubset(eval_images, eval_labels, init_shuffle=init_shuffle)

    @staticmethod
    def rec_search(original_path):
        rx = re.compile(r'data_batch_[0-9]+')
        r = []
        for path, _, file_names in os.walk(original_path):
            r.extend([os.path.join(path, x) for x in file_names if rx.search(x)])
        if len(r) is 0:  # TODO: Is this the best way?
            return original_path
        return os.path.dirname(r[0])

    @staticmethod
    def _load_datafile(filename):
        with open(filename, 'rb') as fo:
            if version.major == 3:
                data_dict = pickle.load(fo, encoding='bytes')
            else:
                data_dict = pickle.load(fo)

            assert data_dict[b'data'].dtype == np.uint8
            image_data = data_dict[b'data']
            image_data = image_data.reshape((10000, 3, 32, 32)).transpose(0, 2, 3, 1)
            return image_data, np.array(data_dict[b'labels'])


class AugmentedCIFAR10Data(object):
    """
    Data augmentation wrapper over a loaded dataset.
    Inputs to constructor
    =====================
        - raw_cifar10data: the loaded CIFAR10 dataset, via the CIFAR10Data class
        - sess: current tensorflow session
        - model: current model (needed for input tensor)
    """

    def __init__(self, raw_cifar10data, sess, model):
        assert isinstance(raw_cifar10data, CIFAR10Data)
        self.image_size = 32

        # create augmentation computational graph
        self.x_input_placeholder = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        padded = tf.map_fn(lambda img: tf.image.resize_image_with_crop_or_pad(
            img, self.image_size + 4, self.image_size + 4),
                           self.x_input_placeholder)
        cropped = tf.map_fn(lambda img: tf.random_crop(img, [self.image_size,
                                                             self.image_size,
                                                             3]), padded)
        flipped = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), cropped)
        self.augmented = flipped

        self.train_data = AugmentedDataSubset(raw_cifar10data.train_data, sess,
                                              self.x_input_placeholder,
                                              self.augmented)
        self.eval_data = AugmentedDataSubset(raw_cifar10data.eval_data, sess,
                                             self.x_input_placeholder,
                                             self.augmented)
        self.label_names = raw_cifar10data.label_names


class DataSubset(object):
    def __init__(self, xs, ys, init_shuffle=True):
        self.xs = xs
        self.n = xs.shape[0]
        self.ys = ys
        self.batch_start = 0
        if init_shuffle:
            self.cur_order = np.random.permutation(self.n)
        else:
            self.cur_order = np.arange(self.n)

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        if self.n < batch_size:
            raise ValueError('Batch size can be at most the dataset size')
        if not multiple_passes:
            actual_batch_size = min(batch_size, self.n - self.batch_start)
            if actual_batch_size <= 0:
                raise ValueError('Pass through the dataset is complete.')
            batch_end = self.batch_start + actual_batch_size
            batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
            batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
            self.batch_start += actual_batch_size
            if actual_batch_size < batch_size:
                print('actual_batch_size < batch_size, padding with zeros')
                batch_xs_pad = np.zeros(shape=(batch_size - actual_batch_size, batch_xs.shape[1], batch_xs.shape[2], batch_xs.shape[3]), dtype=batch_xs.dtype)
                batch_ys_pad = np.zeros(batch_size - actual_batch_size, dtype=batch_ys.dtype)
                batch_xs = np.concatenate([batch_xs, batch_xs_pad], axis=0)
                batch_ys = np.concatenate([batch_ys, batch_ys_pad], axis=0)
            return batch_xs, batch_ys
        actual_batch_size = min(batch_size, self.n - self.batch_start)
        if actual_batch_size < batch_size:
            if reshuffle_after_pass:
                self.cur_order = np.random.permutation(self.n)
            self.batch_start = 0
        batch_end = self.batch_start + batch_size
        batch_xs = self.xs[self.cur_order[self.batch_start: batch_end], ...]
        batch_ys = self.ys[self.cur_order[self.batch_start: batch_end], ...]
        self.batch_start += actual_batch_size
        return batch_xs, batch_ys


class AugmentedDataSubset(object):
    def __init__(self, raw_datasubset, sess, x_input_placeholder,
                 augmented):
        self.sess = sess
        self.raw_datasubset = raw_datasubset
        self.x_input_placeholder = x_input_placeholder
        self.augmented = augmented

    def get_next_batch(self, batch_size, multiple_passes=False, reshuffle_after_pass=True):
        raw_batch = self.raw_datasubset.get_next_batch(batch_size, multiple_passes,
                                                       reshuffle_after_pass)
        images = raw_batch[0].astype(np.float32)
        return self.sess.run(self.augmented, feed_dict={self.x_input_placeholder:
                                                            raw_batch[0]}), raw_batch[1]