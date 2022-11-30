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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os

import numpy as np


def extract_images(file_path):
    '''Extract the images into a 4D uint8 numpy array [index, y, x, depth].'''
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    f = open(file_path, 'rb')
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, f.name))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        return data


def extract_labels(file_path):
    '''Extract the labels into a 1D uint8 numpy array [index].'''
    def _read32(bytestream):
        dt = np.dtype(np.uint32).newbyteorder('>')
        return np.frombuffer(bytestream.read(4), dtype=dt)[0]

    f = open(file_path, 'rb')
    with gzip.GzipFile(fileobj=f) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, f.name))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = np.frombuffer(buf, dtype=np.uint8)
        return labels


def load_mnist_data(data_path, is_uint8=False):
    TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
    TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
    TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
    TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

    train_images = extract_images(os.path.join(data_path, TRAIN_IMAGES))
    if not is_uint8:
        train_images = 2 * (train_images / 255.0 - 0.5)
    train_labels = extract_labels(os.path.join(data_path, TRAIN_LABELS))
    test_images = extract_images(os.path.join(data_path, TEST_IMAGES))
    if not is_uint8:
        test_images = 2 * (test_images / 255.0 - 0.5)
    test_labels = extract_labels(os.path.join(data_path, TEST_LABELS))

    train_data = order_data(train_images, train_labels, 10)
    test_data = order_data(test_images, test_labels, 10)

    return dict(train_images=train_data['images'],
                train_labels=train_data['labels'],
                train_count=train_data['count'],
                test_images=test_data['images'],
                test_labels=test_data['labels'],
                test_count=test_data['count'])


# python2
#def unpickle(file):
#  import cPickle
#  fo = open(file, 'rb')
#  dict = cPickle.load(fo)
#  fo.close()
#  return dict


# python3
def unpickle(file):
    import pickle
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='bytes')
    fo.close()
    return dict


def load_cifar100_data(data_path, is_fine=True, is_uint8=False):
    # train
    train_set = unpickle(os.path.join(data_path, 'train'))
    train_images = train_set[b'data']
    train_images = np.dstack([
        train_images[:, :1024], train_images[:, 1024:2048],
        train_images[:, 2048:]
    ])
    train_images = train_images.reshape([train_images.shape[0], 32, 32, 3])
    if not is_uint8:
        train_images = train_images / 255.0
        train_images = 2.0 * (train_images - 0.5)
    if is_fine:
        train_labels = np.array(train_set[b'fine_labels'])
    else:
        train_labels = np.array(train_set[b'coarse_labels'])

    # test
    test_set = unpickle(os.path.join(data_path, 'test'))
    test_images = test_set[b'data']
    test_images = np.dstack([
        test_images[:, :1024], test_images[:, 1024:2048], test_images[:, 2048:]
    ])
    test_images = test_images.reshape([test_images.shape[0], 32, 32, 3])
    if not is_uint8:
        test_images = test_images / 255.0
        test_images = 2.0 * (test_images - 0.5)
    if is_fine:
        test_labels = np.array(test_set[b'fine_labels'])
    else:
        test_labels = np.array(test_set[b'coarse_labels'])

    return dict(train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels)


def load_cifar10_data(data_path, is_uint8=False):
    # train
    train_names = [
        'data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4',
        'data_batch_5'
    ]
    all_images = []
    all_labels = []
    for filename in train_names:
        train_set = unpickle(os.path.join(data_path, filename))
        all_images.append(train_set[b'data'])
        all_labels.append(train_set[b'labels'])
    train_images = np.concatenate(all_images, axis=0)
    train_images = np.dstack([
        train_images[:, :1024], train_images[:, 1024:2048],
        train_images[:, 2048:]
    ])
    train_images = train_images.reshape([train_images.shape[0], 32, 32, 3])
    if not is_uint8:
        train_images = train_images / 255.0
        train_images = 2.0 * (train_images - 0.5)
    train_labels = np.concatenate(all_labels, axis=0)

    # test
    test_set = unpickle(os.path.join(data_path, 'test_batch'))
    test_images = test_set[b'data']
    test_images = np.dstack([
        test_images[:, :1024], test_images[:, 1024:2048], test_images[:, 2048:]
    ])
    test_images = test_images.reshape([test_images.shape[0], 32, 32, 3])
    if not is_uint8:
        test_images = test_images / 255.0
        test_images = 2.0 * (test_images - 0.5)
    test_labels = np.array(test_set[b'labels'])

    return dict(train_images=train_images,
                train_labels=train_labels,
                test_images=test_images,
                test_labels=test_labels)


def preprocess_py(images, pad_size, target_size):
    '''Preprocess images in python.
    Args:
        images: 4-D numpy array.
    Returns:
        preprocessed images, 4-D numpy array.
    '''
    assert images.shape[1] == images.shape[2], 'can only handle square images!'
    image_number = images.shape[0]
    image_size = images.shape[1]
    # padding, with equal pad size on both sides.
    padded_images = np.pad(images, [(0, 0), (pad_size, pad_size),
                                    (pad_size, pad_size), (0, 0)],
                           mode='constant',
                           constant_values=0)
    # random crop
    idx = np.random.random_integers(low=0,
                                    high=2 * pad_size,
                                    size=[image_number, 2])
    cropped_images = np.zeros([image_number, target_size, target_size, 3])
    for i in np.arange(image_number):
        cropped_images[i] = padded_images[i, idx[i, 0]:idx[i, 0] +
                                          target_size, idx[i, 1]:idx[i, 1] +
                                          target_size]
    # random flip
    if np.random.rand() > 0.5:
        cropped_images = cropped_images[:, :, ::-1]
    return cropped_images


def one_hot(y, dim):
    y_dense = np.zeros([len(y), dim])
    y_dense[np.arange(len(y)), y] = 1.0
    return y_dense
