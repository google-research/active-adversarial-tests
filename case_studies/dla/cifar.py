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

import numpy as np


class CIFAR10:
  def __init__(self, seed = 43, tf_mode=False):
    if tf_mode:
      import tensorflow.compat.v1 as tf
      tf.disable_v2_behavior()
      tf.compat.v1.disable_eager_execution()
      (train_data, train_labels),(self.test_data, self.test_labels) = tf.keras.datasets.cifar10.load_data()
    else:
      import torchvision
      train_dataset = torchvision.datasets.CIFAR10("data", train=True)
      test_dataset = torchvision.datasets.CIFAR10("data", train=False)
      train_data, train_labels = train_dataset.data, np.array(train_dataset.targets, dtype=int)
      self.test_data, self.test_labels = test_dataset.data, np.array(test_dataset.targets, dtype=int)

    train_data = train_data/255.
    self.test_data = self.test_data/255.

    VALIDATION_SIZE = 5000

    np.random.seed(seed)
    shuffled_indices = np.arange(len(train_data))
    np.random.shuffle(shuffled_indices)
    train_data = train_data[shuffled_indices]
    train_labels = train_labels[shuffled_indices]

    shuffled_indices = np.arange(len(self.test_data))
    np.random.shuffle(shuffled_indices)
    self.test_data = self.test_data[shuffled_indices]
    self.test_labels = self.test_labels[shuffled_indices].flatten()

    self.validation_data = train_data[:VALIDATION_SIZE, :, :, :]
    self.validation_labels = train_labels[:VALIDATION_SIZE]
    self.train_data = train_data[VALIDATION_SIZE:, :, :, :]
    self.train_labels = train_labels[VALIDATION_SIZE:]