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
Implementation of attack methods. Running this file as a program will
apply the attack to the model specified by the config file and store
the examples in an .npy file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import tensorflow as tf
import numpy as np

import cifar10_input

import config_attack

class LinfPGDAttack:
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, dataset='cifar10'):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.dataset = dataset

    if loss_func == 'xent':
      loss = model.xent
    elif loss_func == "logit-diff":
      loss = model.top2_logit_diff_loss
    elif loss_func == 'target_task_xent':
      raise ValueError("Not implemented")
      loss = model.target_task_mean_xent
      correct_logit = tf.reduce_sum(label_mask * model.pre_softmax, axis=1)
      wrong_logit = tf.reduce_max((1-label_mask) * model.pre_softmax - 1e4*label_mask, axis=1)
      loss = -tf.nn.relu(correct_logit - wrong_logit + 50)
    else:
      print('Unknown loss function. Defaulting to cross-entropy')
      loss = model.xent

    self.grad = tf.gradients(loss, model.x_input)[0]
    self.loss = loss
    # self.logit = tf.placeholder(tf.float32, shape=[None, 100])
    # self.grad2 = tf.gradients(loss + tf.reduce_mean(tf.reduce_sum(tf.pow(tf.subtract(self.logit, model.pre_softmax), 2.0), keepdims=True)), model.x_input)[0]

  def perturb(self, x_nat, y, sess, feed_dict={}):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, 255) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      loss, grad = sess.run((self.loss, self.grad), feed_dict={self.model.x_input: x,
                                            self.model.y_input: y,
                                            **feed_dict})

      x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, 255) # ensure valid pixel range

    return x

  def perturb_l2(self, x_nat, y, sess, feed_dict={}):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_2 norm."""
    if self.rand:
      pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      pert_norm = np.linalg.norm(pert)
      pert = pert / max(1, pert_norm)
    else:
      pert = np.zeros(x_nat.shape)

    for i in range(self.num_steps):
      x = x_nat + pert
      # x = np.clip(x, 0, 255)
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y,
                                            **feed_dict})

      normalized_grad = grad / np.linalg.norm(grad)
      pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')
      
      # project pert to norm ball
      pert_norm = np.linalg.norm(pert)
      rescale_factor = pert_norm / self.epsilon
      pert = pert / max(1, rescale_factor)

    x = x_nat + pert
    x = np.clip(x, 0, 255)
    
    return x

  # def perturb_TRADES(self, x_nat, y, sess):
  #   """Given a set of examples (x_nat, y), returns a set of adversarial
  #      examples within epsilon of x_nat in l_2 norm of TRADES Loss."""
  #   if self.rand:
  #     pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
  #     pert_norm = np.linalg.norm(pert)
  #     pert = pert / max(1, pert_norm)
  #   else:
  #     pert = np.zeros(x_nat.shape)
    
  #   nat_logit = sess.run(model.pre_softmax, feed_dict={self.model.x_input: x_nat,
  #                                           self.model.y_input: y})
  #   for i in range(self.num_steps):
  #     x = x_nat + pert
  #     grad = sess.run(self.grad2, feed_dict={self.model.x_input: x,
  #                                           self.model.y_input: y, self.logit: nat_logit})
  #     normalized_grad = grad / np.linalg.norm(grad)
  #     pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')
  #     pert_norm = np.linalg.norm(pert)
  #     rescale_factor = pert_norm / self.epsilon
  #     pert = pert / max(1, rescale_factor)

  #   #x = x_nat + pert
  #   x = np.clip(x, 0, 255)
    
  #   return x


  def modified_perturb_l2(self, x_nat, y, feed_dict={}):
    if self.rand:
      pert = np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      pert_norm = np.linalg.norm(pert)
      pert = pert / max(1, pert_norm)
    else:
      pert = np.zeros(x_nat.shape)

    for i in range(self.num_steps):
      x = x_nat + pert
      # x = np.clip(x, 0, 255)
      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y,
                                              **feed_dict})

      normalized_grad = grad / np.linalg.norm(grad)
      pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')
      
      # project pert to norm ball
      pert_norm = np.linalg.norm(pert)
      rescale_factor = pert_norm / self.epsilon
      pert = pert / max(1, rescale_factor)

    x = x_nat + pert
    x = np.clip(x, 0, 255)
    
    return (x - x_nat)


