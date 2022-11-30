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
  def __init__(self, model, epsilon, num_steps, step_size, random_start, loss_func, dataset='cifar10',
  clip_max=255.0):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    self.model = model
    self.epsilon = epsilon
    self.num_steps = num_steps
    self.step_size = step_size
    self.rand = random_start
    self.clip_max = clip_max

    self.loss = model.loss

    self.logits = model.pre_softmax

    self.grad = tf.gradients(self.loss, model.x_input)[0]

  def perturb(self, x_nat, y, sess):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    if self.rand:
      x = x_nat + np.random.uniform(-self.epsilon, self.epsilon, x_nat.shape)
      x = np.clip(x, 0, self.clip_max) # ensure valid pixel range
    else:
      x = np.copy(x_nat)

    for i in range(self.num_steps):
      loss, logits, grad = sess.run((self.loss, self.logits, self.grad), feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      # added by AUTHOR
      if np.all(logits.argmax(-1) != y):
        break

      print(i, loss, logits)
      # x = np.add(x, self.step_size * np.sign(grad), out=x, casting='unsafe')

      # changed by AUTHOR
      grad = np.sign(grad)
      #grad = grad / (grad.reshape(len(grad), -1)**2).sum(-1)
      x = x + self.step_size * grad

      x = np.clip(x, x_nat - self.epsilon, x_nat + self.epsilon)
      x = np.clip(x, 0, self.clip_max) # ensure valid pixel range

    return x

  def perturb_l2(self, x_nat, y, sess):
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
      grad = sess.run(self.grad, feed_dict={self.model.x_input: x,
                                            self.model.y_input: y})

      normalized_grad = grad / np.linalg.norm(grad)
      pert = np.add(pert, self.step_size * normalized_grad, out=pert, casting='unsafe')
      
      # project pert to norm ball
      pert_norm = np.linalg.norm(pert)
      rescale_factor = pert_norm / self.epsilon
      pert = pert / max(1, rescale_factor)

    x = x_nat + pert
    x = np.clip(x, 0, 255)
    
    return x


if __name__ == '__main__':
  import json
  import sys
  import math

  config = vars(config_attack.get_args())

  tf.set_random_seed(config['tf_seed'])
  np.random.seed(config['np_seed'])

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  print("config['model_dir']: ", config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()
  
  print("JARN MODEL")
  from model_jarn import Model
  if "_zeromeaninput" in config['model_dir']:
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)
  else:
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'])

  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'],
                         dataset=config['dataset'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  
  print("load cifar10 dataset")
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    # Restore the checkpoint
    saver.restore(sess, model_file)
    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = [] # adv accumulator

    print('Iterating over {} batches'.format(num_batches))

    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)
      print('batch size: {}'.format(bend - bstart))

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      x_batch_adv = attack.perturb(x_batch, y_batch, sess)

      x_adv.append(x_batch_adv)

    print('Storing examples')
    path = config['store_adv_path']
    if path == None:
      model_name = config['model_dir'].split('/')[1]
      if config['attack_name'] == None:
        path = "attacks/{}_attack.npy".format(model_name)
      else:
        path = "attacks/{}_{}_attack.npy".format(model_name, config['attack_name'])

    if not os.path.exists("attacks/"):
      os.makedirs("attacks/")

    x_adv = np.concatenate(x_adv, axis=0)
    np.save(path, x_adv)
    print('Examples stored in {}'.format(path))

  if config['save_eval_log']:
    if not os.path.exists("attack_log/"):
      os.makedirs("attack_log/")
    date_str = datetime.now().strftime("%d_%b")
    log_dir = "attack_log/" + date_str
    if not os.path.exists(log_dir):
      os.makedirs(log_dir)
    log_filename = path.split("/")[-1].replace('.npy', '.txt') 
    log_file_path = os.path.join(log_dir, log_filename)
    with open(log_file_path, "w") as f:
      f.write('Saved model name: {} \n'.format(model_file))
    print('Model name saved at ', log_file_path)
