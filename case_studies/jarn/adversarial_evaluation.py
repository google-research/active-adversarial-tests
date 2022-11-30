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
Modified according to main method in pgd_attack.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow as tf
import numpy as np
import sys
import math


import cifar10_input

import config_attack
from pgd_attack import LinfPGDAttack

def main():

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
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True,
                  # added by AUTHOR
                  mode='eval')
  else:
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'],
                  # added by AUTHOR
                  mode='eval')

  saver = tf.train.Saver()

  data_path = config['data_path']

  print("load cifar10 dataset")
  cifar = cifar10_input.CIFAR10Data(data_path)

  with tf.Session() as sess:
    print("Using attack:", config['attack'])
    if config['attack'] == 'pgd' or config['attack'] == 'pgd-ld':
      attack = LinfPGDAttack(model,
                             config['epsilon'] / 255.0,
                             config['num_steps'],
                             config['step_size'],
                             config['random_start'],
                             config['loss_func'],
                             dataset=config['dataset'],
                             clip_max=1.0)
      attack_fn = lambda x, y: attack.perturb(x, y, sess)
    elif config['attack'] == 'apgd':
      from autoattack import autopgd_base
      from autoattack_adapter import ModelAdapter
      autoattack_model = ModelAdapter(
          model.pre_softmax, model.x_input,
          model.y_input, sess, num_classes=10, device="cpu")
      attack = autopgd_base.APGDAttack(
          autoattack_model, n_restarts=5, n_iter=100, verbose=True,
          eps=config["epsilon"] / 255.0, norm="Linf", eot_iter=1, rho=.99,
          is_tf_model=True, device="cpu", loss='dlr')
      attack_fn = lambda x, y: attack.perturb(
          torch.tensor(x.transpose((0, 3, 1, 2)), device="cpu"),
          torch.tensor(y, device="cpu")
      ).detach().cpu().numpy().transpose((0, 2, 3, 1))
    else:
      raise ValueError("invalid attack")


    # Restore the checkpoint
    saver.restore(sess, model_file)
    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    preds =[]
    adv_preds = []
    ys = []
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :] / 255.0
      y_batch = cifar.eval_data.ys[bstart:bend]

      x_batch_adv = attack_fn(x_batch, y_batch)

      logits = sess.run(model.pre_softmax, {model.x_input: x_batch})
      adv_logits = sess.run(model.pre_softmax, {model.x_input: x_batch_adv})

      preds.append(logits.argmax(-1))
      adv_preds.append(adv_logits.argmax(-1))
      ys.append(y_batch)

    preds = np.concatenate(preds)
    adv_preds = np.concatenate(adv_preds)
    ys = np.concatenate(ys)

    acc = np.mean(preds == ys)
    adv_acc = np.mean(adv_preds == ys)

    print("Accuracy:", acc)
    print("Robust Accuracy:", adv_acc)

if __name__ == "__main__":
  main()
