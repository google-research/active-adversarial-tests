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
from functools import partial

import torch

import utils
from argparse_utils import DecisionBoundaryBinarizationSettings
from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper

logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

from active_tests import decision_boundary_binarization as dbb

import tensorflow as tf
import numpy as np
import sys
import math

import cifar10_input

import config_attack
from pgd_attack import LinfPGDAttack


class BinarizedModel:
  def __init__(self, model, logit_diff_loss=False):
    self.model = model
    self.x_input = model.x_input
    self.y_input = model.y_input

    features = model.neck

    with tf.variable_scope("binarized_readout"):
      # build linear readout
      bro_w = tf.get_variable(
          'DW', [features.shape[-1], 2],
          initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
      bro_b = tf.get_variable('biases', [2],
                              initializer=tf.constant_initializer())
      self.bro_w_pl = tf.placeholder(tf.float32, shape=[features.shape[-1], 2])
      self.bro_b_pl = tf.placeholder(tf.float32, shape=[2])
      self.bro_w_set_weight = bro_w.assign(self.bro_w_pl)
      self.bro_b_set_weight = bro_b.assign(self.bro_b_pl)
      self.pre_softmax = tf.nn.xw_plus_b(features, bro_w, bro_b)

      if logit_diff_loss:
        yh = tf.one_hot(self.y_input, 2)
        self.loss = tf.reduce_max(self.pre_softmax - yh * 1e9) - tf.gather(
          self.pre_softmax, self.y_input, axis=-1)
      else:
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pre_softmax, labels=self.y_input)
        self.loss = tf.reduce_sum(self.y_xent, name='y_xent')


def run_attack(m, l, sess, logits, x_pl, bro_w_pl, bro_b_pl, bro_w_assign,
    bro_b_assign, attack):
  linear_layer = m[-1]
  del m

  sess.run(bro_w_assign, {bro_w_pl: linear_layer.weight.data.numpy().T})
  sess.run(bro_b_assign, {bro_b_pl: linear_layer.bias.data.numpy()})

  for x, y in l:
    x, y = x.numpy(), y.numpy()
    x = x.transpose((0, 2, 3, 1))

    x_adv = attack(x, y)

    clean_logits = sess.run(logits, {x_pl: x})
    adv_logits = sess.run(logits, {x_pl: x_adv})
    is_adv = adv_logits.argmax(-1) != y

    print(is_adv, clean_logits, adv_logits)

    return is_adv, (torch.tensor(x_adv.transpose((0, 3, 1, 2))),
                    torch.tensor(adv_logits))


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
    model = Model(dataset=config['dataset'],
                  train_batch_size=config['eval_batch_size'],
                  normalize_zero_mean=True,
                  zero_one=True,
                  # added by AUTHOR
                  mode='eval')
  else:
    model = Model(dataset=config['dataset'],
                  train_batch_size=config['eval_batch_size'],
                  zero_one=True,
                  # added by AUTHOR
                  mode='eval')
  print("model eval mode:", model.mode)
  sess = tf.Session()
  saver = tf.train.Saver()
  # Restore the checkpoint
  saver.restore(sess, model_file)

  binarized_model = BinarizedModel(model,
                                   logit_diff_loss=config['attack'] == 'pgd-ld')

  print("Using attack:", config['attack'])
  if config['attack'] == 'pgd' or config['attack'] == 'pgd-ld':
    attack = LinfPGDAttack(binarized_model,
                           config['epsilon'] / 255.0,
                           config['num_steps'],
                           config['step_size'] / 255.0,
                           config['random_start'],
                           config['loss_func'],
                           dataset=config['dataset'],
                           clip_max=1.0)
    attack_fn = lambda x, y: attack.perturb(x, y, sess)
  elif config['attack'] == 'apgd':
    from autoattack import autopgd_base
    from autoattack_adapter import ModelAdapter
    autoattack_model = ModelAdapter(
        binarized_model.pre_softmax, binarized_model.x_input,
        binarized_model.y_input, sess, num_classes=2, device="cpu")
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

  data_path = config['data_path']

  print("load cifar10 dataset")
  cifar = cifar10_input.CIFAR10Data(data_path)

  # Iterate over the samples batch-by-batch
  num_eval_examples = config['num_eval_examples']
  eval_batch_size = config['eval_batch_size']

  x_data = cifar.eval_data.xs[:num_eval_examples]
  y_data = cifar.eval_data.ys[:num_eval_examples]
  x_data = x_data.transpose((0, 3, 1, 2)) / 255.0
  assert x_data.max() <= 1 and x_data.min() >= 0, (x_data.min(), x_data.max())

  test_loader = utils.build_dataloader_from_arrays(x_data, y_data,
                                                   eval_batch_size)

  def feature_extractor_forward_pass(x, features_and_logits: bool = False,
      features_only: bool = False):
    if features_and_logits:
      assert not features_only, "Only one of the flags must be set."
    if features_and_logits:
      return sess.run(
          (model.neck, model.pre_softmax),
          feed_dict={model.x_input: x.transpose(0, 2, 3, 1)})
    elif features_only:
      return sess.run(
          model.neck,
          feed_dict={model.x_input: x.transpose(0, 2, 3, 1)})
    else:
      return sess.run(
          model.pre_softmax,
          feed_dict={model.x_input: x.transpose(0, 2, 3, 1)})

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=feature_extractor_forward_pass,
      logit_forward_and_backward_pass=None,
  )

  attack_fn_partial = partial(
      run_attack,
      sess=sess, logits=binarized_model.pre_softmax,
      x_pl=model.x_input,
      bro_w_pl=binarized_model.bro_w_pl, bro_b_pl=binarized_model.bro_b_pl,
      bro_w_assign=binarized_model.bro_w_set_weight,
      bro_b_assign=binarized_model.bro_b_set_weight,
      attack=attack_fn)

  scores_logit_differences_and_validation_accuracies = \
    dbb.interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        # m, l, sess, logits, x_pl, is_train, bro_w_pl, bro_b_pl,
        #     bro_w_assign, bro_b_assign, attack_fn)
        attack_fn=lambda m, l, kwargs: attack_fn_partial(m, l),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=config["epsilon"] / 255.0,
            norm="linf",
            lr=10000,
            n_boundary_points=config["n_boundary_points"],
            n_inner_points=config["n_inner_points"],
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=config["num_eval_examples"],
        device="cpu",
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        #rescale_logits="adaptive",
        sample_training_data_from_corners=config["sample_from_corners"],
        #decision_boundary_closeness=0.9999,
        fail_on_exception=False
        # args.num_samples_test * 10
    )

  print(dbb.format_result(scores_logit_differences_and_validation_accuracies,
                          config["num_eval_examples"]))


if __name__ == "__main__":
  main()
