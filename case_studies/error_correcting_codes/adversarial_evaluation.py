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
Run this to attack a trained model via TrainModel. 
Use the "loadFullModel" submethod to load in an already trained model (trained via TrainModel)
The main attack function is "runAttacks" which runs attacks on trained models
"""

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

from cleverhans.attacks import ProjectedGradientDescent
from Model_Implementations import Model_Softmax_Baseline, \
  Model_Logistic_Baseline, Model_Logistic_Ensemble, Model_Tanh_Ensemble, \
  Model_Tanh_Baseline
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend
import tensorflow as tf;
import numpy as np
import scipy.linalg

model_path = 'checkpoints/ECOC/tanh32/checkpoints'  #path with saved model parameters

def setup_model_and_data(adaptive_attack=False):
  print("Modifying model for adaptive attack:", adaptive_attack)
  # Dataset-specific parameters - should be same as those used in TrainModel
  DATA_DESC = 'CIFAR10';
  (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()
  epochs = None;
  weight_save_freq = None
  num_classes = 10  # how many classes (categories) are in this dataset?
  Y_train = np.squeeze(Y_train);
  Y_test = np.squeeze(Y_test)
  num_filters_std = [32, 64, 128];
  num_filters_ens = [32, 64, 128];
  num_filters_ens_2 = 16;
  dropout_rate_std = 0.0;
  dropout_rate_ens = 0.0;
  weight_decay = 0
  model_rep_baseline = 2;
  model_rep_ens = 2;
  DATA_AUGMENTATION_FLAG = 1;
  BATCH_NORMALIZATION_FLAG = 1
  num_channels = 3;
  inp_shape = (32, 32, 3);
  lr = 1e-4;
  batch_size = 80;
  noise_stddev = 0.032;
  blend_factor = .032

  # DATA PRE-PROCESSING
  X_train = (X_train / 255).astype(np.float32);
  X_test = (X_test / 255).astype(np.float32)
  # reshape (add third (image) channel)
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                            X_train.shape[2],
                            num_channels);
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2],
                          num_channels)
  X_valid = X_test[1000:2000];
  Y_valid = Y_test[1000:2000];  # validation data, used to attack model

  ## ENSEMBLE TANH 32 MODEL DEFINITION
  name = 'tanh_32_diverse' + '_' + DATA_DESC;
  seed = 59;
  code_length = 32;
  num_codes = code_length;
  num_chunks = 4;
  base_model = None;

  def output_activation(x):
    if adaptive_attack:
      return x
    else:
      return tf.nn.tanh(x)

  M = scipy.linalg.hadamard(code_length).astype(np.float32)
  M[np.arange(0, num_codes,
              2), 0] = -1  # replace first col, which for scipy's Hadamard construction is always 1, hence not a useful classifier; this change still ensures all codewords have dot product <=0; since our decoder ignores negative correlations anyway, this has no net effect on probability estimation
  np.random.seed(seed)
  np.random.shuffle(M)
  idx = np.random.permutation(code_length)
  M = M[0:num_codes, idx[0:code_length]]
  params_dict = {'BATCH_NORMALIZATION_FLAG': BATCH_NORMALIZATION_FLAG,
                 'DATA_AUGMENTATION_FLAG': DATA_AUGMENTATION_FLAG, 'M': M,
                 'base_model': base_model, 'num_chunks': num_chunks,
                 'model_rep': model_rep_ens,
                 'output_activation': output_activation,
                 'num_filters_ens': num_filters_ens,
                 'num_filters_ens_2': num_filters_ens_2, 'batch_size': batch_size,
                 'epochs': epochs, 'dropout_rate': dropout_rate_ens, 'lr': lr,
                 'blend_factor': blend_factor, 'inp_shape': inp_shape,
                 'noise_stddev': noise_stddev,
                 'weight_save_freq': weight_save_freq, 'name': name,
                 'model_path': model_path,
                 'zero_one_input': True,
                 'adaptive_attack': adaptive_attack
                 }
  m5 = Model_Tanh_Ensemble({}, params_dict)
  m5.loadFullModel()  # load in the saved model, which should have already been trained first via TrainModel

  m5.legend = 'TEns32';

  model = m5

  return model, (X_valid, Y_valid), (X_test, Y_test)


def wbAttack(sess, model, x_ph, x_adv_op, X, Y, batch_size=500, verbose=True):
  n_correct = 0
  n_total = 0
  all_logits = []
  all_x_adv = []
  import tqdm
  pbar = np.arange(0, X.shape[0], batch_size)
  if verbose:
    pbar = tqdm.tqdm(pbar)

  for start_idx in pbar:
    x = X[start_idx:start_idx + batch_size]
    y = Y[start_idx:start_idx + batch_size]
    x_adv = sess.run(x_adv_op, {x_ph: x})
    logits = sess.run(model.logits, {model.input: x_adv})
    preds = np.argmax(logits, -1)
    n_correct += np.sum(np.equal(preds, y))
    n_total += len(x)
    all_logits.append(logits)
    all_x_adv.append(x_adv)

  all_x_adv = np.concatenate(all_x_adv, 0)
  all_logits = np.concatenate(all_logits, 0)

  adv_acc = n_correct / n_total
  return adv_acc, all_logits, all_x_adv


def patch_pgd_loss():
  import cleverhans

  def fgm(x,
      logits,
      y=None,
      eps=0.3,
      ord=np.inf,
      clip_min=None,
      clip_max=None,
      targeted=False,
      sanity_checks=True):

    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
      asserts.append(cleverhans.utils_tf.assert_greater_equal(
          x, tf.cast(clip_min, x.dtype)))

    if clip_max is not None:
      asserts.append(cleverhans.utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

    # Make sure the caller has not passed probs by accident
    assert logits.op.type != 'Softmax'

    if y is None:
      # Using model predictions as ground truth to avoid label leaking
      preds_max = tf.reduce_max(logits, 1, keepdims=True)
      y = tf.to_float(tf.equal(logits, preds_max))
      y = tf.stop_gradient(y)
    y = y / tf.reduce_sum(y, 1, keepdims=True)

    # Compute loss
    loss = loss_fn(labels=y, logits=logits)
    if targeted:
      loss = -loss

    # loss = tf.Print(loss, [loss])

    # Define gradient of loss wrt input
    grad, = tf.gradients(loss, x)

    optimal_perturbation = cleverhans.attacks.optimize_linear(grad, eps, ord)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
      # We don't currently support one-sided clipping
      assert clip_min is not None and clip_max is not None
      adv_x = cleverhans.utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def loss_fn(sentinel=None,
      labels=None,
      logits=None,
      dim=-1):
    """
    Wrapper around tf.nn.softmax_cross_entropy_with_logits_v2 to handle
    deprecated warning
    """
    # Make sure that all arguments were passed as named arguments.
    if sentinel is not None:
      name = "softmax_cross_entropy_with_logits"
      raise ValueError("Only call `%s` with "
                       "named arguments (labels=..., logits=..., ...)"
                       % name)
    if labels is None or logits is None:
      raise ValueError("Both labels and logits must be provided.")

    labels = tf.stop_gradient(labels)
    # modified from
    # https://github.com/carlini/nn_robust_attacks/blob/master/li_attack.py
    real = tf.reduce_sum(labels * logits, -1)
    other = tf.reduce_max((1-labels) * logits - (labels*10000), -1)

    loss = other - real

    # loss = tf.Print(loss, [loss])

    return loss

  cleverhans.attacks.fgm = fgm

def main():
  sess = backend.get_session()
  backend.set_learning_phase(
      0)  # need to do this to get CleverHans to work with batchnorm

  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument("--eps", type=float, default=8, help="in 0-255")
  parser.add_argument("--pgd-n-steps", default=200, type=int)
  parser.add_argument("--pgd-step-size", type=float, default=2 / 3 * 8, help="in 0-255")
  parser.add_argument("--n-samples", type=int, default=512)
  parser.add_argument("--adaptive-attack", action="store_true")
  args = parser.parse_args()

  model, (X_valid, Y_valid), (X_test, Y_test) = setup_model_and_data(adaptive_attack=args.adaptive_attack)

  test_indices = list(range(len(X_test)))
  np.random.shuffle(test_indices)
  X_test, Y_test = X_test[test_indices], Y_test[test_indices]

  X_test, Y_test = X_test[:args.n_samples], Y_test[:args.n_samples]

  model_ch = model.modelCH()
  attack = ProjectedGradientDescent(model_ch, sess=sess)
  att_params = {'clip_min': 0.0, 'clip_max': 1.0,
                'eps': args.eps / 255.0, 'eps_iter': args.pgd_step_size / 255.0,
                'nb_iter': args.pgd_n_steps, 'ord': np.inf,
                }
  if args.adaptive_attack:
    patch_pgd_loss()

  x_ph = tf.placeholder(shape=model.input.shape, dtype=tf.float32)
  x_adv_op = attack.generate(x_ph, **att_params)
  adv_acc, all_logits, all_x_adv = wbAttack(sess, model,
                                       x_ph, x_adv_op,
                                       X_test, Y_test, batch_size=512)

  print("Robust accuracy:", adv_acc)


if __name__ == "__main__":
  main()
