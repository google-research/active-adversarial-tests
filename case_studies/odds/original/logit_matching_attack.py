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

import argparse
import functools

import cleverhans.model
import torch
from cleverhans import utils_tf
from cleverhans.attacks import Attack
import cleverhans.attacks
from cleverhans.utils_tf import clip_eta

import os
import math
import numpy as np
import tensorflow as tf


class ProjectedGradientDescentWithDetectorLogitMatching(Attack):
  def __init__(self, model, get_features_for_detector,
      sess=None, dtypestr='float32',
      default_rand_init=True, verbose=False, eot_ensemble_size=None,
      eot_multinoise=False, **kwargs):
    """
    Create a ProjectedGradientDescent instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(ProjectedGradientDescentWithDetectorLogitMatching, self).__init__(model, sess=sess,
                                                                            dtypestr=dtypestr, **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'clip_min',
                            'clip_max', 'loss_lambda')
    self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
    self.default_rand_init = default_rand_init
    self.get_features_for_detector = get_features_for_detector
    self.verbose = verbose

    self.eot_ensemble_size = eot_ensemble_size
    assert eot_ensemble_size is None or eot_ensemble_size > 0
    self.eot_multinoise = eot_multinoise

  def generate(self, x, x_reference, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x,
                                                   tf.cast(self.clip_min,
                                                           x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x,
                                                tf.cast(self.clip_max,
                                                        x.dtype)))

    # Initialize loop variables
    if self.rand_init:
      eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-self.rand_minmax, x.dtype),
                              tf.cast(self.rand_minmax, x.dtype),
                              dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    fgm_params = {
        'eps': self.eps_iter,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max,
        "eot_ensemble_size": self.eot_ensemble_size,
        "eot_multinoise": self.eot_multinoise,
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    def cond(i, _):
      return tf.less(i, self.nb_iter)

    def body(i, adv_x):
      adv_x = self.fgm_generate(x_adv=adv_x,
                                x_reference=x_reference,
                                **fgm_params, step=i)

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      return i + 1, adv_x

    _, adv_x = tf.while_loop(cond, body, [tf.zeros([]), adv_x], back_prop=True)


    # Asserts run only on CPU.
    # When multi-GPU eval code tries to force all PGD ops onto GPU, this
    # can cause an error.
    #asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps_iter,
    #                                                  dtype=self.eps.dtype),
    #                                          self.eps))
    if self.ord == np.inf and self.clip_min is not None:
      # The 1e-6 is needed to compensate for numerical error.
      # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
      # clip_max=.7
      asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps, x.dtype),
                                                1e-6 + tf.cast(self.clip_max,
                                                               x.dtype)
                                                - tf.cast(self.clip_min,
                                                          x.dtype)))

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def fgm_generate(self,
      x_adv,
      x_reference,
      step,
      eps=0.3,
      ord=np.inf,
      clip_min=None,
      clip_max=None,
      targeted=False,
      sanity_checks=True,
      eot_ensemble_size=None,
      eot_multinoise=False):
    asserts = []

    # If a data range was specified, check that the input was in that range
    if clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(
          x_adv, tf.cast(clip_min, x_adv.dtype)))

    if clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x_adv, tf.cast(clip_max, x_adv.dtype)))

    if targeted:
      raise ValueError("targeted mode not supported")

    # while this check looks good in theory in pratice it doesnt make sense
    # since the softmax op might not add an additional AddV2 operation at the end
    # Make sure the caller has not passed probs by accident
    #assert logits.op.type != 'Softmax'
    #assert target_logits.op.type != 'Softmax'

    target_detector_features = tf.stop_gradient(
        self.get_features_for_detector(x_reference))
    labels = tf.one_hot(self.y, 2)
    if eot_ensemble_size is None:
      # no EOT
      detector_features = self.get_features_for_detector(x_adv)
      classifier_logits = self.model.get_logits(x_adv)
      real = tf.reduce_sum(labels * classifier_logits, -1)
      other = tf.reduce_max((1-labels) * classifier_logits - (labels*10000), -1)
      classifier_loss = -tf.clip_by_value(real - other, -1e-2, 1e9)
      detector_features_matching_loss = -tf.reduce_mean(
          tf.reduce_sum((detector_features - target_detector_features)**2,-1),
          0)
      loss = self.loss_lambda * classifier_loss + (1.0 - self.loss_lambda) * detector_features_matching_loss

      # Define gradient of loss wrt input
      grad, = tf.gradients(loss, x_adv)
      grad = tf.stop_gradient(grad)
    else:
      grads = []
      for i in range(eot_ensemble_size):
        if i == 0:
          # dont add noise to first forward pass
          x_adv_noisy = x_adv
        else:
          if eot_multinoise:
            if i % 2 == 0:
              noise = tf.random.normal(tf.shape(x_adv), 0.0, 1.0)
            elif i % 2 == 1:
              noise = tf.random.uniform(tf.shape(x_adv), -1.0, 1.0)
            else:
              # defined in https://github.com/wielandbrendel/adaptive_attacks_paper/blob/master/02_odds/Attack.ipynb
              # but doesnt make sense to me since this never gets called
              noise = tf.sign(tf.random.uniform(tf.shape(x_adv), -1.0, 1.0))
            noise *= 0.01 * 255.0
          else:
            noise = tf.random.normal(tf.shape(x_adv), 0.0, 1.0)
            noise *= 2.0
          x_adv_noisy = tf.clip_by_value(x_adv + noise, 0, 255.0)
        detector_features = self.get_features_for_detector(x_adv_noisy)
        classifier_logits = self.model.get_logits(x_adv_noisy)
        real = tf.reduce_sum(labels * classifier_logits, -1)
        other = tf.reduce_max((1-labels) * classifier_logits - (labels*10000), -1)
        classifier_loss = -tf.clip_by_value(real - other, -1e-2, 1e9)
        detector_features_matching_loss = -tf.reduce_mean(
            tf.reduce_sum((detector_features - target_detector_features)**2,-1),
            0)
        loss = self.loss_lambda * classifier_loss + (1.0 - self.loss_lambda) * detector_features_matching_loss

        # Define gradient of loss wrt input
        grad, = tf.gradients(loss, x_adv_noisy)
        grad = tf.stop_gradient(grad)
        grads.append(grad)
      grad = tf.reduce_mean(grads, axis=0)

    optimal_perturbation = cleverhans.attacks.optimize_linear(grad, eps, ord)

    # Add perturbation to original example to obtain adversarial example
    adv_x = x_adv + optimal_perturbation

    # If clipping is needed, reset all values outside of [clip_min, clip_max]
    if (clip_min is not None) or (clip_max is not None):
      # We don't currently support one-sided clipping
      assert clip_min is not None and clip_max is not None
      adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

    if sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    if self.verbose:
      adv_x = tf.Print(adv_x, [step, loss, classifier_loss, detector_features_matching_loss])

    return adv_x

  def parse_params(self,
      eps=0.3,
      eps_iter=0.05,
      nb_iter=10,
      y=None,
      ord=np.inf,
      clip_min=None,
      clip_max=None,
      y_target=None,
      rand_init=None,
      rand_minmax=0.3,
      sanity_checks=True,
      loss_lambda=0.5,
      **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    if rand_init is None:
      rand_init = self.default_rand_init
    self.rand_init = rand_init
    if self.rand_init:
      self.rand_minmax = eps
    else:
      self.rand_minmax = 0.
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    self.loss_lambda = loss_lambda

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    self.sanity_checks = sanity_checks

    return True
