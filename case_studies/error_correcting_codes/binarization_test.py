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
import warnings

import torch
from cleverhans.attacks import ProjectedGradientDescent
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from Model_Implementations import Model_Softmax_Baseline, \
  Model_Logistic_Baseline, Model_Logistic_Ensemble, Model_Tanh_Ensemble, \
  Model_Tanh_Baseline
from tensorflow.keras.datasets import cifar10
from tensorflow.keras import backend
import tensorflow as tf;
import numpy as np
import scipy.linalg
from adversarial_evaluation import setup_model_and_data, patch_pgd_loss
from active_tests import decision_boundary_binarization as dbb
from functools import partial

from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper
from utils import build_dataloader_from_arrays


def train_classifier(
    n_features: int,
    train_loader: DataLoader,
    raw_train_loader: DataLoader,
    logits: torch.Tensor,
    device: str,
    rescale_logits: dbb.LogitRescalingType,
    model,
    sess,
):
  del raw_train_loader

  # fit a linear readout for each of the submodels of the ensemble
  assert len(train_loader.dataset.tensors[0].shape) == 3
  assert train_loader.dataset.tensors[0].shape[1] == len(model.binarized_readouts)

  classifier_weights = []
  classifier_biases = []
  for i in range(len(model.binarized_readouts)):
    x_ = train_loader.dataset.tensors[0][:, i]
    y_ = train_loader.dataset.tensors[1]

    cls = dbb._train_logistic_regression_classifier(
        n_features,
        DataLoader(TensorDataset(x_, y_), batch_size=train_loader.batch_size),
        logits[:, i] if logits is not None else None,
        "sklearn",
        20000,
        device,
        n_classes=2,
        rescale_logits=rescale_logits,
        solution_goodness="good",
        class_weight="balanced"
    )
    classifier_weights.append(cls.weight.data.cpu().numpy().transpose()[:, [0]])
    classifier_biases.append(cls.bias.data.cpu().numpy()[0])

  # update weights of the binary models
  for l, vw, vb in zip(model.binarized_readouts, classifier_weights, classifier_biases):
    l.set_weights([vw, vb.reshape((1,))])

  return BinarizedModelWrapper(model, sess)


class BinarizedModelWrapper:
  def __init__(self, model, sess):
    self.model = model
    self.sess = sess

  def __call__(self, x):
    x = x.numpy()
    x = x.transpose((0, 2, 3, 1))
    p = self.sess.run(self.model.binarized_probs, {self.model.input: x})
    return torch.tensor(p)


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

  parser.add_argument("-n-inner-points", type=int, default=999)
  parser.add_argument("-n-boundary-points", type=int, default=1)
  parser.add_argument("--sample-from-corners", action="store_true")
  args = parser.parse_args()

  model, (X_valid, Y_valid), (X_test, Y_test) = setup_model_and_data(adaptive_attack=args.adaptive_attack)
  model.defineBinarizedModel()
  binarized_model_ch = model.modelBinarizedCH()

  if args.adaptive_attack:
    patch_pgd_loss()

  attack = ProjectedGradientDescent(binarized_model_ch, sess=sess)
  att_params = {'clip_min': 0.0, 'clip_max': 1.0,
                'eps': args.eps / 255.0, 'eps_iter': args.pgd_step_size / 255.0,
                'nb_iter': args.pgd_n_steps, 'ord': np.inf}
  x_ph = tf.placeholder(shape=model.input.shape, dtype=tf.float32)
  x_adv_op = attack.generate(x_ph, **att_params)

  def _model_forward_pass(x_np, features_only=False, features_and_logits=False):
    x_np = np.transpose(x_np, (0, 2, 3, 1))

    if features_only:
      f = sess.run(model.features, {model.input : x_np})
      f = np.stack(f, 1)
      return f
    elif features_and_logits:
      f, l = sess.run((model.features,
                        model.logits), {model.input : x_np})
      f = np.stack(f, 1)
      return f, l
    else:
      l = sess.run(model.logits, {model.input : x_np})
      return l


  def run_attack(m, l, sess):
    model = m.model
    for x, y in l:
      assert len(x) == 1
      x, y = x.numpy(), y.numpy()
      x = x.transpose((0, 2, 3, 1))
      x_adv = sess.run(x_adv_op, {x_ph: x})

      warnings.warn("ATTENTION: Clipping perturbation just to TEST something. Remove this again!")
      delta = x_adv - x
      delta[delta > 0] = args.eps / 255.0
      delta[delta < 0] = -args.eps / 255.0
      x_adv = np.clip(x + delta, 0, 1)

      logits, probs = sess.run((model.binarized_logit, model.binarized_probs),
                               {model.input: x_adv})
      is_adv = np.argmax(probs) != y
      return is_adv, (torch.tensor(x_adv.transpose((0, 3, 1, 2))), torch.tensor(logits))

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=_model_forward_pass,
      logit_forward_and_backward_pass=None
  )

  test_indices = list(range(len(X_test)))
  np.random.shuffle(test_indices)
  X_test, Y_test = X_test[test_indices], Y_test[test_indices]

  X_test = np.transpose(X_test, (0, 3, 1, 2))
  test_loader = build_dataloader_from_arrays(X_test, Y_test, batch_size=32)

  from argparse_utils import DecisionBoundaryBinarizationSettings
  scores_logit_differences_and_validation_accuracies = \
    dbb.interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        attack_fn=lambda m, l, kwargs: run_attack(m, l, sess),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=args.eps / 255.0,
            norm="linf",
            lr=25000,
            n_boundary_points=args.n_boundary_points,
            n_inner_points=args.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=args.n_samples,
        device="cpu",
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        train_classifier_fn=partial(
            train_classifier,
            model=model,
            sess=sess
        ),
        fail_on_exception=False,
        # needs to be set to None as logit rescaling introduces a weird behavior
        # of very high R. ASR (probably due to the log in the logit calculation)
        rescale_logits=None,
        decision_boundary_closeness=0.999,
        sample_training_data_from_corners=args.sample_from_corners

    )

  print(dbb.format_result(scores_logit_differences_and_validation_accuracies,
                          args.n_samples))

if __name__ == "__main__":
  main()