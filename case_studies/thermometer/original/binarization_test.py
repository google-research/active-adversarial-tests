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

import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import torch

import robustml
from robustml_model import Thermometer
import sys
import argparse

import numpy as np
from robustml_attack import LSPGDAttack, Attack
from active_tests.decision_boundary_binarization import interior_boundary_discrimination_attack, format_result


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cifar-path', type=str, required=True,
                      help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
  parser.add_argument('--debug', action='store_true')
  parser.add_argument("--attack", default="adaptive", choices=("original", "adaptive", "modified", "modified2"))
  parser.add_argument("--n-samples", default=512, type=int)
  parser.add_argument("--n-boundary-points", default=49, type=int)
  parser.add_argument("--n-inner-points", default=10, type=int)
  parser.add_argument("--epsilon", default=8, type=int)
  parser.add_argument("--decision-boundary-closeness", type=float, default=None)
  parser.add_argument("--sample-from-corners", action="store_true")
  args = parser.parse_args()

  # set up TensorFlow session
  sess = tf.Session()

  # initialize a model
  model = Thermometer(sess, epsilon=args.epsilon)

  # initialize a data provider for CIFAR-10 images
  provider = robustml.provider.CIFAR10(args.cifar_path)

  random_indices = list(range(len(provider)))
  np.random.shuffle(random_indices)

  x_batch = []
  y_batch = []
  for j in range(args.n_samples):
    x_, y_ = provider[random_indices[j]]
    x_batch.append(x_)
    y_batch.append(y_)
  x_batch = np.array(x_batch).transpose((0, 3, 1, 2))
  y_batch = np.array(y_batch)

  from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper, PyTorchToTensorFlow1Wrapper
  from utils import build_dataloader_from_arrays

  test_loader = build_dataloader_from_arrays(x_batch, y_batch, batch_size=32)

  def _model_forward_pass(x, features_and_logits: bool = False, features_only: bool = False):
    if features_and_logits:
      assert not features_only, "Only one of the flags must be set."
    if features_and_logits:
      return model.get_features_and_logits(x.transpose(0, 2, 3, 1))
    elif features_only:
      return model.get_features(x.transpose(0, 2, 3, 1))
    else:
      raise ValueError

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=_model_forward_pass,
      logit_forward_and_backward_pass=lambda x: model.get_features_and_gradients(x.transpose(0, 2, 3, 1))
  )

  class ModelWrapper:
    def __init__(self, model, weight_shape, bias_shape):
      self.weight = tf.placeholder(dtype=tf.float32, shape=weight_shape)
      self.bias = tf.placeholder(dtype=tf.float32, shape=bias_shape)
      self.model = model
      self.first = True

    def __call__(self, x, **kwargs):
      y = self.model(x, features_only=True, **kwargs)
      logits = y @ tf.transpose(self.weight) + tf.reshape(self.bias, (1, -1))
      return logits

    def logits_and_predictions(self, x = None):
      if x == None: assert not self.first
      if self.first:
        self.logits = self(x)
        self.predictions = tf.argmax(self.logits, 1)
        self.first = False
      return self.logits, self.predictions

  wrapped_model = ModelWrapper(model.model, (2, 640), (2,))
  if args.attack == "adaptive":
    attack = Attack(sess, wrapped_model, epsilon=model.threat_model.epsilon, batch_size=1, n_classes=2)
  elif args.attack == "original":
    attack = LSPGDAttack(sess, wrapped_model, epsilon=model.threat_model.epsilon, n_classes=2)
  elif args.attack == "modified":
    attack = LSPGDAttack(sess, wrapped_model, epsilon=model.threat_model.epsilon, num_steps=50, step_size=0.25, n_classes=2)
  elif args.attack == "modified2":
    attack = LSPGDAttack(sess, wrapped_model, epsilon=model.threat_model.epsilon, num_steps=100, step_size=0.1, n_classes=2)
  else:
    raise ValueError("invalid attack mode")

  #@profile
  def run_attack(m, l, epsilon):
    linear_layer = m[-1]
    del m

    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
    # m = PyTorchToTensorFlow1Wrapper(m, "cpu")

    weights_feed_dict = {
        wrapped_model.weight: linear_layer.weight.data.numpy(),
        wrapped_model.bias: linear_layer.bias.data.numpy()
    }

    for x, y in l:
      x = x.numpy().transpose((0, 2, 3, 1))
      y = y.numpy()
      x_adv = attack.run(x, y, None, weights_feed_dict)

      x_adv = x_adv * 255.0
      if not args.attack in ("original", "modified", "modified2"):
        # first encode the input, then classify it
        x_adv = model.encode(x_adv)
      logits, y_adv = model._sess.run(
          wrapped_model.logits_and_predictions(model._model.x_input),
          {
              model._model.x_input: x_adv,
              **weights_feed_dict
          }
      )
      is_adv = (y_adv != y).mean()
      return is_adv, (torch.Tensor(x_adv), torch.Tensor(logits))


  from argparse_utils import DecisionBoundaryBinarizationSettings
  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
      feature_extractor,
      test_loader,
      attack_fn=lambda m, l, kw: run_attack(m, l, args.epsilon/255.0),
      linearization_settings=DecisionBoundaryBinarizationSettings(
          epsilon=args.epsilon/255.0,
          norm="linf",
          lr=10000,
          n_boundary_points=args.n_boundary_points,
          n_inner_points=args.n_inner_points,
          adversarial_attack_settings=None,
          optimizer="sklearn"
      ),
      n_samples=args.n_samples,
      device="cpu",
      n_samples_evaluation=200,
      n_samples_asr_evaluation=200,
      decision_boundary_closeness=args.decision_boundary_closeness,
      rescale_logits="adaptive",
      sample_training_data_from_corners=args.sample_from_corners
      #args.num_samples_test * 10
    )

  print(format_result(scores_logit_differences_and_validation_accuracies,
                      args.n_samples))



if __name__ == '__main__':
  main()
