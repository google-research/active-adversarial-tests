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
from __future__ import unicode_literals

# disable tf logging
# some of these might have to be commented out to use verbose=True in the
# adaptive attack
import os

import torch

from ml_loo import collect_layers
from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper
from utils import build_dataloader_from_arrays

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

try:
  import cPickle as pickle
except:
  import _pickle as pickle

import cleverhans
from cleverhans.utils_keras import KerasModelWrapper

from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack


import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm

from build_model import ImageModel
from load_data import ImageData, split_data
from attack_model import BIM, CW, FMA



if __name__ == '__main__':
  import argparse
  parser = argparse.ArgumentParser()
  parser.add_argument('--dataset_name', type = str,
                      choices = ['cifar10'],
                      default = 'cifar10')
  parser.add_argument('--model_name', type = str,
                      choices = ['resnet'],
                      default = 'resnet')
  parser.add_argument(
      '--attack',
      type = str,
      choices = ['cw', 'bim', 'fma'],
      default = 'cw'
  )
  parser.add_argument(
      '--detector-attack',
      type = str,
      choices = ['cw', 'bim', 'bim2'],
      default = 'cw'
  )
  parser.add_argument("--n-samples", type=int, default=500)

  parser.add_argument("--n-boundary-points", type=int, default=1)
  parser.add_argument("--n-inner-points", type=int, default=999)

  # default equals value for FPPR5; obtained from train_and_evaluate.py
  parser.add_argument("--detector-threshold", type=float, default=0.6151412488088068)

  parser.add_argument("--inverted-test", action="store_true")
  args = parser.parse_args()

  if args.inverted_test:
    print("Running inverted test")
  else:
    print("Running normal/non-inverted test")

  dict_a = vars(args)
  args.data_model = args.dataset_name + args.model_name

  # load detector
  with open(f"{args.data_model}/models/ml_loo_{args.detector_attack}_lr.pkl", "rb") as f:
    lr = pickle.load(f)

  print('Loading dataset...')
  dataset = ImageData(args.dataset_name)
  model = ImageModel(args.model_name, args.dataset_name, train = False, load = True)

  class ModelWrapper(cleverhans.model.Model):
    def __init__(self, model, sess, input_ph, weight_shape, bias_shape):
      self.weight = tf.placeholder(dtype=tf.float32, shape=weight_shape)
      self.bias = tf.placeholder(dtype=tf.float32, shape=bias_shape)
      self.model = model
      self.sess = sess
      self.input_ph = input_ph
      self.num_classes = 2
      self.first = True

    def fprop(self, x):
      y = self.model.get_layer(x, "flatten_1")
      logits = y @ tf.transpose(self.weight) + tf.reshape(self.bias, (1, -1))
      return {"logits": logits, "probs": tf.nn.softmax(logits, -1), "predictions": tf.argmax(logits, -1)}

    def get_probs(self, x):
      return self.fprop(x)["probs"]

    def predict(self, x, weights_feed_dict, logits=True):
      if self.first:
        self.targets = self.fprop(self.input_ph)
        self.first = False
      targets = self.targets
      if logits:
        target = targets["logits"]
      else:
        target = targets["probs"]
      return self.sess.run(target, {self.input_ph: x, **weights_feed_dict})

  keras_model = KerasModelWrapper(model.model)
  wrapped_model = ModelWrapper(keras_model, model.sess, model.input_ph, (2, 256), (2,))

  features = keras_model.get_layer(model.input_ph, "flatten_1")
  feature_gradients = tf.gradients(features, model.input_ph)[0]
  logits = keras_model.get_logits(model.input_ph)

  def run_features(x: np.ndarray, features_only=True, features_and_logits=False):
    if features_only:
      assert not features_and_logits
      targets = features
    elif features_and_logits:
      targets = (features, logits)
    else:
      targets = logits
    return model.sess.run(targets,
                    feed_dict={model.input_ph: x.transpose(0, 2, 3, 1)})

  def run_features_and_gradients(x: np.ndarray):
    return model.sess.run((features, feature_gradients),
                    feed_dict={model.input_ph: x.transpose(0, 2, 3, 1)})

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=lambda x, features_only = False,
          features_and_logits = False: run_features(x, features_only,
                                                    features_and_logits),
      logit_forward_and_backward_pass=lambda x: run_features_and_gradients(x)
  )

  if args.dataset_name == 'cifar10':
    X_train, Y_train, X_test, Y_test = split_data(dataset.x_val,
                                                  dataset.y_val, model, num_classes = 10,
                                                  split_rate = 0.8, sample_per_class = 1000)
  else:
    raise NotImplementedError()

  if args.n_samples == -1:
    args.n_samples = len(X_test)

  X_test = X_test[:args.n_samples]
  Y_test = Y_test[:args.n_samples]

  from ml_loo import get_ml_loo_features

  if args.model_name == 'resnet':
    interested_layers = [14,24,35,45,56,67,70]
  else:
    raise ValueError()

  # only relevant feature used by logistic regression model
  stat_names = ['quantile']
  reference = - dataset.x_train_mean
  get_ml_loo_features_ = lambda x: get_ml_loo_features(model, x, reference, interested_layers, stat_names=stat_names)[:, 0]
  detector = lambda x: lr.predict_proba(get_ml_loo_features_(x))[:, 1]

  batch_size = 50
  detector_threshold = args.detector_threshold

  if args.attack == 'cw':
    if args.dataset_name in ['cifar10']:
      if args.model_name == 'resnet':
        attack_model = CW(
            wrapped_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            wrapped_model.num_classes,
            source_samples = 1,
            binary_search_steps = 5,
            cw_learning_rate = 1e-2,
            confidence = 0,
            attack_iterations = 100,
            attack_initial_const = 1e-2,
        )
        original_attack_model = CW(
            keras_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            model.num_classes,
            source_samples = 1,
            binary_search_steps = 5,
            cw_learning_rate = 1e-2,
            confidence = 0,
            attack_iterations = 100,
            attack_initial_const = 1e-2,
        )
  elif args.attack == "bim":
    if args.dataset_name in ['cifar10']:
      if args.model_name == 'resnet':
        attack_model = BIM(
            wrapped_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            wrapped_model.num_classes,
            attack_iterations = 100,
            epsilon=0.03,
            learning_rate=2.5 * 0.03 / 100,
            random_init=True
        )
        original_attack_model = BIM(
            keras_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            model.num_classes,
            attack_iterations = 100,
            epsilon=0.03,
            learning_rate=2.5 * 0.03 / 100,
            random_init=True
        )
  elif args.attack == "fma":
    if args.dataset_name in ['cifar10']:
      if args.model_name == 'resnet':
        target_samples = []
        for y in range(10):
          target_samples.append(X_train[np.argmax(Y_train == y)])
        target_samples = np.array(target_samples)
        attack_model = FMA(
            model,
            wrapped_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            wrapped_model.num_classes,
            target_samples=target_samples[:2],
            reference=reference,
            features=collect_layers(model, interested_layers),
            attack_iterations = 500,
            epsilon=0.03,
            learning_rate=4 * 0.03 / 100,
            num_random_features=3100,
            random_init=True
        )
        original_attack_model = BIM(
            keras_model,
            wrapped_model.sess,
            wrapped_model.input_ph,
            model.num_classes,
            attack_iterations = 100,
            epsilon=0.03,
            learning_rate=2.5 * 0.03 / 100,
            random_init=True
        )

  assert 0 < X_test.max() <= 1.0, (X_test.min(), X_test.max())
  test_loader = build_dataloader_from_arrays(X_test.transpose((0, 3, 1, 2)), Y_test, batch_size=32)

  def run_attack(m, l, attack_kwargs):
    # attack_kwargs contains values that might be useful for e.g. constructing logit matching evasion attacks
    if args.attack == "fma":
      reference_points = attack_kwargs["reference_points_x"]
      if len(reference_points) < 2:
        reference_points = np.concatenate([reference_points, reference_points], 0)
      reference_points = reference_points.transpose((0, 2, 3, 1))
      attack_model.target_samples = reference_points
    else:
      del attack_kwargs
    linear_layer = m[-1]
    del m

    weights_feed_dict = {
        wrapped_model.weight: linear_layer.weight.data.numpy(),
        wrapped_model.bias: linear_layer.bias.data.numpy()
    }

    for x, y in l:
      x = x.numpy()
      x = x.transpose((0, 2, 3, 1))
      assert len(x) == 1
      x_adv = attack_model.attack(x, feedable_dict=weights_feed_dict)

      logits_adv = wrapped_model.predict(
          x_adv, weights_feed_dict=weights_feed_dict, logits=True)
      y_adv = logits_adv.argmax(-1)

      is_adv = y_adv != y
      is_not_detected = verify_input_data_fn(torch.tensor(x_adv.transpose((0, 3, 1, 2))))
      is_adv_and_not_detected = np.logical_and(is_adv, is_not_detected)
      is_adv_and_detected = np.logical_and(is_adv, ~is_not_detected)

      if args.inverted_test:
        test_result = is_adv_and_detected
      else:
        test_result = is_adv_and_not_detected

      return test_result, (torch.tensor(x_adv), torch.tensor(logits_adv))


  def get_boundary_adversarials(x, y, n_samples, epsilon):
    """Generate adversarial examples for the base classifier that get
    rejected by detector."""

    assert len(x.shape) == 3
    x = x.unsqueeze(0)
    x = torch.repeat_interleave(x, n_samples, dim=0)

    x = x.numpy()
    x = x.transpose((0, 2, 3, 1))

    for _ in range(25):
      x_adv = original_attack_model.attack(x)
      diff = x_adv - x
      diff = diff / np.max(np.abs(diff)) * epsilon
      x_adv = np.clip(x + diff, 0, 1)
      is_detected = ~verify_input_data_fn(torch.tensor(x_adv.transpose((0, 3, 1, 2))))

      if np.all(is_detected):
        # generative until we finally found (an) adversarial example(s) that
        # get(s) detected
        break
    else:
      warnings.warn("Could not generate adversarial example that gets "
                    "detected after 25 trials.")
    x_adv = x_adv.transpose((0, 3, 1, 2))

    return torch.tensor(x_adv)


  from argparse_utils import DecisionBoundaryBinarizationSettings
  from active_tests.decision_boundary_binarization import format_result

  if args.inverted_test:
    additional_settings = dict(
        n_boundary_points=args.n_boundary_points,
        n_boundary_adversarial_points=1,
        n_far_off_boundary_points=1,
        n_far_off_adversarial_points=1,
    )
  else:
    additional_settings = dict(
        n_boundary_points=args.n_boundary_points,
        n_boundary_adversarial_points=args.n_boundary_points - 1,
        n_far_off_boundary_points=1,
        n_far_off_adversarial_points=0,
    )

  far_off_distance = 1.75

  def verify_input_data_fn(x: torch.Tensor) -> np.ndarray:
    """Checks if detector does not reject input data as adversarial, i.e.
    input is clean."""
    #if args.inverted_test:
    #  return detector(x.numpy().transpose((0, 2, 3, 1))) > detector_threshold
    #else:
    return detector(x.numpy().transpose((0, 2, 3, 1))) < detector_threshold

  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        attack_fn=lambda m, l, attack_kwargs: run_attack(m, l, attack_kwargs),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=0.03,
            norm="linf",
            lr=10000,
            adversarial_attack_settings=None,
            optimizer="sklearn",
            n_inner_points=args.n_inner_points,
            **additional_settings
        ),
        n_samples=args.n_samples,
        device="cpu",
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,

        verify_valid_boundary_training_data_fn=verify_input_data_fn,
        get_boundary_adversarials_fn=get_boundary_adversarials,
        verify_valid_inner_training_data_fn=None,
        verify_valid_input_validation_data_fn=None,
        fill_batches_for_verification=False,
        far_off_distance=far_off_distance,
        rejection_resampling_max_repetitions=25,
        rescale_logits="adaptive"
    )

  print(format_result(scores_logit_differences_and_validation_accuracies, args.n_samples))






