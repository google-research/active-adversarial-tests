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

from cleverhans.utils_keras import KerasModelWrapper

from ml_loo import collect_layers

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
      choices = ['cw', 'bim', 'bim2', 'fma'],
      default = 'cw'
  )
  parser.add_argument("--batch-size", type=int, default=50)
  parser.add_argument("--detector-attack", choices=['cw', 'bim', 'bim2'], default='cw')
  parser.add_argument("--n-samples", type=int, default=500)

  # default equals value for FPPR5; obtained from train_and_evaluate.py
  parser.add_argument("--detector-threshold", type=float, default=0.6151412488088068)

  args = parser.parse_args()
  dict_a = vars(args)
  args.data_model = args.dataset_name + args.model_name

  # load detector
  with open(f"{args.data_model}/models/ml_loo_{args.detector_attack}_lr.pkl", "rb") as f:
    lr = pickle.load(f)

  print('Loading dataset...')
  dataset = ImageData(args.dataset_name)
  model = ImageModel(args.model_name, args.dataset_name, train = False, load = True)

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

  batch_size = args.batch_size
  detector_threshold = args.detector_threshold

  if args.attack == 'cw':
    if args.dataset_name in ['cifar10']:
      if args.model_name == 'resnet':
        attack_model = CW(
            model,
            source_samples = batch_size,
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
            KerasModelWrapper(model.model),
            model.sess,
            model.input_ph,
            model.num_classes,
            attack_iterations = 100,
            epsilon=0.03,
            learning_rate=2.5 * 0.03 / 100,
            random_init=True
        )
  elif args.attack == "bim2":
    if args.dataset_name in ['cifar10']:
      if args.model_name == 'resnet':
        attack_model = BIM(
            KerasModelWrapper(model.model),
            model.sess,
            model.input_ph,
            model.num_classes,
            attack_iterations = 10,
            epsilon=0.03,
            learning_rate=2.5 * 0.03 / 10,
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
            KerasModelWrapper(model.model),
            model.sess,
            model.input_ph,
            model.num_classes,
            target_samples=target_samples,
            reference=reference,
            features=collect_layers(model, interested_layers),
            attack_iterations = 500,
            epsilon=0.03,
            learning_rate=4 * 0.03 / 100,
            num_random_features=3100,
            random_init=True
        )


  n_batches = int(np.ceil(len(X_test) / batch_size))

  all_is_adv = []
  all_is_detected = []
  all_is_adv_not_detected = []
  pbar = tqdm(range(n_batches))
  for i in pbar:
    x = X_test[i * batch_size:(i+1) * batch_size]
    y = Y_test[i * batch_size:(i+1) * batch_size]
    # undo one-hot encoding
    y = y.argmax(-1)

    x_adv = attack_model.attack(x)
    y_adv = model.predict(x_adv, verbose=False, logits=False).argmax(-1)

    is_adv = y_adv != y
    is_detected = detector(x_adv) > detector_threshold
    all_is_adv.append(is_adv)
    all_is_detected.append(is_detected)

    is_adv_not_detected = np.logical_and(is_adv, ~is_detected)
    all_is_adv_not_detected.append(is_adv_not_detected)

    pbar.set_description(
        f"ASR (w/o detector): {np.mean(np.concatenate(all_is_adv))} "
        f"ASR (w/ detector): {np.mean(np.concatenate(all_is_adv_not_detected))}")

  all_is_adv = np.concatenate(all_is_adv)
  all_is_detected = np.concatenate(all_is_detected)
  all_is_adv_not_detected = np.concatenate(all_is_adv_not_detected)
  print("ASR (w/o detector):", np.mean(all_is_adv))
  print("ASR (w/ detector):", np.mean(all_is_adv_not_detected))








