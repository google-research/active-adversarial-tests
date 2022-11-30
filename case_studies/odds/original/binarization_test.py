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

# disable tf logging
# some of these might have to be commented out to use verbose=True in the
# adaptive attack
import warnings
import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import os
import math
import numpy as np
import tensorflow as tf

from cleverhans.attacks import MadryEtAl
from cleverhans.dataset import CIFAR10
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import \
  make_wresnet as ResNet
from cleverhans.utils_tf import initialize_uninitialized_global_variables

import tf_robustify
from cleverhans.augmentation import random_horizontal_flip, random_shift

from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack, format_result
from argparse_utils import DecisionBoundaryBinarizationSettings
from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper

from logit_matching_attack import \
  ProjectedGradientDescentWithDetectorLogitMatching


def init_defense(sess, x, preds, batch_size, multi_noise=False):
  data = CIFAR10()

  if multi_noise:
    defense_data_path = os.path.join("checkpoints/tf_madry_wrn_vanilla",
                                     "defense_alignment_data_multi_noise")
  else:
    defense_data_path = os.path.join("checkpoints/tf_madry_wrn_vanilla",
                                     "defense_alignment_data")

  if os.path.exists(defense_data_path):
    print("Trying to load defense statistics")
    load_alignments_dir = defense_data_path
    save_alignments_dir = None
  else:
    print("Defense statistics not found; generating and saving them now.")
    load_alignments_dir = None
    save_alignments_dir = defense_data_path

  dataset_size = data.x_train.shape[0]
  dataset_train = data.to_tensorflow()[0]
  dataset_train = dataset_train.map(
      lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
  dataset_train = dataset_train.batch(batch_size)
  dataset_train = dataset_train.prefetch(16)
  x_train, y_train = data.get_set('train')
  x_train *= 255

  nb_classes = y_train.shape[1]

  n_collect = 10000  # TODO: for debugging set to 100, otherwise to 10000
  p_ratio_cutoff = .999
  just_detect = True
  clip_alignments = True
  fit_classifier = True
  noise_eps = 'n30.0'
  num_noise_samples = 256

  if multi_noise:
    noises = 'n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008'.split(
        ',')
    noise_eps_detect = []
    for n in noises:
      new_noise = n[0] + str(float(n[1:]) * 255)
      noise_eps_detect.append(new_noise)
  else:
    noise_eps_detect = 'n30.0'

  # these attack parameters are just for initializing the defense
  eps = 8.0
  pgd_params = {
      'eps': eps,
      'eps_iter': (eps / 5),
      'nb_iter': 10,
      'clip_min': 0,
      'clip_max': 255
  }

  logits_op = preds.op
  while logits_op.type != 'MatMul':
    logits_op = logits_op.inputs[0].op
  latent_x_tensor, weights = logits_op.inputs
  logits_tensor = preds

  predictor = tf_robustify.collect_statistics(
      x_train[:n_collect], y_train[:n_collect], x, sess,
      logits_tensor=logits_tensor,
      latent_x_tensor=latent_x_tensor,
      weights=weights,
      nb_classes=nb_classes,
      p_ratio_cutoff=p_ratio_cutoff,
      noise_eps=noise_eps,
      noise_eps_detect=noise_eps_detect,
      pgd_eps=pgd_params['eps'],
      pgd_lr=pgd_params['eps_iter'] / pgd_params['eps'],
      pgd_iters=pgd_params['nb_iter'],
      save_alignments_dir=save_alignments_dir,
      load_alignments_dir=load_alignments_dir,
      clip_min=pgd_params['clip_min'],
      clip_max=pgd_params['clip_max'],
      batch_size=batch_size,
      num_noise_samples=num_noise_samples,
      debug_dict=None,
      debug=False,
      targeted=False,
      pgd_train=None,
      fit_classifier=fit_classifier,
      clip_alignments=clip_alignments,
      just_detect=just_detect,
  )

  next(predictor)
  return predictor


def do_binarized_eval(sess, x, y, x_adv, binarized_logits, binarized_preds, x_set,
    y_set, predictor, batch_size, binarized_sess_kwargs={},
    inverted_detector_test=False):
  n_batches = math.ceil(x_set.shape[0] / batch_size)

  # first generative adversarial examples
  x_adv_set, binarized_logits_set, binarized_p_set = [], [], []
  for b in range(n_batches):
    values = sess.run((x_adv, binarized_logits, binarized_preds),
                      {**binarized_sess_kwargs,
                       x: x_set[b * batch_size:(b + 1) * batch_size],
                       y: y_set[b * batch_size:(b + 1) * batch_size]})
    x_adv_set.append(values[0])
    binarized_logits_set.append(values[1])
    binarized_p_set.append(values[2])
  x_adv_set = np.concatenate(x_adv_set)
  binarized_logits_set = np.concatenate(binarized_logits_set)
  binarized_p_set = np.concatenate(binarized_p_set)

  # now run test
  p_set, p_det = np.concatenate(
      [predictor.send(x_adv_set[b * batch_size:(b + 1) * batch_size]) for b in
       range(n_batches)]).T

  correctly_classified = np.equal(binarized_p_set,
                                  y_set[:len(p_set)].argmax(-1))
  if inverted_detector_test:
    adversarial_example_detected = np.equal(p_det, False)
  else:
    adversarial_example_detected = np.equal(p_det, True)
  # model_fooled = np.logical_or(
  #    np.logical_and(~correctly_classified, ~adversarial_example_detected), # fooled classifier & evaded detector
  #    np.logical_and(correctly_classified, adversarial_example_detected) # did not fool classifier but triggered detector (false positive)
  # )
  model_fooled = np.logical_and(~correctly_classified,
                                ~adversarial_example_detected)  # fooled classifier & evaded detector

  # print(len(adversarial_example_detected), np.sum(~correctly_classified),
  #      np.sum(adversarial_example_detected))

  # asr = model_fooled.mean()
  # acc = correctly_classified.mean()
  # print('Accuracy of base model: %0.4f' % acc)
  # print('ASR (w/ detection defense): %0.4f' % asr)
  #print(model_fooled, ~correctly_classified, ~adversarial_example_detected)
  #print(binarized_logits_set)

  return model_fooled, (x_adv_set, binarized_logits_set)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--multi-noise', action='store_true')
  parser.add_argument("--n-samples", default=512, type=int)
  parser.add_argument("--n-boundary-points", default=49, type=int)
  parser.add_argument("--n-inner-points", default=10, type=int)
  parser.add_argument("--batch-size", default=512, type=int)
  parser.add_argument("--attack", choices=("original", "adaptive",
                                           "adaptive-eot"),
                      default="original")
  parser.add_argument("--dont-verify-training-data", action="store_true")
  parser.add_argument("--use-boundary-adverarials", action="store_true")
  parser.add_argument("--inverted-test", action="store_true")
  args = parser.parse_args()

  if args.inverted_test:
    print("Running inverted test")
  else:
    print("Running normal/non-inverted test")

  # load data
  data = CIFAR10()
  x_test, y_test = data.get_set('test')

  sess = tf.Session()

  img_rows, img_cols, nchannels = x_test.shape[1:4]
  nb_classes = y_test.shape[1]

  # define model & restore weights
  # Define input TF placeholder
  x_placeholder = tf.placeholder(
      tf.float32, shape=(None, img_rows, img_cols, nchannels))
  y_placeholder = tf.placeholder(tf.int32, shape=(None,))
  # needed for adaptive attack
  x_reference_placeholder = tf.placeholder(
      tf.float32, shape=(None, img_rows, img_cols, nchannels))
  cifar_model = ResNet(scope='ResNet')

  ckpt = tf.train.get_checkpoint_state("checkpoints/tf_madry_wrn_vanilla")
  saver = tf.train.Saver(var_list=dict(
      (v.name.split('/', 1)[1].split(':')[0], v) for v in
      tf.global_variables()))
  saver.restore(sess, ckpt.model_checkpoint_path)
  initialize_uninitialized_global_variables(sess)

  class Model:
    def __init__(self, model):
      self.model = model

    def __call__(self, x, features_only=True):
      assert features_only
      return self.get_features(x)

    def get_features(self, x):
      return self.model.fprop(x * 255.0)["Flatten2"]

    def get_features_and_gradients(self, x):
      features = self.model.fprop(x * 255.0)["Flatten2"]
      grad = tf.gradients(features, x)[0]
      return features, grad

    def get_features_logits_and_gradients(self, x):
      values = self.model.fprop(x * 255.0)
      features = values["Flatten2"]
      predictions = values["logits"]
      grad = tf.gradients(features, x)[0]
      return features, grad, predictions

  model = Model(cifar_model)

  features, feature_gradients, logits = model.get_features_logits_and_gradients(
      x_placeholder)

  # setup defense
  # if multi_noise = True, instantiate the defense with 9 types of noise.
  # if multi_noise = False, instantiate the defense with a single type of high-magnitude noise.
  print("multi noise:", args.multi_noise)
  defense_predictor = init_defense(sess, x_placeholder, logits, args.batch_size,
                                   multi_noise=args.multi_noise)

  class ModelWrapper(cleverhans.model.Model):
    def __init__(self, model, weight_shape, bias_shape):
      self.weight = tf.placeholder(dtype=tf.float32, shape=weight_shape)
      self.bias = tf.placeholder(dtype=tf.float32, shape=bias_shape)
      self.model = model
      self.first = True

    def fprop(self, x, **kwargs):
      y = self.model.get_features(x, *kwargs)
      logits = y @ tf.transpose(self.weight) + tf.reshape(self.bias, (1, -1))
      return {"logits": logits}

    def logits_and_predictions(self, x=None):
      if x == None: assert not self.first
      if self.first:
        self.logits = self(x)
        self.predictions = tf.argmax(self.logits, 1)
        self.first = False
      return self.logits, self.predictions

  def run_features(x: np.ndarray, features_only=True,
      features_and_logits=False):
    if features_only:
      assert not features_and_logits
      targets = features
    elif features_and_logits:
      targets = (features, logits)
    else:
      targets = logits
    x = x.transpose(0, 2, 3, 1) * 255.0
    return sess.run(targets,
                    feed_dict={x_placeholder: x})

  def run_features_and_gradients(x: np.ndarray):
    x = x.transpose(0, 2, 3, 1) * 255.0
    return sess.run((features, feature_gradients),
                    feed_dict={x_placeholder: x})

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=lambda x, features_only=False,
          features_and_logits=False: run_features(x, features_only,
                                                  features_and_logits),
      logit_forward_and_backward_pass=lambda x: run_features_and_gradients(x)
  )

  # prepare dataloader
  random_indices = list(range(len(x_test)))
  np.random.shuffle(random_indices)
  x_batch = []
  y_batch = []
  for j in range(args.n_samples):
    x_, y_ = x_test[random_indices[j]], y_test[random_indices[j]]
    x_batch.append(x_)
    y_batch.append(y_)
  x_batch = np.array(x_batch).transpose((0, 3, 1, 2))
  y_batch = np.array(y_batch)

  from utils import build_dataloader_from_arrays

  test_loader = build_dataloader_from_arrays(x_batch, y_batch, batch_size=32)

  # TODO: update shapes? apparently not necessary...
  wrapped_model = ModelWrapper(model, (2, 640), (2,))

  baseline_cifar_pgd = MadryEtAl(cifar_model, sess=sess)
  original_pgd_params = {
      # ord: ,
      'eps': 8,
      'eps_iter': (8 / 5),
      'nb_iter': 10,
      'clip_min': 0,
      'clip_max': 255
  }
  adaptive_pgd_params = {
      # ord: ,
      'eps': 8,
      'eps_iter': 8.0 / 300,
      'nb_iter': 300,
      'clip_min': 0,
      'clip_max': 255,
      'x_reference': x_reference_placeholder,
      'y': y_placeholder
  }

  if args.attack == "original":
    pgd = MadryEtAl(wrapped_model, sess=sess)
    print("Using MadryEtAl attack")
  elif args.attack == "adaptive":
    pgd = ProjectedGradientDescentWithDetectorLogitMatching(
        wrapped_model,
        lambda x: model.model.get_logits(x),
        sess=sess,
        verbose=False)
    print("Using logit-matching attack")
  elif args.attack == "adaptive-eot":
    pgd = ProjectedGradientDescentWithDetectorLogitMatching(
        wrapped_model,
        lambda x: model.model.get_logits(x),
        sess=sess,
        eot_ensemble_size=20,
        verbose=False)
    print("Using logit-matching attack w/ EOT")
  else:
    raise ValueError("invalid attack")

  # was 1.75
  far_off_distance = 1.75  # TODO, was 1.01

  larger_pgd_params = {**original_pgd_params}
  larger_pgd_params["eps"] *= far_off_distance

  pgd_params = original_pgd_params if args.attack == "original" else adaptive_pgd_params

  adv_x = tf.stop_gradient(pgd.generate(x_placeholder, **pgd_params))

  cifar_adv_x = tf.stop_gradient(
      baseline_cifar_pgd.generate(x_placeholder, **original_pgd_params))
  larger_cifar_adv_x = tf.stop_gradient(
      baseline_cifar_pgd.generate(x_placeholder, **original_pgd_params))

  adv_binarized_logits = wrapped_model.get_logits(adv_x)
  adv_binarized_predictions = tf.argmax(adv_binarized_logits, 1)

  def run_attack(m, l, kwargs, inverted_detector_test=False):
    linear_layer = m[-1]
    del m

    weights_feed_dict = {
        wrapped_model.weight: linear_layer.weight.data.numpy(),
        wrapped_model.bias: linear_layer.bias.data.numpy()
    }

    if "reference_points_x" in kwargs:
      weights_feed_dict[x_reference_placeholder] = \
        kwargs["reference_points_x"].numpy().transpose((0, 2, 3, 1)) * 255.0

    # should_be_rejected = ~verify_valid_input_data(kwargs["reference_points_x"])
    # print("should_be_rejected", should_be_rejected)

    for x, y in l:
      x = x.numpy().transpose((0, 2, 3, 1)) * 255.0
      y = y.numpy()

      is_adv_np, (x_adv_np, logits_np) = do_binarized_eval(
          sess=sess, x=x_placeholder, y=y_placeholder, x_adv=adv_x,
          batch_size=args.batch_size,
          binarized_logits=adv_binarized_logits,
          binarized_preds=adv_binarized_predictions, x_set=x, y_set=y,
          predictor=defense_predictor, binarized_sess_kwargs=weights_feed_dict,
          inverted_detector_test=inverted_detector_test
      )

      # print(is_adv_np, y, logits_np)

      return is_adv_np, (torch.Tensor(x_adv_np), torch.Tensor(logits_np))

  def verify_valid_input_data(x_set):
    """Returns True if something is not detected as an adversarial example."""
    x_set = x_set.numpy().transpose((0, 2, 3, 1)) * 255.0
    n_batches = math.ceil(x_set.shape[0] / args.batch_size)
    _, p_det = np.concatenate(
        [defense_predictor.send(
            x_set[b * args.batch_size:(b + 1) * args.batch_size]
        ) for b in range(n_batches)]
    ).T
    # p_det is True of a possible adversarial example has been detected
    valid_sample = np.equal(p_det, False)

    return valid_sample

  def get_boundary_adversarials(x, y, n_samples, epsilon):
    """Generate adversarial examples for the base classifier."""
    assert len(x.shape) == 3
    del y
    device = x.device
    x = x.unsqueeze(0).numpy()
    x = x.transpose((0, 2, 3, 1)) * 255.0
    x = np.repeat(x, n_samples, axis=0)

    # select correct tf placeholder depending on the epsilon ball
    if epsilon == pgd_params["eps"] / 255.0:
      x_adv_ph = cifar_adv_x
    elif epsilon == larger_pgd_params["eps"] / 255.0:
      x_adv_ph = larger_cifar_adv_x
    else:
      raise ValueError("Cannot generate adversarials at eps =", epsilon)

    for _ in range(10):
      x_advs = []
      for x_ in np.array_split(x, int(np.ceil(len(x) / args.batch_size))):
        x_advs.append(sess.run(x_adv_ph, feed_dict={x_placeholder: x_}))
      x_adv = np.concatenate(x_advs, 0)
      x_adv = x_adv.transpose((0, 3, 1, 2)) / 255.0

      x_adv = torch.Tensor(x_adv, device=device)

      # make sure adversarial examples are really detected as adversarial examples
      is_valid = verify_valid_input_data(x_adv)
      is_invalid = ~is_valid
      if np.all(is_invalid):
        # generative until we finally found an adversarial example that gets
        # detected
        break
    else:
      warnings.warn("Could not generate adversarial example that gets "
                    "detected after 10 trials.")

    return x_adv

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

  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        attack_fn=functools.partial(
            run_attack,
            inverted_detector_test=args.inverted_test
        ),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=8 / 255.0,
            norm="linf",
            lr=10000,
            n_inner_points=args.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn",
            **additional_settings,
        ),
        rescale_logits="adaptive",
        n_samples=args.n_samples,
        device="cpu",
        batch_size=args.batch_size,
        # decision_boundary_closeness=0.999,
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        # verify_valid_boundary_training_data_fn=None if args.dont_verify_training_data else verify_valid_input_data,
        verify_valid_boundary_training_data_fn=verify_valid_input_data,
        get_boundary_adversarials_fn=get_boundary_adversarials,
        verify_valid_inner_training_data_fn=None,
        verify_valid_input_validation_data_fn=None,
        # verify_valid_input_data if args.use_boundary_adverarials else None,
        # get_boundary_adversarials_fn=get_boundary_adversarials if args.use_boundary_adverarials else None,
        fill_batches_for_verification=False,
        far_off_distance=far_off_distance
    )

  print(format_result(scores_logit_differences_and_validation_accuracies,
                      args.n_samples))


if __name__ == "__main__":
  main()
