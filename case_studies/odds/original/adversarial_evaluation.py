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


def do_eval(sess, x, x_adv, logits, preds, x_set,
    y_set, predictor, batch_size, attack_kwargs={}):
  n_batches = math.ceil(x_set.shape[0] / batch_size)

  # first generative adversarial examples
  x_adv_set, logits_set, p_set = [], [], []
  for b in range(n_batches):
    values = sess.run((x_adv, logits, preds),
                      {**attack_kwargs,
                       x: x_set[b * batch_size:(b + 1) * batch_size]})
    x_adv_set.append(values[0])
    logits_set.append(values[1])
    p_set.append(values[2])
  x_adv_set = np.concatenate(x_adv_set)
  logits_set = np.concatenate(logits_set)
  p_set = np.concatenate(p_set)

  del x_set

  # now run test
  p_set, p_det = np.concatenate(
      [predictor.send(x_adv_set[b * batch_size:(b + 1) * batch_size]) for b in
       range(n_batches)]).T

  correctly_classified = np.equal(p_set,
                                  y_set[:len(p_set)].argmax(-1))

  adversarial_example_detected = np.equal(p_det, True)
  # model_fooled = np.logical_or(
  #    np.logical_and(~correctly_classified, ~adversarial_example_detected), # fooled classifier & evaded detector
  #    np.logical_and(correctly_classified, adversarial_example_detected) # did not fool classifier but triggered detector (false positive)
  # )
  model_fooled = np.logical_and(~correctly_classified,
                                ~adversarial_example_detected)  # fooled classifier & evaded detector

  correctly_classified_not_detected = np.logical_and(correctly_classified,
                                                     ~adversarial_example_detected)

  # print(len(adversarial_example_detected), np.sum(~correctly_classified),
  #      np.sum(adversarial_example_detected))

  # asr = model_fooled.mean()
  # acc = correctly_classified.mean()
  # print('Accuracy of base model: %0.4f' % acc)
  # print('ASR (w/ detection defense): %0.4f' % asr)

  return model_fooled, correctly_classified, adversarial_example_detected


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--debug', action='store_true')
  parser.add_argument('--multi-noise', action='store_true')
  parser.add_argument("--n-samples", default=512, type=int)
  parser.add_argument("--batch-size", default=512, type=int)
  parser.add_argument("--epsilon", default=8, type=int)
  parser.add_argument("--attack", choices=("clean", "original", "adaptive",
                                           "adaptive-eot"),
                      default="original")
  args = parser.parse_args()

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

  logits = cifar_model.get_logits(x_placeholder)

  # setup defense
  # if multi_noise = True, instantiate the defense with 9 types of noise.
  # if multi_noise = False, instantiate the defense with a single type of high-magnitude noise.
  print("multi noise:", args.multi_noise)
  defense_predictor = init_defense(sess, x_placeholder, logits, args.batch_size,
                                   multi_noise=args.multi_noise)

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

  test_loader = build_dataloader_from_arrays(x_batch, y_batch,
                                             batch_size=args.batch_size)

  original_pgd_params = {
      # ord: ,
      'eps': args.epsilon,
      'eps_iter': (args.epsilon / 5.0),
      'nb_iter': 10,
      'clip_min': 0,
      'clip_max': 255
  }
  adaptive_pgd_params = {
      # ord: ,
      'eps': args.epsilon,
      'eps_iter': args.epsilon / 100.0,
      'nb_iter': 100,
      'clip_min': 0,
      'clip_max': 255,
      'x_reference': x_reference_placeholder,
  }

  if args.attack == "clean":
    adv_x = x_placeholder
  else:
    if args.attack == "original":
      pgd = MadryEtAl(cifar_model, sess=sess)
      print("Using MadryEtAl attack")
    elif args.attack == "adaptive":
      pgd = ProjectedGradientDescentWithDetectorLogitMatching(
          cifar_model,
          lambda x: cifar_model.get_logits(x),
          sess=sess,
          verbose=False)
      print("Using logit-matching attack")
    elif args.attack == "adaptive-eot":
      pgd = ProjectedGradientDescentWithDetectorLogitMatching(
          cifar_model,
          lambda x: cifar_model.get_logits(x),
          sess=sess,
          eot_ensemble_size=20,
          verbose=False)
      print("Using logit-matching attack w/ EOT")
    else:
      raise ValueError("invalid attack")

    pgd_params = original_pgd_params if args.attack == "original" else adaptive_pgd_params
    adv_x = tf.stop_gradient(pgd.generate(x_placeholder, **pgd_params))

  adv_logits = cifar_model.get_logits(adv_x)
  adv_predictions = tf.argmax(adv_logits, 1)

  def run_eval(l):
    # should_be_rejected = ~verify_valid_input_data(kwargs["reference_points_x"])
    # print("should_be_rejected", should_be_rejected)

    is_advs = []
    correctly_classifieds = []
    adv_detecteds = []
    model_fooleds = []
    for x, y in l:
      x = x.numpy().transpose((0, 2, 3, 1)) * 255.0
      y = y.numpy()

      # pick targets. We'll keep it simple and just target the logits
      # of the first clean example, except for inputs that have the
      # same class as that example. For those, we target the logits
      # of the first clean example w/ different class.
      y_cls = np.argmax(y, -1)
      reference_x = x.copy()
      reference_x[:] = x[0]
      # get first element that has different class than first sample
      idx = np.argmax(y_cls != y_cls[0])
      reference_x[y_cls == y_cls[0]] = x[idx]

      print(x.shape, reference_x.shape)

      model_fooled_np, correctly_classified_np, adv_detected_np = do_eval(
          sess=sess, x=x_placeholder, x_adv=adv_x,
          batch_size=args.batch_size,
          logits=adv_logits,
          preds=adv_predictions, x_set=x, y_set=y,
          predictor=defense_predictor,
          attack_kwargs={x_reference_placeholder: reference_x}
      )

      # print(is_adv_np, y, logits_np)
      adv_detecteds.append(adv_detected_np)
      correctly_classifieds.append(correctly_classified_np)
      model_fooleds.append(model_fooled_np)
    adv_detecteds = np.concatenate(adv_detecteds)
    correctly_classifieds = np.concatenate(correctly_classifieds)
    model_fooleds = np.concatenate(model_fooleds)

    print("ASR:", np.mean(model_fooleds))
    print("correctly_classifieds", np.mean(correctly_classifieds),
          "adversarial detected", np.mean(adv_detecteds))

  run_eval(test_loader)


if __name__ == "__main__":
  main()
