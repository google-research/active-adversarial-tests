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

import logging
from functools import partial
from typing import Tuple

import torch

from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper

logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import time
import sys

import numpy as np

from bat_utils import *
from wide_resnet import Model

from utils import build_dataloader_from_arrays
from active_tests import decision_boundary_binarization as dbb
from argparse_utils import DecisionBoundaryBinarizationSettings


def load_data(FLAGS):
  # load data
  if FLAGS.dataset == 'SVHN':
    raise ValueError("not supported")
  elif FLAGS.dataset == 'CIFAR':
    if FLAGS.num_classes == 10:
      dataset = load_cifar10_data('data/cifar-10-batches-py/')
    elif FLAGS.num_classes == 20:
      dataset = load_cifar100_data('cifar100_data', is_fine=False)
    elif FLAGS.num_classes == 100:
      dataset = load_cifar100_data('cifar100_data', is_fine=True)
    else:
      raise ValueError('Number of classes not valid!')
    train_images = dataset['train_images']
    train_labels = dataset['train_labels']
    test_images = dataset['test_images']
    test_labels = dataset['test_labels']
  else:
    raise ValueError('Dataset not valid!')

  return train_images, train_labels, test_images, test_labels


def load_model(FLAGS):
  x_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
  y_pl = tf.placeholder(tf.int64, shape=[None], name='y')
  is_train = tf.placeholder(tf.bool, name='is_train')

  model = Model(is_train)
  x_transformed = x_pl * 2.0 - 1.0
  fe_logits, features = model.build_model(images=x_transformed,
                                  num_classes=FLAGS.num_classes)

  saver = tf.train.Saver(max_to_keep=100)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  saver.restore(sess, FLAGS.ckpt_path)
  print('restored checkpoint from %s' % FLAGS.ckpt_path)

  # create binary classifier
  bro_w = tf.get_variable(
      'DW', [features.shape[-1], 2],
      initializer=tf.uniform_unit_scaling_initializer(factor=1.0))
  bro_b = tf.get_variable('biases', [2], initializer=tf.constant_initializer())
  bro_w_pl = tf.placeholder(tf.float32, shape=[features.shape[-1], 2])
  bro_b_pl = tf.placeholder(tf.float32, shape=[2])
  bro_w_set_weight = bro_w.assign(bro_w_pl)
  bro_b_set_weight = bro_b.assign(bro_b_pl)
  logits = tf.nn.xw_plus_b(features, bro_w, bro_b)

  prob = tf.nn.softmax(logits)
  correct = tf.cast(tf.equal(tf.argmax(logits, axis=1), y_pl), tf.float32)
  accuracy = tf.reduce_mean(correct)

  return sess, (x_pl, y_pl, is_train, logits, fe_logits, features, accuracy), \
         (bro_w_pl, bro_b_pl, bro_w_set_weight, bro_b_set_weight)


def setup_attack(logits, x_pl, y_pl, FLAGS):
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                        labels=y_pl)
  # loss for adversarial attack
  if FLAGS.loss_type == 'xent':
    if FLAGS.targeted:
      loss_att = tf.reduce_sum(xent)
    else:
      loss_att = -tf.reduce_sum(xent)
  elif FLAGS.loss_type == 'CW':
    y_loss = tf.one_hot(y_pl, FLAGS.num_classes)
    self = tf.reduce_sum(y_loss * logits, axis=1)
    other = tf.reduce_max((1 - y_loss) * logits - y_loss * 1e4, axis=1)
    if FLAGS.targeted:
      raise ValueError("not supported")
    else:
      loss_att = tf.reduce_sum(tf.maximum(self - other + FLAGS.margin, 0))
  else:
    raise ValueError('loss type not supported!')

  grad, = tf.gradients(loss_att, x_pl)

  return grad


def adv_attack(sess, grad, x_pl, y_pl, is_train, x_batch, y_batch, FLAGS):
  epsilon = FLAGS.epsilon / 255.0
  step_size = FLAGS.step_size / 255.0

  if not FLAGS.targeted:
    y_att = np.copy(y_batch)
  else:
    raise ValueError("targeted mode not supported")

  # randomly perturb the original images
  if FLAGS.random_start:
    x = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
  else:
    x = np.copy(x_batch)

  for i in range(FLAGS.num_steps):
    grad_val = sess.run(grad,
                        feed_dict={
                            x_pl: x,
                            y_pl: y_att,
                            is_train: False
                        })

    x = x - step_size * np.sign(grad_val)
    x = np.clip(x, x_batch - epsilon, x_batch + epsilon)
    x = np.clip(x, 0, 1.0)

  return x


def parse_args():
  tf.flags.DEFINE_string('ckpt_path', '', '')
  tf.flags.DEFINE_string('dataset', 'CIFAR', '')
  tf.flags.DEFINE_integer('num_classes', 10, '')
  tf.flags.DEFINE_integer('batch_size', 100, '')
  tf.flags.DEFINE_string('loss_type', 'xent', '')
  tf.flags.DEFINE_float('margin', 50.0, '')
  tf.flags.DEFINE_float('epsilon', 8.0, '')
  tf.flags.DEFINE_integer('num_steps', 10, '')
  tf.flags.DEFINE_float('step_size', 2.0, '')
  tf.flags.DEFINE_boolean('random_start', False, '')
  tf.flags.DEFINE_boolean('targeted', False, '')
  tf.flags.DEFINE_integer('n_samples', 2048, '')

  tf.flags.DEFINE_integer('n_boundary_points', 1, '')
  tf.flags.DEFINE_integer('n_inner_points', 999, '')
  tf.flags.DEFINE_boolean('sample_from_corners', False, '')

  FLAGS = tf.flags.FLAGS

  # print(FLAGS.flag_values_dict())

  return FLAGS


def run_attack(m, l, sess, logits, x_pl, is_train, bro_w_pl, bro_b_pl,
    bro_w_assign, bro_b_assign, attack_fn):
  linear_layer = m[-1]
  del m

  sess.run(bro_w_assign, {bro_w_pl: linear_layer.weight.data.numpy().T})
  sess.run(bro_b_assign, {bro_b_pl: linear_layer.bias.data.numpy()})

  for x, y in l:
    x, y = x.numpy(), y.numpy()
    x = x.transpose((0, 2, 3, 1))
    x_batch_adv = attack_fn(x, y)

    adv_logits: np.ndarray = sess.run(logits,
                          feed_dict={
                              x_pl: x_batch_adv,
                              is_train: False
                          })
    is_adv: np.ndarray = adv_logits.argmax(-1) != y

    return is_adv, (torch.tensor(x_batch_adv.transpose((0, 3, 1, 2))),
                    torch.tensor(adv_logits))


def main():
  FLAGS = parse_args()
  _, _, test_images, test_labels = load_data(FLAGS)

  print(FLAGS.flag_values_dict())

  # normalize to [0,1] value range from [-1, 1] since we put normalization
  # inside of the model
  test_images = (test_images + 1.0) / 2.0

  # subsample test data
  if FLAGS.n_samples == -1:
    FLAGS.n_samples = len(test_images)
  idxs = np.arange(len(test_images))
  np.random.shuffle(idxs)
  test_images, test_labels = test_images[idxs], test_labels[idxs]
  test_images = test_images.transpose((0, 3, 1, 2))

  test_loader = build_dataloader_from_arrays(test_images, test_labels,
                                             FLAGS.batch_size)

  sess, (x_pl, y_pl, is_train, logits, fe_logits, features, accuracy), \
  (bro_w_pl, bro_b_pl, bro_w_set_weight, bro_b_set_weight) = load_model(FLAGS)
  attack_grad = setup_attack(logits, x_pl, y_pl, FLAGS)
  attack_fn = lambda x, y: adv_attack(sess, attack_grad, x_pl, y_pl, is_train,
                                      x, y, FLAGS)

  def feature_extractor_forward_pass(x, features_and_logits: bool = False,
      features_only: bool = False):
    if features_and_logits:
      assert not features_only, "Only one of the flags must be set."
    if features_and_logits:
      f, l = sess.run(
          (features, fe_logits),
          feed_dict={x_pl: x.transpose(0, 2, 3, 1), is_train: False})
      return f, l
    elif features_only:
      return sess.run(
          features,
          feed_dict={x_pl: x.transpose(0, 2, 3, 1), is_train: False})
    else:
      return sess.run(
          fe_logits,
          feed_dict={x_pl: x.transpose(0, 2, 3, 1), is_train: False})

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=feature_extractor_forward_pass,
      logit_forward_and_backward_pass=None,
  )

  scores_logit_differences_and_validation_accuracies = \
    dbb.interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        # m, l, sess, logits, x_pl, is_train, bro_w_pl, bro_b_pl,
        #     bro_w_assign, bro_b_assign, attack_fn)
        attack_fn=lambda m, l, kwargs: partial(run_attack,
            sess=sess, logits=logits, x_pl=x_pl, is_train=is_train,
            bro_w_pl=bro_w_pl, bro_b_pl=bro_b_pl, bro_w_assign=bro_w_set_weight,
            bro_b_assign=bro_b_set_weight,
            attack_fn=attack_fn)(m, l),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=FLAGS.epsilon / 255.0,
            norm="linf",
            lr=10000,
            n_boundary_points=FLAGS.n_boundary_points,
            n_inner_points=FLAGS.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=FLAGS.n_samples,
        device="cpu",
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        # rescale_logits="adaptive",
        sample_training_data_from_corners=FLAGS.sample_from_corners,
        decision_boundary_closeness=0.9999,
        # args.num_samples_test * 10
    )

  print(dbb.format_result(scores_logit_differences_and_validation_accuracies,
                          FLAGS.n_samples))

if __name__ == "__main__":
  main()
