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
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


import time
import sys

import numpy as np

from bat_utils import *
from wide_resnet import Model


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
  logits, _ = model.build_model(images=x_transformed, num_classes=FLAGS.num_classes)

  prob = tf.nn.softmax(logits)
  correct = tf.cast(tf.equal(tf.argmax(logits, axis=1), y_pl), tf.float32)
  accuracy = tf.reduce_mean(correct)

  saver = tf.train.Saver(max_to_keep=100)

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True
  sess = tf.Session(config=config)

  saver.restore(sess, FLAGS.ckpt_path)
  print('restored checkpoint from %s' % FLAGS.ckpt_path)

  return sess, (x_pl, y_pl, is_train, logits, accuracy)


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
    logits, grad_val = sess.run(grad,
                        feed_dict={
                            x_pl: x,
                            y_pl: y_att,
                            is_train: False
                        })

    x = x - step_size * np.sign(grad_val)

    x = np.clip(x, x_batch - epsilon, x_batch + epsilon)

    x = np.clip(x, 0, 1.0)

  return x


def run_eval(sess, grad, x_pl, y_pl, is_train, logits, FLAGS,
    test_images, test_labels, num_classes=10):
  test_size = test_images.shape[0]
  epoch_steps = np.ceil(test_size / FLAGS.batch_size).astype(np.int32)
  nat_total = 0.0
  adv_total = 0.0
  class_nat_total = np.zeros([num_classes])
  class_adv_total = np.zeros([num_classes])
  nat_cnt_list = np.zeros([test_size])
  adv_cnt_list = np.zeros([test_size])
  idx = np.random.permutation(test_size)
  for step_idx in range(epoch_steps):
    start = step_idx * FLAGS.batch_size
    end = np.minimum((step_idx + 1) * FLAGS.batch_size,
                     test_size).astype(np.int32)
    x_batch = test_images[idx[start:end]]
    y_batch = test_labels[idx[start:end]]

    start_time = time.time()

    nat_logits = sess.run(logits,
                          feed_dict={
                              x_pl: x_batch,
                              is_train: False
                          })
    nat_cnt = nat_logits.argmax(-1) == y_batch

    x_batch_adv = adv_attack(sess, grad, x_pl, y_pl, is_train, x_batch, y_batch, FLAGS)

    adv_logits = sess.run(logits,
                          feed_dict={
                              x_pl: x_batch_adv,
                              y_pl: y_batch,
                              is_train: False
                          })
    adv_cnt = adv_logits.argmax(-1) == y_batch

    nat_cnt_list[start:end] = nat_cnt
    adv_cnt_list[start:end] = adv_cnt

    for ii in range(FLAGS.num_classes):
      class_nat_total[ii] += np.sum(nat_cnt[y_batch == ii])
      class_adv_total[ii] += np.sum(adv_cnt[y_batch == ii])

    nat_total += np.sum(nat_cnt)
    adv_total += np.sum(adv_cnt)

    duration = time.time() - start_time
    print('finished batch %d/%d, duration %.2f, nat acc %.2f, adv acc %.2f' %
          (step_idx, epoch_steps, duration, 100 * np.mean(nat_cnt),
           100 * np.mean(adv_cnt)))
    sys.stdout.flush()

  nat_acc = nat_total / test_size
  adv_acc = adv_total / test_size
  class_nat_total /= (test_size / FLAGS.num_classes)
  class_adv_total /= (test_size / FLAGS.num_classes)
  print('clean accuracy: %.2f, adv accuracy: %.2f' %
        (100 * nat_acc, 100 * adv_acc))
  for ii in range(FLAGS.num_classes):
    print('class %d, clean %.2f, adv %.2f' %
          (ii, 100 * class_nat_total[ii], 100 * class_adv_total[ii]))


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

  FLAGS = tf.flags.FLAGS

  print(FLAGS.flag_values_dict())

  return FLAGS


def main():
  FLAGS = parse_args()
  _, _, test_images, test_labels = load_data(FLAGS)

  # normalize to [0,1] value range from [-1, 1] since we put normalization
  # inside of the model
  test_images = (test_images + 1.0) / 2.0

  # subsample test data
  if FLAGS.n_samples == -1:
    FLAGS.n_samples = len(test_images)
  idxs = np.arange(len(test_images))
  np.random.shuffle(idxs)
  idxs = idxs[:FLAGS.n_samples]

  test_images, test_labels = test_images[idxs], test_labels[idxs]

  sess, (x_pl, y_pl, is_train, logits, accuracy) = load_model(FLAGS)
  attack_grad = setup_attack(logits, x_pl, y_pl, FLAGS)
  run_eval(sess, attack_grad, x_pl, y_pl, is_train, logits, FLAGS,
           test_images, test_labels)


if __name__ == "__main__":
  main()
