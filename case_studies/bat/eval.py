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

import time
import os
import sys

import numpy as np
import tensorflow as tf

from utils import *
from wide_resnet import Model

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

tf.flags.DEFINE_string('cuda_device', '3', '')

FLAGS = tf.flags.FLAGS

os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.cuda_device

print(FLAGS.flag_values_dict())

# load data
if FLAGS.dataset == 'SVHN':
    train_data = np.load('svhn_data/train_32x32.npz')
    train_images = train_data['arr_0']
    train_images = 2.0 * (train_images / 255.0 - 0.5)
    train_labels = train_data['arr_1']
    test_data = np.load('svhn_data/test_32x32.npz')
    test_images = test_data['arr_0']
    test_images = 2 * (test_images / 255.0 - 0.5)
    test_labels = test_data['arr_1']
elif FLAGS.dataset == 'CIFAR':
    if FLAGS.num_classes == 10:
        dataset = load_cifar10_data('cifar10_data')
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

x_pl = tf.placeholder(tf.float32, shape=[None, 32, 32, 3], name='x')
y_pl = tf.placeholder(tf.int64, shape=[None], name='y')
is_train = tf.placeholder(tf.bool, name='is_train')

model = Model(is_train)
logits, _ = model.build_model(images=x_pl, num_classes=FLAGS.num_classes)
prob = tf.nn.softmax(logits)
xent = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits,
                                                      labels=y_pl)

correct = tf.cast(tf.equal(tf.argmax(logits, axis=1), y_pl), tf.float32)
accuracy = tf.reduce_mean(correct)

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
        loss_att = tf.reduce_sum(tf.maximum(other - self + FLAGS.margin, 0))
    else:
        loss_att = tf.reduce_sum(tf.maximum(self - other + FLAGS.margin, 0))
else:
    raise ValueError('loss type not supported!')

grad, = tf.gradients(loss_att, x_pl)

saver = tf.train.Saver(max_to_keep=100)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

saver.restore(sess, FLAGS.ckpt_path)
print('restored checkpoint from %s' % FLAGS.ckpt_path)


def adv_attack(nat_prob, x_batch, y_batch):
    epsilon = FLAGS.epsilon / 255.0 * 2
    step_size = FLAGS.step_size / 255.0 * 2

    if not FLAGS.targeted:
        y_att = np.copy(y_batch)
    else:
        # most confusing targeted attack
        nat_prob[np.arange(y_batch.shape[0]), y_batch] = 0.0
        y_att = np.argmax(nat_prob, axis=1)

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

        x = np.clip(x, -1.0, 1.0)

    return x


test_size = test_images.shape[0]
epoch_steps = np.ceil(test_size / FLAGS.batch_size).astype(np.int32)
nat_total = 0.0
adv_total = 0.0
class_nat_total = np.zeros([FLAGS.num_classes])
class_adv_total = np.zeros([FLAGS.num_classes])
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

    nat_cnt, nat_prob = sess.run([correct, prob],
                                 feed_dict={
                                     x_pl: x_batch,
                                     y_pl: y_batch,
                                     is_train: False
                                 })

    x_batch_adv = adv_attack(nat_prob, x_batch, y_batch)

    adv_cnt = sess.run(correct,
                       feed_dict={
                           x_pl: x_batch_adv,
                           y_pl: y_batch,
                           is_train: False
                       })

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
