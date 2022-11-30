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

tf.flags.DEFINE_string('model_dir', '/tmp/adv_train/', '')
tf.flags.DEFINE_string('dataset', '', '')
tf.flags.DEFINE_integer('num_classes', 10, '')

tf.flags.DEFINE_string('restore_ckpt_path', '', '')
tf.flags.DEFINE_integer('start_epoch', 0, '')
tf.flags.DEFINE_integer('max_epoch', 201, '')
tf.flags.DEFINE_integer('decay_epoch1', 100, '')
tf.flags.DEFINE_integer('decay_epoch2', 150, '')
tf.flags.DEFINE_float('decay_rate', 0.1, '')
tf.flags.DEFINE_float('learning_rate', 0.1, '')
tf.flags.DEFINE_float('momentum', 0.9, '')
tf.flags.DEFINE_integer('batch_size', 128, '')
tf.flags.DEFINE_float('weight_decay', 2e-4, '')

tf.flags.DEFINE_float('margin', 50.0, '')
tf.flags.DEFINE_string('loss_type', 'xent', '')
tf.flags.DEFINE_float('epsilon', 8.0, '')
tf.flags.DEFINE_integer('num_steps', 7, '')
tf.flags.DEFINE_float('step_size', 2.0, '')
tf.flags.DEFINE_boolean('random_start', True, '')
tf.flags.DEFINE_boolean('targeted', True, '')
tf.flags.DEFINE_string('target_type', 'MC', '')

tf.flags.DEFINE_boolean('label_adversary', True, '')
tf.flags.DEFINE_float('multi', 9, '')

tf.flags.DEFINE_integer('log_steps', 10, '')
tf.flags.DEFINE_integer('save_epochs', 20, '')
tf.flags.DEFINE_integer('eval_epochs', 10, '')

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
y_loss = tf.placeholder(tf.float32,
                        shape=[None, FLAGS.num_classes],
                        name='y_loss')
lr = tf.placeholder(tf.float32, name='lr')
is_train = tf.placeholder(tf.bool, name='is_train')
global_step = tf.Variable(0, trainable=False, name='global_step')

model = Model(is_train)
logits, _ = model.build_model(images=x_pl, num_classes=FLAGS.num_classes)
prob = tf.nn.softmax(logits)
xent = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_loss)
mean_xent = tf.reduce_mean(xent)
total_loss = mean_xent + FLAGS.weight_decay * model.weight_decay_loss

correct = tf.cast(tf.equal(tf.argmax(logits, axis=1), y_pl), tf.float32)
accuracy = tf.reduce_mean(correct)

# loss for adversarial attack
if FLAGS.loss_type == 'xent':
    if FLAGS.targeted:
        loss_att = tf.reduce_sum(xent)
    else:
        loss_att = -tf.reduce_sum(xent)
elif FLAGS.loss_type == 'CW':
    self = tf.reduce_sum(y_loss * logits, axis=1)
    other = tf.reduce_max((1 - y_loss) * logits - y_loss * 1000.0, axis=1)
    if FLAGS.targeted:
        loss_att = tf.reduce_sum(tf.maximum(other - self + FLAGS.margin, 0))
    else:
        loss_att = tf.reduce_sum(tf.maximum(self - other + FLAGS.margin, 0))
else:
    raise ValueError('loss type not supported!')

grad, = tf.gradients(loss_att, x_pl)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
opt = tf.train.MomentumOptimizer(lr, FLAGS.momentum)
grads_and_vars = opt.compute_gradients(total_loss, tf.trainable_variables())
with tf.control_dependencies(update_ops):
    train_step = opt.apply_gradients(grads_and_vars, global_step=global_step)

saver = tf.train.Saver(max_to_keep=100)

init_op = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(init_op)

if FLAGS.restore_ckpt_path:
    saver.restore(sess, os.path.abspath(FLAGS.restore_ckpt_path))
    print('Restored checkpoints from %s' % FLAGS.restore_ckpt_path)


def adv_attack(nat_logits,
               x_batch,
               y_batch,
               epsilon=FLAGS.epsilon,
               step_size=FLAGS.step_size,
               num_steps=FLAGS.num_steps):
    epsilon = epsilon / 255.0 * 2.0
    step_size = step_size / 255.0 * 2.0
    y_batch_dense = one_hot(y_batch, FLAGS.num_classes)
    if not FLAGS.targeted:  # non-targeted
        y_att = np.copy(y_batch)
    elif FLAGS.target_type == 'MC':  # most confusing target label
        nat_logits[np.arange(y_batch.shape[0]), y_batch] = -1e4
        y_att = np.argmax(nat_logits, axis=1)
    elif FLAGS.target_type == 'RAND':  # random target label
        y_att = np.zeros_like(y_batch)
        for ii in np.arange(y_batch.shape[0]):
            tmp = np.ones([FLAGS.num_classes]) / (FLAGS.num_classes - 1)
            tmp[y_batch[ii]] = 0.0
            y_att[ii] = np.random.choice(FLAGS.num_classes, p=tmp)
    elif FLAGS.target_type == 'MOSA':  # most one-step adversarial one-step target label
        weight = sess.run(tf.get_default_graph().get_tensor_by_name(
            'logit/DW:0')).T  # num_classes * num_features
        dist = euclidean_distances(
            weight[y_batch],
            weight) + y_batch_dense  # batch_size * num_classes
        gt_logits = np.sum(nat_logits * y_batch_dense, axis=1)
        diff_logits = np.reshape(gt_logits, [-1, 1]) - nat_logits
        truncated = np.where(diff_logits > 1e-4, diff_logits, 1e4)
        y_att = np.argmin(truncated / dist, axis=1)
    elif FLAGS.target_type == 'MIX':  # mix of MC and MOSA
        weight = sess.run(tf.get_default_graph().get_tensor_by_name(
            'logit/DW:0')).T  # num_classes * num_features
        dist = euclidean_distances(
            weight[y_batch],
            weight) + y_batch_dense  # batch_size * num_classes
        gt_logits = np.sum(nat_logits * y_batch_dense, axis=1)
        diff_logits = np.reshape(gt_logits, [-1, 1]) - nat_logits
        truncated = np.where(diff_logits > 1e-4, diff_logits, 1e4)
        y_att_MOSA = np.argmin(truncated / dist, axis=1)
        y_att_MC = np.argmax((1.0 - y_batch_dense) * nat_logits, axis=1)
        y_att = np.where(
            np.argmax(nat_logits, axis=1) == y_batch, y_att_MOSA, y_att_MC)
    else:
        raise ValueError('Target type not valid!')
    y_att_dense = one_hot(y_att, FLAGS.num_classes)

    # randomly perturb as initialization
    if FLAGS.random_start:
        noise = np.random.uniform(-epsilon, epsilon, x_batch.shape)
        x = x_batch + noise
    else:
        x = np.copy(x_batch)

    for i in range(num_steps):
        grad_val = sess.run(grad,
                            feed_dict={
                                x_pl: x,
                                y_loss: y_att_dense,
                                is_train: False
                            })

        x = x - step_size * np.sign(grad_val)

        x = np.clip(x, x_batch - epsilon, x_batch + epsilon)

        x = np.clip(x, -1.0, 1.0)

    return x


def adv_labels(nat_prob, y_batch, gamma=0.01):
    L = -np.log(nat_prob + 1e-8)  # log-likelihood
    LL = np.copy(L)
    LL[np.arange(y_batch.shape[0]), y_batch] = 1e4
    minval = np.min(LL, axis=1)
    LL[np.arange(y_batch.shape[0]), y_batch] = -1e4
    maxval = np.max(LL, axis=1)

    denom = np.sum(L, axis=1) - L[np.arange(y_batch.shape[0]), y_batch] - (
        FLAGS.num_classes - 1) * (minval - gamma)
    delta = 1 / (1 + FLAGS.multi * (maxval - minval + gamma) / denom)
    alpha = delta / denom

    y_batch_adv = np.reshape(
        alpha, [-1, 1]) * (L - np.reshape(minval, [-1, 1]) + gamma)
    y_batch_adv[np.arange(y_batch.shape[0]), y_batch] = 1.0 - delta

    return y_batch_adv


# training loop
train_size = train_images.shape[0]
epoch_steps = np.ceil(train_size / FLAGS.batch_size).astype(np.int32)
for epoch_idx in np.arange(FLAGS.start_epoch, FLAGS.max_epoch):
    if epoch_idx < FLAGS.decay_epoch1:
        lr_val = FLAGS.learning_rate
    elif epoch_idx < FLAGS.decay_epoch2:
        lr_val = FLAGS.learning_rate * FLAGS.decay_rate
    else:
        lr_val = FLAGS.learning_rate * FLAGS.decay_rate * FLAGS.decay_rate

    # each epoch random shuffle of training images
    idx = np.random.permutation(train_size)
    for step_idx in np.arange(epoch_steps):
        start = step_idx * FLAGS.batch_size
        end = np.minimum((step_idx + 1) * FLAGS.batch_size,
                         train_size).astype(np.int32)
        x_batch = preprocess_py(train_images[idx[start:end]], 4, 32)
        y_batch = train_labels[idx[start:end]]
        y_batch_dense = one_hot(y_batch, FLAGS.num_classes)

        start_time = time.time()

        nat_prob, nat_logits = sess.run([prob, logits],
                                        feed_dict={
                                            x_pl: x_batch,
                                            is_train: False
                                        })

        # generate adversarial images
        x_batch_adv = adv_attack(nat_logits, x_batch, y_batch)

        # generate adversarial labels
        y_batch_adv = adv_labels(nat_prob, y_batch)

        # eval accuracy
        if step_idx % FLAGS.log_steps == 0:
            nat_acc = sess.run(accuracy,
                               feed_dict={
                                   x_pl: x_batch,
                                   y_pl: y_batch,
                                   is_train: False
                               })
            adv_acc = sess.run(accuracy,
                               feed_dict={
                                   x_pl: x_batch_adv,
                                   y_pl: y_batch,
                                   is_train: False
                               })

        # training step
        if FLAGS.label_adversary:
            _, loss_val = sess.run([train_step, total_loss],
                                   feed_dict={
                                       x_pl: x_batch_adv,
                                       y_loss: y_batch_adv,
                                       is_train: True,
                                       lr: lr_val
                                   })
        else:
            _, loss_val = sess.run(
                [train_step, total_loss],
                feed_dict={
                    x_pl: x_batch_adv,
                    y_loss: y_batch_dense,
                    is_train: True,
                    lr: lr_val
                })

        duration = time.time() - start_time

        # print to stdout
        if step_idx % FLAGS.log_steps == 0:
            print(
                "epoch %d, step %d, lr %.4f, duration %.2f, training nat acc %.2f, training adv acc %.2f, training adv loss %.4f"
                % (epoch_idx, step_idx, lr_val, duration, 100 * nat_acc,
                   100 * adv_acc, loss_val))
            sys.stdout.flush()

    # save checkpoint
    if epoch_idx % FLAGS.save_epochs == 0:
        saver.save(sess, os.path.join(FLAGS.model_dir, 'checkpoint'),
                   epoch_idx)

    # evaluate
    def eval_once():
        eval_size = test_images.shape[0]
        epoch_steps = np.ceil(eval_size / FLAGS.batch_size).astype(np.int32)
        # random shuffle of test images does not affect the result
        idx = np.random.permutation(eval_size)
        count = 0.0
        for step_idx in np.arange(epoch_steps):
            start = step_idx * FLAGS.batch_size
            end = np.minimum((step_idx + 1) * FLAGS.batch_size,
                             eval_size).astype(np.int32)
            x_batch = test_images[idx[start:end]]
            y_batch = test_labels[idx[start:end]]
            nat_logits = sess.run(logits,
                                  feed_dict={
                                      x_pl: x_batch,
                                      is_train: False
                                  })
            x_batch_adv = adv_attack(nat_logits, x_batch, y_batch)
            count += np.sum(
                sess.run(correct,
                         feed_dict={
                             x_pl: x_batch_adv,
                             y_pl: y_batch,
                             is_train: False
                         }))
        acc = count / eval_size
        return acc

    if epoch_idx % FLAGS.eval_epochs == 0:
        print('epoch %d, adv acc %.2f' % (epoch_idx, 100 * eval_once()))
