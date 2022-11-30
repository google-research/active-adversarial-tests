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

import tensorflow as tf
import numpy as np

import cifar10_input

import config_attack

import sys
import math
from tqdm import tqdm

from case_studies.curriculum_at.PGD_attack import LinfPGDAttack

if __name__ == '__main__':
  config = vars(config_attack.get_args())

  tf.set_random_seed(config['tf_seed'])
  np.random.seed(config['np_seed'])

  model_file = tf.train.latest_checkpoint(config['model_dir'])
  # print("config['model_dir']: ", config['model_dir'])
  if model_file is None:
    print('No model found')
    sys.exit()

  if 'GTP' in config['model_dir']:
    from model_new import Model, ModelTinyImagnet

    if config['dataset'] == 'cifar10' or config['dataset'] == 'cifar100':
      # TODO: verify this with the authors
      # ATTENTION: mode was "train" before
      model = Model(mode=config["inference_mode"], dataset=config['dataset'],
                    train_batch_size=config['eval_batch_size'],
                    normalize_zero_mean=True)
    else:
      model = ModelTinyImagnet(mode='train', dataset=config['dataset'],
                               train_batch_size=config['eval_batch_size'],
                               normalize_zero_mean=True)

  elif 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config[
    'model_dir']:
    print("finetuned tinyimagenet MODEL")
    from model_new import ModelTinyImagenetSourceExtendedLogits

    full_source_model_x_input = tf.placeholder(tf.float32,
                                               shape=[None, 32, 32, 3])
    upresized_full_source_model_x_input = tf.image.resize_images(
      full_source_model_x_input, size=[64, 64])
    if config['dataset'] == 'cifar10':
      model = ModelTinyImagenetSourceExtendedLogits(mode='train',
                                                    dataset='tinyimagenet',
                                                    target_task_class_num=10,
                                                    train_batch_size=config[
                                                      'eval_batch_size'],
                                                    input_tensor=upresized_full_source_model_x_input)
    elif config['dataset'] == 'cifar100':
      model = ModelTinyImagenetSourceExtendedLogits(mode='train',
                                                    dataset='tinyimagenet',
                                                    target_task_class_num=100,
                                                    train_batch_size=config[
                                                      'eval_batch_size'],
                                                    input_tensor=upresized_full_source_model_x_input)

    model.x_input = full_source_model_x_input

    t_vars = tf.trainable_variables()
    source_model_vars = [var for var in t_vars if (
          'discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]
    source_model_target_logit_vars = [var for var in t_vars if
                                      'target_task_logit' in var.name]
    source_model_saver = tf.train.Saver(var_list=source_model_vars)
    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars
    finetuned_source_model_saver = tf.train.Saver(
      var_list=finetuned_source_model_vars)
  elif 'finetuned_on_cifar100' in config['model_dir']:
    raise NotImplementedError
    print("finetuned MODEL")
    from model_original_cifar_challenge import ModelExtendedLogits

    model = ModelExtendedLogits(mode='train', target_task_class_num=100,
                                train_batch_size=config['eval_batch_size'])

    t_vars = tf.trainable_variables()
    source_model_vars = [var for var in t_vars if (
          'discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]
    source_model_target_logit_vars = [var for var in t_vars if
                                      'target_task_logit' in var.name]
    source_model_saver = tf.train.Saver(var_list=source_model_vars)
    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars
    finetuned_source_model_saver = tf.train.Saver(
      var_list=finetuned_source_model_vars)
  elif ('adv_trained' in config['model_dir'] or 'naturally_trained' in config[
    'model_dir'] or 'a_very_robust_model' in config['model_dir']):
    raise NotImplementedError
    print("original challenge MODEL")
    from free_model_original import Model

    model = Model(mode='eval', dataset=config['dataset'],
                  train_batch_size=config['eval_batch_size'])
  elif 'IGAM' in config['model_dir']:
    print("IGAM MODEL")
    from model_new import Model

    model = Model(mode='train', dataset=config['dataset'],
                  train_batch_size=config['eval_batch_size'],
                  normalize_zero_mean=True)
  else:
    raise NotImplementedError
    print("other MODEL")
    from free_model import Model

    model = Model(mode='eval', dataset=config['dataset'],
                  train_batch_size=config['eval_batch_size'])

  attack = LinfPGDAttack(model,
                         config['epsilon'],
                         config['num_steps'],
                         config['step_size'],
                         config['random_start'],
                         config['loss_func'],
                         dataset=config['dataset'])
  saver = tf.train.Saver()

  data_path = config['data_path']
  # print(data_path)
  # x = input()

  if config['dataset'] == 'cifar10':
    # print("load cifar10 dataset")
    cifar = cifar10_input.CIFAR10Data(data_path)
  elif config['dataset'] == 'cifar100':
    raise NotImplementedError
    print("load cifar100 dataset")
    cifar = cifar100_input.CIFAR100Data(data_path)
  else:
    raise NotImplementedError
    print("load tinyimagenet dataset")
    cifar = tinyimagenet_input.TinyImagenetData()

  with tf.Session() as sess:
    # Restore the checkpoint
    if 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config[
      'model_dir']:
      sess.run(tf.global_variables_initializer())
      source_model_file = tf.train.latest_checkpoint(
        "models/model_AdvTrain-igamsource-IGAM-tinyimagenet_b16")
      source_model_saver.restore(sess, source_model_file)
      finetuned_source_model_file = tf.train.latest_checkpoint(
          config['model_dir'])
      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    elif 'finetuned_on_cifar100' in config['model_dir']:
      sess.run(tf.global_variables_initializer())
      source_model_file = tf.train.latest_checkpoint("models/adv_trained")
      source_model_saver.restore(sess, source_model_file)
      finetuned_source_model_file = tf.train.latest_checkpoint(
          config['model_dir'])
      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    else:
      saver.restore(sess, model_file)

    # Iterate over the samples batch-by-batch
    num_eval_examples = config['num_eval_examples']
    eval_batch_size = config['eval_batch_size']
    num_batches = int(math.ceil(num_eval_examples / eval_batch_size))

    x_adv = []  # adv accumulator
    x = []
    y = []
    y_p = []
    y_adv = []
    is_correct = []
    # print('Iterating over {} batches'.format(num_batches))

    for ibatch in tqdm(range(num_batches)):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = cifar.eval_data.xs[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      if config['attack_norm'] == 'inf':
        x_batch_adv = attack.perturb(x_batch, y_batch, sess)
      elif config['attack_norm'] == '2':
        x_batch_adv = attack.perturb_l2(x_batch, y_batch, sess)
      elif config['attack_norm'] == 'TRADES':
        x_batch_adv = attack.perturb_TRADES(x_batch, y_batch, sess)
      elif config['attack_norm'] == "":
        x_batch_adv = x_batch

      y_pred = sess.run(model.predictions, feed_dict={model.x_input: x_batch_adv})

      y_pred_clean = sess.run(model.predictions, feed_dict={model.x_input: x_batch})

      x_adv.append(x_batch_adv)
      x.append(x_batch)
      y.append(y_batch)
      y_p.append(y_pred_clean)
      y_adv.append(y_pred)

      is_correct.append(y_pred == y_batch)
    is_correct = np.concatenate(is_correct)
    x_adv = np.concatenate(x_adv)
    x = np.concatenate(x)
    y = np.concatenate(y)
    y_p = np.concatenate(y_p)
    y_adv = np.concatenate(y_adv)
    if config["save_data_path"] is not None:
      x = x.astype(int)
      x_adv = x_adv.astype(int)
      np.savez(config["save_data_path"], x_a=x, x_b=x_adv, y_a=y_p, y_b=y_adv)
    print(f"Robust accuracy: {np.mean(is_correct)}")


