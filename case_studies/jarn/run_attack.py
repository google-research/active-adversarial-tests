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

"""Evaluates a model against examples from a .npy file as specified
   in attack_config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import math
import os
import sys
import time

import tensorflow as tf
import numpy as np
from tqdm import tqdm

# from model import Model
import cifar10_input
# import cifar100_input

import config_attack

# with open('attack_config.json') as config_file:
#     config = json.load(config_file)

config = vars(config_attack.get_args())

# if config['model_dir'] in ["models/adv_trained", "models/naturally_trained"]:
#   from free_model_original import Model
# elif 'DefPert2' in config['model_dir']:
#   from model_jarn import ModelDefPert as Model
# elif 'JARN':
#   from model_jarn import Model
# else:
#   from free_model import Model

data_path = config['data_path']

def run_attack(checkpoint, x_adv, epsilon):
#   cifar = cifar10_input.CIFAR10Data(data_path)
  cifar = cifar10_input.CIFAR10Data(data_path)
  # if config['dataset'] == 'cifar10':
  #   cifar = cifar10_input.CIFAR10Data(data_path)
  # else:
  #   cifar = cifar100_input.CIFAR100Data(data_path)

  
  print("JARN MODEL")
  from model_jarn import Model
  if "_zeromeaninput" in config['model_dir']:
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)
  else:
    model = Model(dataset=config['dataset'], train_batch_size=config['eval_batch_size'])

  saver = tf.train.Saver()

  num_eval_examples = 10000
  eval_batch_size = 100

  num_batches = int(math.ceil(num_eval_examples / eval_batch_size))
  total_corr = 0

  x_nat = cifar.eval_data.xs
  l_inf = np.amax(np.abs(x_nat - x_adv))

  if l_inf > epsilon + 0.0001:
    print('maximum perturbation found: {}'.format(l_inf))
    print('maximum perturbation allowed: {}'.format(epsilon))
    return

  y_pred = [] # label accumulator

  with tf.Session() as sess:    
    # Restore the checkpoint
    saver.restore(sess, checkpoint)
    # if 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:
    #   sess.run(tf.global_variables_initializer())
    #   source_model_file = tf.train.latest_checkpoint("models/model_AdvTrain-jrtsource-JRT-tinyimagenet_b16")
    #   source_model_saver.restore(sess, source_model_file)   
    #   finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])
    #   finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    # elif 'mnist_adv_trained_finetuned_on_cifar10_bwtransform' in config['model_dir']:
    #   sess.run(tf.global_variables_initializer())
    #   source_model_file = tf.train.latest_checkpoint("models/mnist_adv_trained")
    #   source_model_saver.restore(sess, source_model_file)   
    #   finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])
    #   finetuned_source_model_saver.restore(sess, finetuned_source_model_file)
    # elif 'finetuned_on_cifar100' in config['model_dir']:
    #   sess.run(tf.global_variables_initializer())
    #   source_model_file = tf.train.latest_checkpoint("models/adv_trained")
    #   source_model_saver.restore(sess, source_model_file)   
    #   finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])
    #   finetuned_source_model_saver.restore(sess, finetuned_source_model_file)

    #   # sess.run(tf.global_variables_initializer())
    #   # source_model_file = tf.train.latest_checkpoint("models/adv_trained")
    #   # source_model_saver.restore(sess, source_model_file)   
    #   # finetuned_source_model_file = tf.train.latest_checkpoint("models/adv_trained_finetuned_on_cifar100_b32_20ep")
    #   # finetuned_source_model_saver.restore(sess, finetuned_source_model_file) 
    # else:
    #   saver.restore(sess, checkpoint)

    # Iterate over the samples batch-by-batch
    for ibatch in range(num_batches):
      bstart = ibatch * eval_batch_size
      bend = min(bstart + eval_batch_size, num_eval_examples)

      x_batch = x_adv[bstart:bend, :]
      y_batch = cifar.eval_data.ys[bstart:bend]

      dict_adv = {model.x_input: x_batch,
                  model.y_input: y_batch}
      
      cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
                                        feed_dict=dict_adv)
      # if 'finetuned_on_cifar10' in config['model_dir'] or 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:
      #   cur_corr, y_pred_batch = sess.run([model.target_task_num_correct, model.target_task_predictions],
      #                                     feed_dict=dict_adv)
      # else:
      #   cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],
      #                                     feed_dict=dict_adv)

      total_corr += cur_corr
      y_pred.append(y_pred_batch)

    accuracy = total_corr / num_eval_examples

    print('Adv Accuracy: {:.2f}%'.format(100.0 * accuracy))
    y_pred = np.concatenate(y_pred, axis=0)

    store_adv_pred_path = "preds/" + adv_examples_path.split("/")[-1]
    if not os.path.exists("preds/"):
      os.makedirs("preds/")
    np.save(store_adv_pred_path, y_pred)
    print('Output saved at ', store_adv_pred_path)

    if config['save_eval_log']:
      date_str = datetime.now().strftime("%d_%b")
      log_dir = "attack_log/" + date_str
      if not os.path.exists(log_dir):
        os.makedirs(log_dir)
      log_filename = adv_examples_path.split("/")[-1].replace('.npy', '.txt')
      model_name = config['model_dir'].split('/')[1]
      # if model_name not in log_filename or config['xfer_attack']:
      #   print("Transfer Attack!")
      #   if config['custom_output_model_name'] is not None:
      #     new_log_filename = config['custom_output_model_name'] +"-xferattacked_by-"+ log_filename
      #   else:
      #     new_log_filename = model_name +"-xferattacked_by-"+ log_filename
      #   log_filename = new_log_filename
      log_file_path = os.path.join(log_dir, log_filename)
      with open(log_file_path, "w") as f:
        f.write('Model checkpoint: {} \n'.format(checkpoint))
        f.write('Adv Accuracy: {:.2f}%'.format(100.0 * accuracy))
      print('Results saved at ', log_file_path)

      # full test evaluation
    #   raw_data = cifar10_input.CIFAR10Data(data_path)
      if config['dataset'] == 'cifar10':
        raw_data = cifar10_input.CIFAR10Data(data_path)
      else:
        raw_data = cifar100_input.CIFAR100Data(data_path)
      data_size = raw_data.eval_data.n
      if data_size % config['eval_batch_size'] == 0:
          eval_steps = data_size // config['eval_batch_size']
      else:
          eval_steps = data_size // config['eval_batch_size'] + 1
      total_num_correct = 0
      for ii in tqdm(range(eval_steps)):
          x_eval_batch, y_eval_batch = raw_data.eval_data.get_next_batch(config['eval_batch_size'], multiple_passes=False)            
          eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch}
          if 'finetuned_on_cifar10' in config['model_dir'] or 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:
            num_correct = sess.run(model.target_task_num_correct, feed_dict=eval_dict)
          else:
            num_correct = sess.run(model.num_correct, feed_dict=eval_dict)
          total_num_correct += num_correct
      eval_acc = total_num_correct / data_size
      with open(log_file_path, "a+") as f:
        f.write('\nClean Accuracy: {:.2f}%'.format(100.0 * eval_acc))
      print('Clean Accuracy: {:.2f}%'.format(100.0 * eval_acc))
      print('Results saved at ', log_file_path)

if __name__ == '__main__':
  import json

  # with open('attack_config.json') as config_file:
  #   config = json.load(config_file)

  model_dir = config['model_dir']

  checkpoint = tf.train.latest_checkpoint(model_dir)
  
  adv_examples_path = config['store_adv_path']
  if adv_examples_path == None:
    model_name = config['model_dir'].split('/')[1]
    if config['attack_name'] == None:
      adv_examples_path = "attacks/{}_attack.npy".format(model_name)
      # if config['dataset'] == 'cifar10':
      #     adv_examples_path = "attacks/{}_attack.npy".format(model_name)
      # else:
      #     adv_examples_path = "attacks/{}_c100attack.npy".format(model_name)
    else:
      adv_examples_path = "attacks/{}_{}_attack.npy".format(model_name, config['attack_name'])
      # if config['dataset'] == 'cifar10':
      #     adv_examples_path = "attacks/{}_{}_attack.npy".format(model_name, config['attack_name'])
      # else:
      #     adv_examples_path = "attacks/{}_{}_c100attack.npy".format(model_name, config['attack_name'])

    # if config['attack_norm'] == '2':
    #   adv_examples_path = adv_examples_path.replace("attack.npy", "l2attack.npy")

  x_adv = np.load(adv_examples_path)
    
  tf.set_random_seed(config['tf_seed'])
  np.random.seed(config['np_seed'])

  if checkpoint is None:
    print('No checkpoint found')
  elif x_adv.shape != (10000, 32, 32, 3):
    print('Invalid shape: expected (10000, 32, 32, 3), found {}'.format(x_adv.shape))
  elif np.amax(x_adv) > 255.0001 or np.amin(x_adv) < -0.0001:
    print('Invalid pixel range. Expected [0, 255], found [{}, {}]'.format(
                                                              np.amin(x_adv),
                                                              np.amax(x_adv)))
  else:
    print("adv_examples_path: ", adv_examples_path)
    run_attack(checkpoint, x_adv, config['epsilon'])
