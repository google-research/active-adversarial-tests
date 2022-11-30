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



import cifar10_input

import config_attack





config = vars(config_attack.get_args())



data_path = config['data_path']



def run_attack(checkpoint, x_adv, epsilon):

  if config['dataset'] == 'cifar10':

    cifar = cifar10_input.CIFAR10Data(data_path)

  elif config['dataset'] == 'cifar100':

    cifar = cifar100_input.CIFAR100Data(data_path)
  else:
    cifar = tinyimagenet_input.TinyImagenetData()



  if 'GTP' in config['model_dir']:

    print("GTP MODEL")

    from model_new import Model, ModelTinyImagnet
    if config['dataset'] == 'cifar10' or config['dataset'] == 'cifar100':
      model = Model(mode='train', dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)
    else:
      model = ModelTinyImagnet(mode='train', dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)

  elif 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:

    print("finetuned tinyimagenet MODEL")

    from model_new import ModelTinyImagenetSourceExtendedLogits

    full_source_model_x_input = tf.placeholder(tf.float32, shape = [None, 32, 32, 3])

    upresized_full_source_model_x_input = tf.image.resize_images(full_source_model_x_input, size=[64, 64])

    if config['dataset'] == 'cifar10':

      model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset='tinyimagenet', target_task_class_num=10, train_batch_size=config['eval_batch_size'], input_tensor=upresized_full_source_model_x_input)

    elif config['dataset'] == 'cifar100':

      model = ModelTinyImagenetSourceExtendedLogits(mode='train', dataset='tinyimagenet', target_task_class_num=100, train_batch_size=config['eval_batch_size'], input_tensor=upresized_full_source_model_x_input)

  

    model.x_input = full_source_model_x_input



    t_vars = tf.trainable_variables()

    source_model_vars = [var for var in t_vars if ('discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]

    source_model_target_logit_vars = [var for var in t_vars if 'target_task_logit' in var.name]

    source_model_saver = tf.train.Saver(var_list=source_model_vars)

    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars

    finetuned_source_model_saver = tf.train.Saver(var_list=finetuned_source_model_vars)

  elif 'finetuned_on_cifar100' in config['model_dir']:

    print("finetuned MODEL")

    from model_original_cifar_challenge import ModelExtendedLogits

    model = ModelExtendedLogits(mode='train', target_task_class_num=100, train_batch_size=config['eval_batch_size'])

    

    t_vars = tf.trainable_variables()

    source_model_vars = [var for var in t_vars if ('discriminator' not in var.name and 'classifier' not in var.name and 'target_task_logit' not in var.name)]

    source_model_target_logit_vars = [var for var in t_vars if 'target_task_logit' in var.name]

    source_model_saver = tf.train.Saver(var_list=source_model_vars)

    finetuned_source_model_vars = source_model_vars + source_model_target_logit_vars

    finetuned_source_model_saver = tf.train.Saver(var_list=finetuned_source_model_vars)

  elif ('adv_trained' in config['model_dir'] or 'naturally_trained' in config['model_dir'] or 'a_very_robust_model' in config['model_dir']):

    print("original challenge MODEL")

    from free_model_original import Model

    model = Model(mode='eval', dataset=config['dataset'], train_batch_size=config['eval_batch_size'])

  elif 'IGAM' in config['model_dir']:

    print("IGAM MODEL")

    from model_new import Model

    model = Model(mode='train', dataset=config['dataset'], train_batch_size=config['eval_batch_size'], normalize_zero_mean=True)

  else:

    print("other MODEL")

    from free_model import Model

    model = Model(mode='eval', dataset=config['dataset'], train_batch_size=config['eval_batch_size'])



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

    if 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:

      sess.run(tf.global_variables_initializer())

      source_model_file = tf.train.latest_checkpoint("models/model_AdvTrain-igamsource-IGAM-tinyimagenet_b16")

      source_model_saver.restore(sess, source_model_file)   

      finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])

      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)

    elif 'finetuned_on_cifar100' in config['model_dir']:

      sess.run(tf.global_variables_initializer())

      source_model_file = tf.train.latest_checkpoint("models/adv_trained")

      source_model_saver.restore(sess, source_model_file)   

      finetuned_source_model_file = tf.train.latest_checkpoint(config['model_dir'])

      finetuned_source_model_saver.restore(sess, finetuned_source_model_file)

    else:

      saver.restore(sess, checkpoint)



    # Iterate over the samples batch-by-batch

    for ibatch in range(num_batches):

      bstart = ibatch * eval_batch_size

      bend = min(bstart + eval_batch_size, num_eval_examples)



      x_batch = x_adv[bstart:bend, :]

      y_batch = cifar.eval_data.ys[bstart:bend]



      dict_adv = {model.x_input: x_batch,

                  model.y_input: y_batch}

      

      if 'finetuned_on_cifar10' in config['model_dir'] or 'adv_trained_tinyimagenet_finetuned_on_c10_upresize' in config['model_dir']:

        cur_corr, y_pred_batch = sess.run([model.target_task_num_correct, model.target_task_predictions],

                                          feed_dict=dict_adv)

      else:

        cur_corr, y_pred_batch = sess.run([model.num_correct, model.predictions],

                                          feed_dict=dict_adv)



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

      log_file_path = os.path.join(log_dir, log_filename)

      with open(log_file_path, "w") as f:

        f.write('Model checkpoint: {} \n'.format(checkpoint))

        f.write('Adv Accuracy: {:.2f}%'.format(100.0 * accuracy))

      print('Results saved at ', log_file_path)



      # full test evaluation

      if config['dataset'] == 'cifar10':
        raw_data = cifar10_input.CIFAR10Data(data_path)
      elif config['dataset'] == 'cifar100':
        raw_data = cifar100_input.CIFAR100Data(data_path)
      else:
        raw_data = tinyimagenet_input.TinyImagenetData()

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
        if config['dataset'] == 'cifar10':
          adv_examples_path = "attacks/{}_attack.npy".format(model_name)
        elif config['dataset'] == 'cifar100':
          adv_examples_path = "attacks/{}_c100attack.npy".format(model_name)
        else:
          adv_examples_path = "attacks/{}_tinyattack.npy".format(model_name)

    else:
        if config['dataset'] == 'cifar10':
          adv_examples_path = "attacks/{}_{}_attack.npy".format(model_name, config['attack_name'])
        elif config['dataset'] == 'cifar100':
          adv_examples_path = "attacks/{}_{}_c100attack.npy".format(model_name, config['attack_name'])
        else:
          adv_examples_path = "attacks/{}_{}_tinyattack.npy".format(model_name, config['attack_name'])



    if config['attack_norm'] == '2':

      adv_examples_path = adv_examples_path.replace("attack.npy", "l2attack.npy")



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


