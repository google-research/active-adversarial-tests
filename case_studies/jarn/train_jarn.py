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

"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os
import shutil
from timeit import default_timer as timer
import tensorflow as tf
import numpy as np
import sys
from model_jarn import Model, JarnConvDiscriminatorModel, InputGradEncoderModel, GradImageMixer
import cifar10_input
import pdb
from tqdm import tqdm
import subprocess
import time
from numba import cuda

import config

def get_path_dir(data_dir, dataset, **_):
    if dataset == "cifar10":
        path = "../../data/cifar10/cifar-10-batches-py/"
    else:
        path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


def train(tf_seed, np_seed, train_steps, out_steps, summary_steps, checkpoint_steps, step_size_schedule,
          weight_decay, momentum, train_batch_size, epsilon, replay_m, model_dir, model_suffix, dataset, 
          beta, gamma, disc_update_steps, adv_update_steps_per_iter, disc_layers, disc_base_channels, steps_before_adv_opt, adv_encoder_type, enc_output_activation, 
          sep_opt_version, grad_image_ratio, final_grad_image_ratio, num_grad_image_ratios, normalize_zero_mean, eval_adv_attack, same_optimizer, only_fully_connected, img_random_pert, **kwargs):
    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)

    model_dir = model_dir + 'JARN_%s_b%d_beta_%.3f_gamma_%.3f_disc_update_steps%d_l%dbc%d' % (dataset, train_batch_size, beta, gamma, disc_update_steps, disc_layers, disc_base_channels) 

    if img_random_pert:
        model_dir = model_dir + '_imgpert'

    if steps_before_adv_opt != 0:
        model_dir = model_dir + '_advdelay%d' % (steps_before_adv_opt)
    if adv_encoder_type != 'simple':
        model_dir = model_dir + '_%senc' % (adv_encoder_type)
    if enc_output_activation != None:
        model_dir = model_dir + '_%sencact' % (enc_output_activation)
    if grad_image_ratio != 1:
        model_dir = model_dir + '_gradmixratio%.2f' % (grad_image_ratio)

    if normalize_zero_mean:
        model_dir = model_dir + '_zeromeaninput'
    if train_steps != 80000:
        model_dir = model_dir + '_%dsteps' % (train_steps)
    if same_optimizer == False:
        model_dir = model_dir + '_adamDopt'
    if only_fully_connected:
        model_dir = model_dir + '_FCdisc'

    if tf_seed != 451760341:
        model_dir = model_dir + '_tf_seed%d' % (tf_seed)
    if np_seed != 216105420:
        model_dir = model_dir + '_np_seed%d' % (np_seed)

    model_dir = model_dir + model_suffix

    # Setting up the data and the model
    data_path = get_path_dir(dataset=dataset, **kwargs)
    raw_data = cifar10_input.CIFAR10Data(data_path)
    global_step = tf.train.get_or_create_global_step()
    increment_global_step_op = tf.assign(global_step, global_step+1)
    model = Model(mode='train', dataset=dataset, train_batch_size=train_batch_size, normalize_zero_mean=normalize_zero_mean)

    # Setting up the optimizers
    boundaries = [int(sss[0]) for sss in step_size_schedule][1:]
    values = [sss[1] for sss in step_size_schedule]
    learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
    c_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    e_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    
    if same_optimizer:
        d_optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    else:
        print("Using ADAM opt for DISC model")
        d_optimizer = tf.train.AdamOptimizer(learning_rate = 0.001)

    # Using target softmax for input grad
    input_grad = tf.gradients(model.target_softmax, model.x_input, name="gradients_ig")[0]

    # Setting up the gradimagemixer model
    grad_image_mixer = GradImageMixer(train_batch_size=train_batch_size, grad_input_tensor=input_grad, image_input_tensor=model.input_standardized, normalize_zero_mean=normalize_zero_mean)

    # Setting up the discriminator model
    encoder_model = InputGradEncoderModel(mode='train', train_batch_size=train_batch_size, encoder_type=adv_encoder_type, output_activation=enc_output_activation, x_modelgrad_input_tensor=grad_image_mixer.output, normalize_zero_mean=normalize_zero_mean)
    transformed_input_grad = encoder_model.x_modelgrad_transformed
    labels_input_grad = tf.zeros( tf.shape(input_grad)[0] , dtype=tf.int64)
    labels_disc_image_input = tf.ones( tf.shape(input_grad)[0] , dtype=tf.int64)
    disc_model = JarnConvDiscriminatorModel(mode='train', dataset=dataset, train_batch_size=train_batch_size, num_conv_layers=disc_layers, base_num_channels=disc_base_channels, normalize_zero_mean=normalize_zero_mean,
        x_modelgrad_input_tensor=transformed_input_grad, y_modelgrad_input_tensor=labels_input_grad, x_image_input_tensor=model.input_standardized, y_image_input_tensor=labels_disc_image_input, only_fully_connected=only_fully_connected)

    t_vars = tf.trainable_variables()
    C_vars = [var for var in t_vars if 'classifier' in var.name]
    D_vars = [var for var in t_vars if 'discriminator' in var.name]
    E_vars = [var for var in t_vars if 'encoder' in var.name]

    # Classifier: Optimizing computation
    # total classifier loss: Add discriminator loss into total classifier loss
    total_loss = model.mean_xent + weight_decay * model.weight_decay_loss - beta * disc_model.mean_xent
    classification_c_loss = model.mean_xent + weight_decay * model.weight_decay_loss
    adv_c_loss = - beta * disc_model.mean_xent

    # Discriminator: Optimizating computation
    # discriminator loss
    total_d_loss = disc_model.mean_xent + weight_decay * disc_model.weight_decay_loss

    # Train classifier
    # classifier opt step
    # AUTHOR added the next two lines to fix batch norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        final_grads = c_optimizer.compute_gradients(total_loss, var_list=C_vars)
        no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in final_grads]
        c_min_step = c_optimizer.apply_gradients(no_pert_grad)

        classification_final_grads = c_optimizer.compute_gradients(classification_c_loss, var_list=C_vars)
        classification_no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in classification_final_grads]
        c_classification_min_step = c_optimizer.apply_gradients(classification_no_pert_grad)

        # Encoder: Optimizating computation
        # encoder loss
        total_e_loss = weight_decay * encoder_model.weight_decay_loss - gamma * disc_model.mean_xent
        e_min_step = e_optimizer.minimize(total_e_loss, var_list=E_vars)

        # discriminator opt step
        d_min_step = d_optimizer.minimize(total_d_loss, var_list=D_vars)

    # Loss gradients to the model params
    logit_weights = tf.get_default_graph().get_tensor_by_name('classifier/logit/DW:0')
    last_conv_weights = tf.get_default_graph().get_tensor_by_name('classifier/unit_3_4/sub2/conv2/DW:0')
    first_conv_weights = tf.get_default_graph().get_tensor_by_name('classifier/input/init_conv/DW:0')

    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(max_to_keep=1)
    tf.summary.scalar('C accuracy', model.accuracy)
    tf.summary.scalar('D accuracy', disc_model.accuracy)
    tf.summary.scalar('C xent', model.xent / train_batch_size)
    tf.summary.scalar('D xent', disc_model.xent / train_batch_size)
    tf.summary.scalar('total C loss', total_loss / train_batch_size)
    tf.summary.scalar('total D loss', total_d_loss / train_batch_size)
    tf.summary.scalar('adv C loss', adv_c_loss / train_batch_size)
    tf.summary.scalar('C cls xent loss', model.mean_xent)
    merged_summaries = tf.summary.merge_all()

    with tf.Session() as sess:
        print('params >>> \n model dir: %s \n dataset: %s \n training batch size: %d \n' % (model_dir, dataset, train_batch_size))

        data = cifar10_input.AugmentedCIFAR10Data(raw_data, sess, model)

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(model_dir + '/eval')
        sess.run(tf.global_variables_initializer())

        # Main training loop
        for ii in tqdm(range(train_steps)):

            x_batch, y_batch = data.train_data.get_next_batch(train_batch_size, multiple_passes=True)            
            if img_random_pert:
                x_batch = x_batch + np.random.uniform(-epsilon, epsilon, x_batch.shape)
                x_batch = np.clip(x_batch, 0, 255) # ensure valid pixel range

            labels_image_disc = np.ones_like( y_batch, dtype=np.int64)

            nat_dict = {model.x_input: x_batch, model.y_input: y_batch, 
                disc_model.x_image_input: x_batch, disc_model.y_image_input: labels_image_disc, grad_image_mixer.grad_ratio: grad_image_ratio}

            # Output to stdout
            if ii % summary_steps == 0:
                train_acc, train_disc_acc, train_c_loss, train_e_loss, train_d_loss, train_adv_c_loss, summary = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_e_loss, total_d_loss, adv_c_loss, merged_summaries], feed_dict=nat_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
                
                x_eval_batch, y_eval_batch = data.eval_data.get_next_batch(train_batch_size, multiple_passes=True)
                if img_random_pert:
                    x_eval_batch = x_eval_batch + np.random.uniform(-epsilon, epsilon, x_eval_batch.shape)
                    x_eval_batch = np.clip(x_eval_batch, 0, 255) # ensure valid pixel range

                labels_image_disc = np.ones_like( y_eval_batch, dtype=np.int64)

                eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch, 
                    disc_model.x_image_input: x_eval_batch, disc_model.y_image_input: labels_image_disc, grad_image_mixer.grad_ratio: grad_image_ratio}

                val_acc, val_disc_acc, val_c_loss, val_e_loss, val_d_loss, val_adv_c_loss, summary = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_e_loss, total_d_loss, adv_c_loss, merged_summaries], feed_dict=eval_dict)
                eval_summary_writer.add_summary(summary, global_step.eval(sess))
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_acc * 100,
                                                                                                  val_acc * 100))
                print('    training nat disc accuracy {:.4}% -- validation nat disc accuracy {:.4}%'.format(train_disc_acc * 100,
                                                                                                  val_disc_acc * 100))
                print('    training nat c loss: {},     e loss: {},     d loss: {},     adv c loss: {}'.format( train_c_loss, train_e_loss, train_d_loss, train_adv_c_loss))
                print('    validation nat c loss: {},     e loss: {},     d loss: {},     adv c loss: {}'.format( val_c_loss, val_e_loss, val_d_loss, val_adv_c_loss))

                sys.stdout.flush()
            # Tensorboard summaries
            elif ii % out_steps == 0:
                nat_acc, nat_disc_acc, nat_c_loss, nat_e_loss, nat_d_loss, nat_adv_c_loss = sess.run([model.accuracy, disc_model.accuracy, total_loss, total_e_loss, total_d_loss, adv_c_loss], feed_dict=nat_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
                print('    training nat disc accuracy {:.4}%'.format(nat_disc_acc * 100))
                print('    training nat c loss: {},     e loss: {},     d loss: {},      adv c loss: {}'.format( nat_c_loss, nat_e_loss, nat_d_loss, nat_adv_c_loss))

            # Write a checkpoint
            if (ii+1) % checkpoint_steps == 0:
                saver.save(sess, os.path.join(model_dir, 'checkpoint'), global_step=global_step)
            
            if ii >= steps_before_adv_opt:
                # Actual training step for Classifier
                sess.run([c_min_step, e_min_step], feed_dict=nat_dict)
                sess.run(increment_global_step_op)
                
                if ii % disc_update_steps == 0:
                    # Actual training step for Discriminator
                    sess.run(d_min_step, feed_dict=nat_dict)
            else:
                # only train on classification loss
                sess.run(c_classification_min_step, feed_dict=nat_dict)
                sess.run(increment_global_step_op)
                  
        # full test evaluation
        raw_data = cifar10_input.CIFAR10Data(data_path)
        data_size = raw_data.eval_data.n
        
        eval_steps = data_size // train_batch_size

        total_num_correct = 0
        for ii in tqdm(range(eval_steps)):
            x_eval_batch, y_eval_batch = raw_data.eval_data.get_next_batch(train_batch_size, multiple_passes=False)
            eval_dict = {model.x_input: x_eval_batch, model.y_input: y_eval_batch}
            num_correct = sess.run(model.num_correct, feed_dict=eval_dict)
            total_num_correct += num_correct
        eval_acc = total_num_correct / data_size
        
        clean_eval_file_path = os.path.join(model_dir, 'full_clean_eval_acc.txt')
        with open(clean_eval_file_path, "a+") as f:
            f.write("Full clean eval_acc: {}%".format(eval_acc*100))
        print("Full clean eval_acc: {}%".format(eval_acc*100))


        devices = sess.list_devices()
        for d in devices:
            print("sess' device names:")
            print(d.name)

    return model_dir
            
if __name__ == '__main__':
    args = config.get_args()
    args_dict = vars(args)
    model_dir = train(**args_dict)
    if args_dict['eval_adv_attack']:
        cuda.select_device(0)
        cuda.close()

        print("{}: Evaluating on CIFAR10 fgsm and pgd attacks".format(datetime.now()))
        subprocess.run("python pgd_attack.py --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir {} ; python run_attack.py --attack_name fgsm --save_eval_log --model_dir {} ; python pgd_attack.py --save_eval_log --model_dir {} ; python run_attack.py --save_eval_log --model_dir {} ; python pgd_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python run_attack.py --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {}".format(model_dir, model_dir, model_dir, model_dir, model_dir, model_dir), shell=True)
        print("{}: Ended evaluation on CIFAR10 fgsm and pgd  attacks".format(datetime.now()))
