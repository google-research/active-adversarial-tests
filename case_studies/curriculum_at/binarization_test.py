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

import torch

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
import tqdm_utils

from PGD_attack import LinfPGDAttack
from active_tests.decision_boundary_binarization import interior_boundary_discrimination_attack

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

    class ModelWrapper:
      def __init__(self, model, weight_shape, bias_shape, num_classes=2):
        self.weight = tf.placeholder(dtype=tf.float32, shape=weight_shape)
        self.bias = tf.placeholder(dtype=tf.float32, shape=bias_shape)

        y = model.neck

        # TODO: check whether we need a separate placeholder for the binary label
        self.y_input = model.y_input
        self.x_input = model.x_input

        self.logits = y @ tf.transpose(self.weight) + tf.reshape(self.bias, (1, -1))
        self.predictions = tf.argmax(self.logits, 1)

        self.pre_softmax = self.logits

        # define losses
        self.y_xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pre_softmax, labels=self.y_input)
        self.xent = tf.reduce_sum(self.y_xent, name='y_xent')

        # for top-2 logit diff loss
        self.label_mask = tf.one_hot(self.y_input,
                                     num_classes,
                                     on_value=1.0,
                                     off_value=0.0,
                                     dtype=tf.float32)
        self.correct_logit = tf.reduce_sum(self.label_mask * self.pre_softmax, axis=1)
        self.wrong_logit = tf.reduce_max((1-self.label_mask) * self.pre_softmax - 1e4*self.label_mask, axis=1)
        # TODO: why the plus 50?
        # self.top2_logit_diff_loss = -tf.nn.relu(self.correct_logit - self.wrong_logit + 50)
        self.top2_logit_diff_loss = -self.correct_logit + self.wrong_logit


    wrapped_model = ModelWrapper(model, (2, 640), (2,))

    attack = LinfPGDAttack(wrapped_model,
                           config['epsilon'],
                           config['num_steps'],
                           config['step_size'],
                           config['random_start'],
                           config['loss_func'],
                           dataset=config['dataset'])

    def run_attack(m, l):
      linear_layer = m[-1]
      del m

      # initialize an attack (it's a white box attack, and it's allowed to look
      # at the internals of the model in any way it wants)
      # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)
      # m = PyTorchToTensorFlow1Wrapper(m, "cpu")

      weights_feed_dict = {
          wrapped_model.weight: linear_layer.weight.data.numpy(),
          wrapped_model.bias: linear_layer.bias.data.numpy()
      }

      assert len(l) == 1, len(l)
      for x, y in l:
        x_batch = x.numpy().transpose((0, 2, 3, 1)) * 255.0
        y_batch = y.numpy()

        if config['attack_norm'] == 'inf':
          x_batch_adv = attack.perturb(x_batch, y_batch, sess, weights_feed_dict)
        elif config['attack_norm'] == '2':
          x_batch_adv = attack.perturb_l2(x_batch, y_batch, sess, weights_feed_dict)
        elif config['attack_norm'] == 'TRADES':
          x_batch_adv = attack.perturb_TRADES(x_batch, y_batch, sess, weights_feed_dict)

        logits, y_pred = sess.run((wrapped_model.logits, wrapped_model.predictions),
                          feed_dict={model.x_input: x_batch_adv,
                                     **weights_feed_dict})
        is_adv = y_pred != y_batch

        return is_adv, (torch.Tensor(x_batch_adv) / 255.0, torch.Tensor(logits))

    random_indices = list(range(len(cifar.eval_data.xs)))
    np.random.shuffle(random_indices)

    x_batch = []
    y_batch = []
    for j in range(config['num_eval_examples']):
      x_ = cifar.eval_data.xs[random_indices[j]]
      y_ = cifar.eval_data.ys[random_indices[j]]
      x_batch.append(x_)
      y_batch.append(y_)
    x_batch = np.array(x_batch).transpose((0, 3, 1, 2)) / 255.0
    y_batch = np.array(y_batch)

    from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper, PyTorchToTensorFlow1Wrapper
    from utils import build_dataloader_from_arrays

    test_loader = build_dataloader_from_arrays(x_batch, y_batch, batch_size=32)

    def _model_forward_pass(x, features_and_logits: bool = False, features_only: bool = False):
      if features_and_logits:
        assert not features_only, "Only one of the flags must be set."
      if features_and_logits:
        return sess.run(
            (model.neck, model.pre_softmax),
            feed_dict={model.x_input: x.transpose(0, 2, 3, 1) * 255.0})
      elif features_only:
        return sess.run(
            model.neck,
            feed_dict={model.x_input: x.transpose(0, 2, 3, 1) * 255.0})
      else:
        raise ValueError

    feature_extractor = TensorFlow1ToPyTorchWrapper(
        logit_forward_pass=_model_forward_pass,
        logit_forward_and_backward_pass=lambda x, **kwargs: sess.run(
            model.feature_grad,
            feed_dict={model.x_input: x.transpose(0, 2, 3, 1) * 255.0}) / 255.0
    )

    assert config["n_boundary_points"] is not None
    assert config["n_inner_points"] is not None

    from argparse_utils import DecisionBoundaryBinarizationSettings

    with tqdm_utils.tqdm_print():
      scores_logit_differences_and_validation_accuracies = \
        interior_boundary_discrimination_attack(
            feature_extractor,
            test_loader,
            attack_fn=lambda m, l, kwargs: run_attack(m, l),
            linearization_settings=DecisionBoundaryBinarizationSettings(
                epsilon=config["epsilon"]/255.0,
                norm="linf",
                lr=10000,
                n_boundary_points=config["n_boundary_points"],
                n_inner_points=config["n_inner_points"],
                adversarial_attack_settings=None,
                optimizer="sklearn"
            ),
            n_samples=config['num_eval_examples'],
            device="cpu",
            n_samples_evaluation=200,
            n_samples_asr_evaluation=200,
            rescale_logits="adaptive",
            sample_training_data_from_corners=config["sample_from_corners"],
            decision_boundary_closeness=0.99999
            #args.num_samples_test * 10
        )

    scores = [it[0] for it in scores_logit_differences_and_validation_accuracies]
    validation_scores = [it[3] for it in scores_logit_differences_and_validation_accuracies]
    if validation_scores[0] is None:
      validation_scores = (np.nan, np.nan)
    else:
      validation_scores = np.array(validation_scores)
      validation_scores = tuple(np.mean(validation_scores, 0))
    logit_differences = [(it[1], it[2]) for it in
                         scores_logit_differences_and_validation_accuracies]
    logit_differences = np.array(logit_differences)
    relative_performance = (logit_differences[:, 0] - logit_differences[:,
                                                      1]) / logit_differences[:,
                                                            1]

    test_result = (np.mean(scores), np.mean(relative_performance),
                   np.std(relative_performance), validation_scores)

    print("\tinterior-vs-boundary discrimination (ce loss), ASR: {0}\n”, "
          "\t\tNormalized Logit-Difference-Improvement: {1} +- {2}\n"
          "\t\tValidation Accuracy (inner, boundary): {3}".format(
        *test_result))


