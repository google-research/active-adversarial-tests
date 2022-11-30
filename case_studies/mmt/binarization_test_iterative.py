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

from __future__ import print_function

import logging

logging.getLogger('tensorflow').setLevel(logging.FATAL)
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.ERROR)

import pdb
from functools import partial

import keras
import torch
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten, Lambda
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, \
  ReduceLROnPlateau
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import mnist, cifar10, cifar100
import tensorflow as tf
import numpy as np
import os
from scipy.io import loadmat
import math

from torch.utils.data import DataLoader

from active_tests.decision_boundary_binarization import LogitRescalingType
from active_tests.decision_boundary_binarization import \
  _train_logistic_regression_classifier
from active_tests.decision_boundary_binarization import format_result
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack
from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper
from utils import build_dataloader_from_arrays
from mmt_utils.model import resnet_v1, resnet_v2
import cleverhans.attacks as attacks
from cleverhans.utils_tf import model_eval
from mmt_utils.keras_wraper_ensemble import KerasModelWrapper
from mmt_utils.utils_model_eval import model_eval_targetacc
from sklearn.metrics import roc_auc_score

FLAGS = tf.app.flags.FLAGS


def main():
  tf.app.flags.DEFINE_integer('epsilon', 8, 'attack radius')
  tf.app.flags.DEFINE_integer('n_inner_points', 999, '')
  tf.app.flags.DEFINE_integer('n_boundary_points', 1, '')
  tf.app.flags.DEFINE_integer('n_samples', 512, '')

  tf.app.flags.DEFINE_integer('batch_size', 512, 'batch_size for attack')
  tf.app.flags.DEFINE_string('optimizer', 'mom', '')
  tf.app.flags.DEFINE_float('mean_var', 10, 'parameter in MMLDA')
  tf.app.flags.DEFINE_string('attack_method', 'FastGradientMethod', '')
  tf.app.flags.DEFINE_string('attack_method_for_advtrain', 'FastGradientMethod',
                             '')
  tf.app.flags.DEFINE_integer('version', 2, '')
  tf.app.flags.DEFINE_bool('use_target', False,
                           'whether use target attack or untarget attack for adversarial training')
  tf.app.flags.DEFINE_integer('num_iter', 10, '')
  tf.app.flags.DEFINE_bool('use_ball', True, 'whether use ball loss or softmax')
  tf.app.flags.DEFINE_bool('use_MMLDA', True, 'whether use MMLDA or softmax')
  tf.app.flags.DEFINE_bool('use_advtrain', True,
                           'whether use advtraining or normal training')
  tf.app.flags.DEFINE_float('adv_ratio', 1.0,
                            'the ratio of adversarial examples in each mini-batch')
  tf.app.flags.DEFINE_integer('epoch', 1, 'the epoch of model to load')
  tf.app.flags.DEFINE_bool('use_BN', True,
                           'whether use batch normalization in the network')
  tf.app.flags.DEFINE_string('dataset', 'mnist', '')
  tf.app.flags.DEFINE_bool('normalize_output_for_ball', True,
                           'whether apply softmax in the inference phase')
  tf.app.flags.DEFINE_bool('use_random', False,
                           'whether use random center or MMLDA center in the network')
  tf.app.flags.DEFINE_bool('use_dense', True,
                           'whether use extra dense layer in the network')
  tf.app.flags.DEFINE_bool('use_leaky', False,
                           'whether use leaky relu in the network')

  tf.app.flags.DEFINE_string('checkpoint', None, '')

  # For calculate AUC-scores
  tf.app.flags.DEFINE_bool('is_calculate_auc', False,
                           'whether to calculate auc scores')
  tf.app.flags.DEFINE_bool('is_auc_metric_softmax_for_MMC', False,
                           'whether use softmax to calculate auc metrics for MMC')

  tf.app.flags.DEFINE_bool('sample_from_corners', False, '')

  run_test()


# MMLDA prediction function
def MMLDA_layer(x, means, num_class, use_ball,
    normalize_output_for_ball=None):
  if normalize_output_for_ball is None:
    normalize_output_for_ball = FLAGS.normalize_output_for_ball

  # x_shape = batch_size X num_dense
  x_expand = tf.tile(tf.expand_dims(x, axis=1),
                     [1, num_class, 1])  # batch_size X num_class X num_dense
  mean_expand = tf.expand_dims(means, axis=0)  # 1 X num_class X num_dense
  logits = -tf.reduce_sum(tf.square(x_expand - mean_expand),
                          axis=-1)  # batch_size X num_class
  if use_ball == True:
    if normalize_output_for_ball == False:
      return logits
    else:
      return tf.nn.softmax(logits, axis=-1)
  else:
    return tf.nn.softmax(logits, axis=-1)


def setup_model_and_load_data():
  # Load the dataset
  if FLAGS.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = np.expand_dims(x_train, axis=3)
    x_test = np.expand_dims(x_test, axis=3)
    epochs = 50
    num_class = 10
    epochs_inter = [30, 40]
    x_place = tf.placeholder(tf.float32, shape=(None, 28, 28, 3))

  elif FLAGS.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    epochs = 200
    num_class = 10
    epochs_inter = [100, 150]
    x_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

  elif FLAGS.dataset == 'cifar100':
    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    epochs = 200
    num_class = 100
    epochs_inter = [100, 150]
    x_place = tf.placeholder(tf.float32, shape=(None, 32, 32, 3))

  else:
    print('Unknown dataset')

  # These parameters are usually fixed
  subtract_pixel_mean = True
  version = FLAGS.version  # Model version
  n = 18  # n=5 for resnet-32 v1, n=18 for Resnet110 (according to README.md)

  # Computed depth from supplied model parameter n
  if version == 1:
    depth = n * 6 + 2
    feature_dim = 64
  elif version == 2:
    depth = n * 9 + 2
    feature_dim = 256

  if FLAGS.use_random == True:
    name_random = '_random'
  else:
    name_random = ''

  if FLAGS.use_leaky == True:
    name_leaky = '_withleaky'
  else:
    name_leaky = ''

  if FLAGS.use_dense == True:
    name_dense = ''
  else:
    name_dense = '_nodense'

  # Load means in MMLDA
  kernel_dict = loadmat(
      'case_studies/mmt/kernel_paras/meanvar1_featuredim' + str(
          feature_dim) + '_class' + str(
          num_class) + name_random + '.mat')
  mean_logits = kernel_dict['mean_logits']  # num_class X num_dense
  mean_logits = FLAGS.mean_var * tf.constant(mean_logits, dtype=tf.float32)


  # Load the data.
  # Input image dimensions.
  input_shape = x_train.shape[1:]

  # Normalize data.
  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  # clip_min = 0.0
  # clip_max = 1.0
  # If subtract pixel mean is enabled
  if subtract_pixel_mean:
    x_train_mean = np.mean(x_train, axis=0, keepdims=True)
    # x_train -= x_train_mean
    # x_test -= x_train_mean
    # clip_min -= np.max(x_train_mean)
    # clip_max -= np.min(x_train_mean)

  # Convert class vectors to binary class matrices.
  y_train = keras.utils.to_categorical(y_train, num_class)
  y_test = keras.utils.to_categorical(y_test, num_class)

  # Define input TF placeholder
  y_place = tf.placeholder(tf.float32, shape=(None, num_class))
  sess = tf.Session()
  keras.backend.set_session(sess)

  model_input = Input(shape=input_shape)

  if subtract_pixel_mean:
    normalized_model_input = Lambda(lambda x: x - x_train_mean)(model_input)
  else:
    normalized_model_input = model_input

  # preprocessed_input =

  # dim of logtis is batchsize x dim_means
  if version == 2:
    original_model, _, _, _, final_features = resnet_v2(
        immediate_input=normalized_model_input, input=model_input, depth=depth,
        num_classes=num_class, \
        use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense,
        use_leaky=FLAGS.use_leaky)
  else:
    original_model, _, _, _, final_features = resnet_v1(
        immediate_input=normalized_model_input, input=model_input, depth=depth,
        num_classes=num_class, \
        use_BN=FLAGS.use_BN, use_dense=FLAGS.use_dense,
        use_leaky=FLAGS.use_leaky)

  if FLAGS.use_BN == True:
    BN_name = '_withBN'
    print('Use BN in the model')
  else:
    BN_name = '_noBN'
    print('Do not use BN in the model')

  # Whether use target attack for adversarial training
  if FLAGS.use_target == False:
    is_target = ''
  else:
    is_target = 'target'

  if FLAGS.use_advtrain == True:
    dirr = 'advtrained_models/' + FLAGS.dataset + '/'
    attack_method_for_advtrain = '_' + is_target + FLAGS.attack_method_for_advtrain
    adv_ratio_name = '_advratio' + str(FLAGS.adv_ratio)
    mean_var = int(FLAGS.mean_var)
  else:
    dirr = 'trained_models/' + FLAGS.dataset + '/'
    attack_method_for_advtrain = ''
    adv_ratio_name = ''
    mean_var = FLAGS.mean_var

  if FLAGS.use_MMLDA == True:
    print('Using MMLDA')
    new_layer = Lambda(partial(MMLDA_layer, means=mean_logits,
                               num_class=num_class, use_ball=FLAGS.use_ball))
    predictions = new_layer(final_features)
    model = Model(input=model_input, output=predictions)
  else:
    print('Using softmax loss')
    model = original_model

  model.load_weights(FLAGS.checkpoint)

  return (model, model_input, final_features, mean_logits), \
         x_place, y_place, sess, (x_test, y_test)


def run_attack(m, l, kwargs, preds, x_adv, x_ph, y_ph, sess):
  del kwargs
  del m

  for x, y in l:
    x = x.numpy().transpose(0, 2, 3, 1)
    y = y.numpy()

    y_oh = keras.utils.to_categorical(y, 2)

    x_adv_np, logits = sess.run((x_adv, preds), {x_ph: x, y_ph: y_oh})
    y_pred = logits.argmax(-1)

    print(logits)

    is_adv = y_pred != y

    x_adv_np = x_adv_np.transpose((0, 3, 1, 2))

    return is_adv, (torch.tensor(x_adv_np, dtype=torch.float32),
                    torch.tensor(logits, dtype=torch.float32)
                    )


def train_classifier(
    n_features: int,
    train_loader: DataLoader,
    raw_train_loader: DataLoader,
    logits: torch.Tensor,
    device: str,
    rescale_logits: LogitRescalingType,
    linear_layer,
    clean_preds,
    x_ph,
    sess,
    binarized_model_wrapper
):
  # del raw_train_loader
  assert rescale_logits is None

  cls = _train_logistic_regression_classifier(
      n_features,
      train_loader,
      logits if logits is not None else None,
      "sklearn",
      20000,
      device,
      n_classes=2,
      rescale_logits=rescale_logits,
      solution_goodness="perfect",
      class_weight={0: 1, 1:5}
  )

  clw = cls.weight.data.detach().numpy()
  clb = cls.bias.data.detach().numpy()

  # since the first two MMT weights look roughly like this:
  # 1: (10, 0, ..., 0)
  # 2: (-1, 9.9, 0, ..., 0)
  # we can easily construct a weight matrix that remaps the feature space to
  # these two vectors
  nw = np.zeros((256, 256))
  nb = np.zeros(256)
  nw[:2] = clw
  nb[:2] = clb
  linear_layer.set_weights((nw.T, nb))

  # now test
  n_correct_inner = 0
  n_correct_outer = 0
  n_total_inner = 0
  n_total_outer = 0
  for x, y in raw_train_loader:
    x = x.numpy().transpose((0, 2, 3, 1))
    y = y.numpy()

    logits = sess.run(clean_preds, {x_ph: x})
    y_pred = logits.argmax(-1)
    is_correct = y_pred == y

    n_correct_inner += is_correct[y == 0].sum()
    n_correct_outer += is_correct[y == 1].sum()
    n_total_inner += (y == 0).sum()
    n_total_outer += (y == 1).sum()

  accuracy_inner = n_correct_inner / n_total_inner
  accuracy_outer = n_correct_outer / n_total_outer
  if accuracy_outer != 1.0:
    raise RuntimeError(f"Solver failed to find solution that perfectly detects boundary samples {accuracy_outer}")
  if accuracy_inner == 0:
    raise RuntimeError(f"Solver failed to find solution that detects (at least some) inner samples {accuracy_inner}")

  return binarized_model_wrapper


def setup_binarized_model(sess, model_input, final_features, mean_logits):
  assert FLAGS.use_ball

  # means_ph = tf.placeholder(tf.float32, shape=[2, mean_logits.shape[1]])
  # means = tf.Variable(np.zeros([2, mean_logits.shape[1]], dtype=np.float32),
  #                    name="binarized_model_means")
  # set_means = means.assign(means_ph)

  new_layer = Lambda(partial(MMLDA_layer, means=mean_logits[:2],
                             num_class=2, use_ball=FLAGS.use_ball,
                             normalize_output_for_ball=False))
  linear_layer = Dense(256)
  transformed_features = linear_layer(final_features)
  predictions = new_layer(transformed_features)
  model = Model(input=model_input, output=predictions)

  # will be used by binarization test to eval the model
  binarized_model_wrapper = BinarizedModelWrapper(model_input, predictions, sess)

  return model, linear_layer, binarized_model_wrapper


class BinarizedModelWrapper:
    def __init__(self, input, output, sess):
      self.input = input
      self.output = output
      self.sess = sess

    def __call__(self, x):
      return_torch = False
      if isinstance(x, torch.Tensor):
        x = x.cpu().numpy()
        return_torch = True
      if isinstance(x, np.ndarray):
        if x.shape[1] == 3:
          x = x.transpose(0, 2, 3, 1)

      out = self.sess.run(self.output, {self.input: x})
      if return_torch:
        out = torch.tensor(out, dtype=torch.float32)
      return out


def run_test():
  (model, model_input, final_features, mean_logits), \
  x_place, y_place, sess, (x_test, y_test) = \
    setup_model_and_load_data()

  del y_place
  y_place = tf.placeholder(tf.float32, shape=(None, 2))

  binarized_model, linear_layer, binarized_model_wrapper = \
    setup_binarized_model(
        sess,
        model_input,
        final_features,
        mean_logits)


  bin_clean_preds = binarized_model(x_place)
  clean_preds = model(x_place)
  wrap_ensemble = KerasModelWrapper(binarized_model, num_class=2, binarized_model=True)

  # Initialize the attack method
  if FLAGS.attack_method == 'MadryEtAl':
    att = attacks.MadryEtAl(wrap_ensemble)
  elif FLAGS.attack_method == 'FastGradientMethod':
    att = attacks.FastGradientMethod(wrap_ensemble)
  elif FLAGS.attack_method == 'MomentumIterativeMethod':
    att = attacks.MomentumIterativeMethod(wrap_ensemble)
  elif FLAGS.attack_method == 'BasicIterativeMethod':
    att = attacks.BasicIterativeMethod(wrap_ensemble)
  elif FLAGS.attack_method == "Adaptive":
    from adaptive_attack import FeatureSpaceProjectedGradientDescent
    att = FeatureSpaceProjectedGradientDescent(wrap_ensemble, logit_means=mean_logits,
                                               projection="l2")

  # Consider the attack to be constant
  eval_par = {'batch_size': FLAGS.batch_size}

  # TODO: shouldn't this be a 255?
  eps_ = FLAGS.epsilon / 256.0
  print("Epsilon:", eps_)

  y_target = None
  if FLAGS.attack_method == 'FastGradientMethod':
    att_params = {
        'eps': eps_,
        'clip_min': 0,
        'clip_max': 1,
        'y_target': y_target
    }
  else:
    att_params = {
        'eps': eps_,
        # 'eps_iter': eps_*1.0/FLAGS.num_iter,
        # 'eps_iter': 3.*eps_/FLAGS.num_iter,
        'eps_iter': 2. / 256.,
        'clip_min': 0,
        'clip_max': 1,
        'nb_iter': FLAGS.num_iter,
        'y_target': y_target
    }
  if FLAGS.attack_method == "Adaptive":
    att_params["y"] = y_place
    att_params['eps_iter'] = 0.03 / 256.

  print("att_params", att_params)
  if FLAGS.attack_method != "Adaptive":
    import cleverhans.attacks
    from fgm_patched import fgm_patched
    cleverhans.attacks.fgm = fgm_patched
    print("patched fgm function")

  adv_x = tf.stop_gradient(att.generate(x_place, **att_params))
  bin_adv_preds = binarized_model(adv_x)

  def _model_forward_pass(x_np, features_only=False, features_and_logits=False):
    x_np = np.transpose(x_np, (0, 2, 3, 1))

    if features_only:
      f = sess.run(final_features, {model_input: x_np})

      return f
    elif features_and_logits:
      f, l = sess.run((final_features,
                       clean_preds), {model_input: x_np})
      f = np.stack(f, 1)
      return f, l
    else:
      l = sess.run(clean_preds, {model_input: x_np})
      return l

  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=_model_forward_pass,
      logit_forward_and_backward_pass=None
  )

  test_loader = build_dataloader_from_arrays(x_test.transpose((0, 3, 1, 2)),
                                             y_test,
                                             batch_size=FLAGS.batch_size)

  from argparse_utils import DecisionBoundaryBinarizationSettings
  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        attack_fn=partial(run_attack, preds=bin_adv_preds, sess=sess, x_ph=x_place,
                          y_ph=y_place, x_adv=adv_x),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=eps_,
            norm="linf",
            lr=10000,
            n_boundary_points=FLAGS.n_boundary_points,
            n_inner_points=FLAGS.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=FLAGS.n_samples,
        device="cpu",
        batch_size=FLAGS.batch_size,
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        train_classifier_fn=partial(
            train_classifier,
            linear_layer=linear_layer,
            clean_preds=bin_clean_preds,
            x_ph=x_place,
            sess=sess,
            binarized_model_wrapper=binarized_model_wrapper
        ),
        fail_on_exception=False,
        rescale_logits=None,
        sample_training_data_from_corners=FLAGS.sample_from_corners,
        # decision_boundary_closeness=0.9999,
    )
  print(format_result(scores_logit_differences_and_validation_accuracies,
                      FLAGS.n_samples))


if __name__ == "__main__":
  main()
