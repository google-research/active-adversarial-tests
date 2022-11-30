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
from __future__ import unicode_literals

import logging

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from active_tests.decision_boundary_binarization import LogitRescalingType
from active_tests.decision_boundary_binarization import \
  _train_logistic_regression_classifier
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack, format_result
from cifar10_attack import setup_model

logging.getLogger('tensorflow').setLevel(logging.FATAL)
from functools import partial
import tensorflow as tf
from keras.utils.np_utils import to_categorical

tf.logging.set_verbosity(tf.logging.ERROR)

import os
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

class Layer(object):

  def get_output_shape(self):
    return self.output_shape


class Linear(Layer):

  def __init__(self, num_hid, name, useBias=False):
    self.__dict__.update(locals())
    # self.num_hid = num_hid

  def set_input_shape(self, input_shape, reuse):

    # with tf.variable_scope(self.scope_name+ 'init', reuse): # this works
    # with black box, but now can't load checkpoints from wb
    # this works with white-box
    with tf.variable_scope(self.name + '_init', reuse):

      batch_size, dim = input_shape
      self.input_shape = [batch_size, dim]
      self.output_shape = [batch_size, self.num_hid]
      if self.useBias:
        self.bias_shape = self.num_hid
      init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
      init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                 keep_dims=True))
      self.W = tf.get_variable(
          "W", initializer=init)

      if self.useBias:
        bias_init = tf.zeros(self.bias_shape)
        self.bias = tf.get_variable("b", initializer= bias_init)

        self.bias_ph = tf.placeholder(tf.float32, shape=self.bias_shape)
        self.set_bias = self.bias.assign(self.bias_ph)

      self.W_ph = tf.placeholder(tf.float32, shape=[dim, self.num_hid])
      self.set_weight = self.W.assign(self.W_ph)

  def fprop(self, x, reuse):

    # with tf.variable_scope(self.scope_name + '_fprop', reuse):
    # this works with white-box
    with tf.variable_scope(self.name + '_fprop', reuse):

      x = tf.matmul(x, self.W)  # + self.b
      if self.useBias:
        x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))

      return x

FLAGS = flags.FLAGS

ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4
MAX_BATCH_SIZE = 100
MAX_BATCH_SIZE = 100

# Scaling input to softmax
INIT_T = 1.0
# ATTACK_T = 1.0
ATTACK_T = 0.25

from cifar10_attack import data_cifar10

from cifar10_attack import build_adversarial_attack


def main(argv=None):
  """
  CIFAR10 modified_cleverhans tutorial
  :return:
  """

  nb_classes = 2
  targeted = True if FLAGS.targeted else False
  batch_size = FLAGS.batch_size
  nb_samples = FLAGS.nb_samples
  eps = FLAGS.eps

  attack = FLAGS.attack
  nb_iter = FLAGS.nb_iter

  ensembleThree = True if FLAGS.ensembleThree else False
  sess, model, preds, x, y, phase = setup_model()

  # Get CIFAR10 test data
  X_train, Y_train, X_test, Y_test = data_cifar10()
  del X_train, Y_train
  X_test = np.transpose(X_test, (0, 3, 1, 2))
  print(X_test.shape)

  def run_attack(m, l, sess, attack):
    for x_batch, y_batch in l:
      assert len(x_batch) == 1
      x_batch = x_batch.cpu().numpy()
      y_batch = y_batch.cpu().numpy()

      x_batch = x_batch.transpose(0, 2, 3, 1)

      y_batch_oh = to_categorical(y_batch, num_classes=2)
      x_batch_adv = attack(x_batch, y_batch_oh)

      probs = m(x_batch_adv)
      preds = probs.argmax(-1)

      is_adv = preds != y_batch

      return is_adv, (torch.tensor(x_batch_adv.transpose(0, 3, 1, 2), dtype=torch.float32),\
             torch.tensor(probs, dtype=torch.float32))


  def train_classifier(
      n_features: int,
      train_loader: DataLoader,
      raw_train_loader: DataLoader,
      logits: torch.Tensor,
      device: str,
      rescale_logits: LogitRescalingType,
      binarized_ensemble,
      set_weight_ops,
      set_bias_ops,
      sess,
      weights_phs,
      biases_phs
  ):
    #del raw_train_loader

    # fit a linear readout for each of the submodels of the ensemble
    assert len(train_loader.dataset.tensors[0].shape) == 3
    assert train_loader.dataset.tensors[0].shape[1] == len(weights_phs) == len(
      biases_phs)

    classifier_weights = []
    classifier_biases = []
    for i in range(3):
      x_ = train_loader.dataset.tensors[0][:, i]
      y_ = train_loader.dataset.tensors[1]

      cls = _train_logistic_regression_classifier(
          n_features,
          DataLoader(TensorDataset(x_, y_), batch_size=train_loader.batch_size),
          logits[:, i] if logits is not None else None,
          "sklearn",
          10000,
          device,
          n_classes=2,
          rescale_logits=rescale_logits
      )
      classifier_weights.append(cls.weight.data.cpu().numpy().transpose())
      classifier_biases.append(cls.bias.data.cpu().numpy())

    # update weights of the binary models
    for op, ph, v in zip(set_weight_ops, weights_phs, classifier_weights):
      sess.run(op, {ph: v})
    for op, ph, v in zip(set_bias_ops, biases_phs, classifier_biases):
      sess.run(op, {ph: v})

    """ n_corr1 = 0
    n_corr2 = 0
    n_total = 0
    for x, y in raw_train_loader:
      preds1 = binarized_model(x)
      preds2 = binarized_model(x, averaged=False)
      import pdb; pdb.set_trace()
      n_corr1 += (preds1 == y).sum()
      n_corr2 += (preds2 == y).sum()
      n_total += len(x)
    """
    return binarized_ensemble

  from tensorflow_wrapper import TensorFlow1ToPyTorchWrapper, \
    PyTorchToTensorFlow1Wrapper
  from utils import build_dataloader_from_arrays

  test_loader = build_dataloader_from_arrays(X_test, Y_test, batch_size=32)

  from modified_cleverhans.model import Model
  class BinarizedEnsembleModel(Model):
    def __init__(self, base_classifier, input_ph):
      self.num_classes = 2

      self.base_classifier = base_classifier
      self.layer_names = []
      self.layer_names.append('combined_features')
      self.layer_names.append('combined_logits')

      combined_layer_name = 'combined' ## Gives the final class prediction based on max voting
      self.layer_names.append(combined_layer_name)
      combinedCorrectProb_layer_name = 'combinedAvgCorrectProb' ## Gives average probability values of the models that decided the final prediction
      self.layer_names.append(combinedCorrectProb_layer_name)
      combinedProb_layer_name = 'combinedAvgProb' ## Gives average probability values of all the models
      self.layer_names.append(combinedProb_layer_name)

      self.readout_1 = Linear(2, "binarized_ensemble_readout_1", useBias=True)
      self.readout_2 = Linear(2, "binarized_ensemble_readout_2", useBias=True)
      self.readout_3 = Linear(2, "binarized_ensemble_readout_3", useBias=True)

      self.readout_1.set_input_shape((-1, 64), True)
      self.readout_2.set_input_shape((-1, 64), True)
      self.readout_3.set_input_shape((-1, 64), True)

      self.set_weight_ops = [
          self.readout_1.set_weight,
          self.readout_2.set_weight,
          self.readout_3.set_weight
      ]
      self.set_bias_ops = [
          self.readout_1.set_bias,
          self.readout_2.set_bias,
          self.readout_3.set_bias,
      ]

      self.weights_phs = [
          self.readout_1.W_ph,
          self.readout_2.W_ph,
          self.readout_3.W_ph
      ]
      self.biases_phs = [
          self.readout_1.bias_ph,
          self.readout_2.bias_ph,
          self.readout_3.bias_ph
      ]

      self.input_ph = input_ph
      self.ensemble_op = self.get_ensemblepreds(self.input_ph)
      self.averaged_op = self.get_combinedAvgCorrectProbs(self.input_ph)


    def __call__(self, x_, averaged=True, *args, **kwargs):
      return_torch = False
      return_numpy = False
      if isinstance(x_, torch.Tensor):
        x_ = x_.cpu().numpy()
        return_torch = True
      if isinstance(x_, np.ndarray):
        return_numpy = True
        if x_.shape[1] == 3:
          x_ = x_.transpose(0, 2, 3, 1)

        x = self.input_ph
        if averaged:
          op = self.averaged_op
        else:
          op = self.ensemble_op

      else:
        raise NotImplementedError("Calling this with a tf tensor is not supported yet"
                                  " (wasn't necessary).")
        #if averaged:
        #  op = self.get_combinedAvgCorrectProbs(x_, *args, **kwargs)
        #else:
        #  op = self.get_ensemblepreds(x_, *args, **kwargs)
      if return_numpy or return_torch:
        x_ = sess.run(op, {x: x_})
        if return_torch:
          x_ = torch.tensor(x_, dtype=torch.float32)
        return x_

    def fprop(self, x, reuse):
      base_states = self.base_classifier.fprop(x, reuse)

      features1 = base_states["Model1_HiddenLinear10"]
      features2 = base_states["Model2_HiddenLinear10"]
      features3 = base_states["Model3_HiddenLinear10"]

      output1 = self.readout_1.fprop(features1, reuse)
      output2 = self.readout_2.fprop(features2, reuse)
      output3 = self.readout_3.fprop(features3, reuse)

      states = []
      states.append(tf.stack((features1, features2, features3), 1))
      states.append(tf.stack((output1, output2, output3), 1))

      # Find class predictions with each model
      pred1 = tf.argmax(output1, axis=-1)
      pred2 = tf.argmax(output2, axis=-1)
      pred3 = tf.argmax(output3, axis=-1)
      comb_pred = tf.stack([pred1, pred2, pred3], axis=1)
      comb_pred = tf.cast(comb_pred, dtype=tf.int32) # converting to int32 as bincount requires int32

      # Find how many times each of the classes are predicted among the three models and identify the max class
      initial_imidx = 1

      binarray = tf.bincount(comb_pred[0], minlength=self.num_classes)# initial bincount, counts number of occurences of each integer from 0 to 10 for the 1d array, returns a 1d array
      max_class = tf.argmax(binarray, axis=-1)
      count_max = tf.gather(binarray, max_class) # max vote count for a class

      value = tf.cond(tf.less(count_max, 2), lambda: pred3[0], lambda: max_class)
      in_class_array = tf.fill([1], value)

      ## Added below to allow better gradient calculation for max voted model
      in_avgCorrectprob = tf.cond(tf.equal(value, pred3[0]), lambda: output3[0], lambda: tf.zeros_like(output3[0])) # add pred3 if it affected the final decision
      in_avgCorrectprob = tf.cond(tf.equal(value, pred2[0]), lambda: tf.add(output2[0], in_avgCorrectprob), lambda: in_avgCorrectprob) # add pred2 if it affected the final decision
      in_avgCorrectprob = tf.cond(tf.equal(value, pred1[0]), lambda: tf.add(output1[0], in_avgCorrectprob), lambda: in_avgCorrectprob) # add pred2 if it affected the final decision
      in_avgCorrectprob_array = tf.expand_dims(tf.div(in_avgCorrectprob, tf.cast(count_max, dtype=tf.float32)), 0)

      #condition check: when true the loop body executes
      def idx_loop_condition(class_array, avgCorrectprob_array, im_idx):
        return tf.less(im_idx, tf.shape(pred1)[0])

      #loop body to calculate the max voted class for each image
      def idx_loop_body(class_array, avgCorrectprob_array, im_idx):
        binarray_new = tf.bincount(comb_pred[im_idx], minlength=self.num_classes) # counts number of occurences of each integer from 0 to 10 for the 1d array, returns a 1d array
        max_class = tf.argmax(binarray_new, axis=-1)
        count_max = tf.gather(binarray_new, max_class) # max vote count for a class

        value = tf.cond(tf.less(count_max, 2), lambda: pred3[im_idx], lambda: max_class)# If the max vote is less than 2, take the prediction of the full precision model
        new_array = tf.fill([1], value)
        class_array = tf.concat([class_array, new_array], 0)

        ## Added below to allow better gradient calculation for max voted model
        avgCorrectprob = tf.cond(tf.equal(value, pred3[im_idx]), lambda: output3[im_idx], lambda: tf.zeros_like(output3[im_idx])) # add pred3 if it affected the final decision
        avgCorrectprob = tf.cond(tf.equal(value, pred2[im_idx]), lambda: tf.add(output2[im_idx], avgCorrectprob), lambda: avgCorrectprob) # add pred2 if it affected the final decision
        avgCorrectprob = tf.cond(tf.equal(value, pred1[im_idx]), lambda: tf.add(output1[im_idx], avgCorrectprob), lambda: avgCorrectprob) # add pred2 if it affected the final decision
        avgCorrectprob = tf.expand_dims(tf.div(avgCorrectprob, tf.cast(count_max, dtype=tf.float32)), 0)
        avgCorrectprob_array = tf.concat([avgCorrectprob_array, avgCorrectprob], 0)

        return (class_array, avgCorrectprob_array, im_idx+1)

      res = tf.while_loop(
          cond=idx_loop_condition,
          body=idx_loop_body,
          loop_vars=[in_class_array, in_avgCorrectprob_array, initial_imidx],
          shape_invariants=[tf.TensorShape([None]), tf.TensorShape([None, self.num_classes]), tf.TensorShape([])], #add shape invariant saying that the first dimension of in_class_array changes and is thus None
      )

      pred_output = tf.cast(res[0], dtype=tf.int64) # no. of times each class is predicted for all images
      states.append(pred_output)

      avgCorrectprob_output = res[1] # no. of times each class is predicted for all images
      states.append(avgCorrectprob_output)

      avgprob = tf.div(tf.add_n([output2, output1, output3]), tf.cast(3, dtype=tf.float32)) # Average probability across all models
      states.append(avgprob)

      states = dict(zip(self.get_layer_names(), states))
      return states

  binarized_model = BinarizedEnsembleModel(model, x)
  attacker, attack_params = build_adversarial_attack(
      sess, binarized_model, attack,
      targeted, nb_classes,
      ensembleThree,
      nb_samples, nb_iter, eps,
      robust_attack=FLAGS.robust_attack)

  base_model_outputs = model.fprop(x, reuse=True)
  base_model_features = base_model_outputs["combined_features"]
  base_model_logits = base_model_outputs["combined_logits"]
  def _model_forward_pass(x_np, features_only=False, features_and_logits=False):
    x_np = np.transpose(x_np, (0, 2, 3, 1))

    if features_only:
      return sess.run(base_model_features, {x : x_np})
    elif features_and_logits:
      targets = [base_model_features, base_model_logits]
      return tuple(sess.run(targets, {x : x_np}))
    else:
      return sess.run(base_model_logits, {x : x_np})


  feature_extractor = TensorFlow1ToPyTorchWrapper(
      logit_forward_pass=_model_forward_pass,
      logit_forward_and_backward_pass=None
  )
  y = tf.placeholder(tf.float32, shape=(None, 2))
  if FLAGS.use_labels:
    attack_params['y'] = y
  else:
    #del attack_params['y']
    attack_params['y'] = tf.stop_gradient(tf.to_float(tf.one_hot(binarized_model.get_ensemblepreds(x, reuse=True), nb_classes)))
  x_adv = attacker.generate(x, phase, **attack_params)

  from argparse_utils import DecisionBoundaryBinarizationSettings
  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        feature_extractor,
        test_loader,
        attack_fn=lambda m, l, kwargs: run_attack(
            m, l, sess, lambda x_, y_: sess.run(x_adv, {x: x_, y: y_})
        ),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=FLAGS.eps,
            norm="linf",
            lr=10000,
            n_boundary_points=FLAGS.n_boundary_points,
            n_inner_points=FLAGS.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=FLAGS.nb_samples,
        device="cpu",
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        train_classifier_fn=partial(train_classifier,
                                    binarized_ensemble=binarized_model,
                                    set_weight_ops=binarized_model.set_weight_ops,
                                    set_bias_ops=binarized_model.set_bias_ops,
                                    sess=sess,
                                    weights_phs=binarized_model.weights_phs,
                                    biases_phs=binarized_model.biases_phs,
                                    ),
        fail_on_exception=True,
        rescale_logits="adaptive",
        decision_boundary_closeness=0.9999,
        sample_training_data_from_corners=FLAGS.sample_from_corners
    )
  print(format_result(scores_logit_differences_and_validation_accuracies,
                      FLAGS.nb_samples))

if __name__ == '__main__':
  par = argparse.ArgumentParser()

  # Generic flags
  par.add_argument('--gpu', help='id of GPU to use')
  par.add_argument('--model_path', help='Path to save or load model')
  par.add_argument('--data_dir', help='Path to training data',
                   default='cifar10_data')

  # Architecture and training specific flags
  par.add_argument('--nb_epochs', type=int, default=6,
                   help='Number of epochs to train model')
  par.add_argument('--nb_filters', type=int, default=32,
                   help='Number of filters in first layer')
  par.add_argument('--batch_size', type=int, default=128,
                   help='Size of training batches')
  par.add_argument('--learning_rate', type=float, default=0.001,
                   help='Learning rate')
  par.add_argument('--rand', help='Stochastic weight layer?',
                   action="store_true")

  # Attack specific flags
  par.add_argument('--eps', type=float, default=0.1,
                   help='epsilon')
  par.add_argument('--attack', type=int, default=0,
                   help='Attack type, 0=CW, 2=FGSM')
  par.add_argument('--nb_samples', type=int,
                   default=10000, help='Nb of inputs to attack')
  par.add_argument(
      '--targeted', help='Run a targeted attack?', action="store_true")
  # Adversarial training flags
  par.add_argument(
      '--adv', help='Adversarial training type?', type=int, default=0)
  par.add_argument('--delay', type=int,
                   default=10, help='Nb of epochs to delay adv training by')
  par.add_argument('--nb_iter', type=int,
                   default=40,
                   help='Nb of iterations of PGD (set to 50 for CW)')

  # EMPIR specific flags
  par.add_argument('--lowprecision', help='Use other low precision models',
                   action="store_true")
  par.add_argument('--wbits', type=int, default=0,
                   help='No. of bits in weight representation')
  par.add_argument('--abits', type=int, default=0,
                   help='No. of bits in activation representation')
  par.add_argument('--wbitsList', type=int, nargs='+',
                   help='List of No. of bits in weight representation for different layers')
  par.add_argument('--abitsList', type=int, nargs='+',
                   help='List of No. of bits in activation representation for different layers')
  par.add_argument('--stocRound',
                   help='Stochastic rounding for weights (only in training) and activations?',
                   action="store_true")
  par.add_argument('--model_path1',
                   help='Path where saved model1 is stored and can be loaded')
  par.add_argument('--model_path2',
                   help='Path where saved model2 is stored and can be loaded')
  par.add_argument('--ensembleThree',
                   help='Use an ensemble of full precision and two low precision models that can be attacked directly',
                   action="store_true")
  par.add_argument('--model_path3',
                   help='Path where saved model3 in case of combinedThree model is stored and can be loaded')
  par.add_argument('--wbits2', type=int, default=0,
                   help='No. of bits in weight representation of model2, model1 specified using wbits')
  par.add_argument('--abits2', type=int, default=0,
                   help='No. of bits in activation representation of model2, model2 specified using abits')
  par.add_argument('--wbits2List', type=int, nargs='+',
                   help='List of No. of bits in weight representation for different layers of model2')
  par.add_argument('--abits2List', type=int, nargs='+',
                   help='List of No. of bits in activation representation for different layers of model2')
  # extra flags for defensive distillation
  par.add_argument('--distill', help='Train the model using distillation',
                   action="store_true")
  par.add_argument('--student_epochs', type=int, default=50,
                   help='No. of epochs for which the student model is trained')
  # extra flags for input gradient regularization
  par.add_argument('--inpgradreg',
                   help='Train the model using input gradient regularization',
                   action="store_true")
  par.add_argument('--l2dbl', type=int, default=0,
                   help='l2 double backprop penalty')
  par.add_argument('--l2cs', type=int, default=0,
                   help='l2 certainty sensitivity penalty')


  par.add_argument("--n-inner-points", default=999, type=int)
  par.add_argument("--n-boundary-points", default=1, type=int)

  par.add_argument("--robust-attack", action="store_true")
  par.add_argument("--use-labels", action="store_true")
  par.add_argument("--sample-from-corners", action="store_true")

  FLAGS = par.parse_args()

  import cifar10_attack
  cifar10_attack.FLAGS = FLAGS

  if FLAGS.gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  tf.app.run()
