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

"""
This tutorial shows how to generate adversarial examples
using C&W attack in white-box setting.
The original paper can be found at:
https://nicholas.carlini.com/papers/2017_sp_nnrobustattacks.pdf
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
from keras.layers import Input
from cleverhans.attacks import CarliniWagnerL2
from cleverhans.dataset import MNIST
from cleverhans.loss import CrossEntropy
from cleverhans.utils import grid_visual, AccuracyReport
from cleverhans.utils import set_log_level
from cleverhans.utils_tf import model_eval, tf_model_load
from cleverhans.train import train
from cleverhans.utils_keras import KerasModelWrapper
import pickle as pkl
from keras import backend as K

from cleverhans.attacks import CarliniWagnerL2, FastGradientMethod, SaliencyMapMethod, DeepFool, BasicIterativeMethod

class Attack(object):
	def __init__(self, model, *args):
		self.model = model

	def attack(self, x, *args):
		raise NotImplementedError

class CleverhansAttackFeedableRunMixin:
	def generate_np(self, x_val, feedable_dict={}, **kwargs):
		"""
    Generate adversarial examples and return them as a NumPy array.
    Sub-classes *should not* implement this method unless they must
    perform special handling of arguments.
    :param x_val: A NumPy array with the original inputs.
    :param **kwargs: optional parameters used by child classes.
    :return: A NumPy array holding the adversarial examples.
    """

		if self.sess is None:
			raise ValueError("Cannot use `generate_np` when no `sess` was"
											 " provided")

		packed = self.construct_variables(kwargs)
		fixed, feedable, _, hash_key = packed

		if hash_key not in self.graphs:
			self.construct_graph(fixed, feedable, x_val, hash_key)
		else:
			# remove the None arguments, they are just left blank
			for k in list(feedable.keys()):
				if feedable[k] is None:
					del feedable[k]

		x, new_kwargs, x_adv = self.graphs[hash_key]

		feed_dict = {x: x_val}

		for name in feedable:
			feed_dict[new_kwargs[name]] = feedable[name]

		return self.sess.run(x_adv, {**feed_dict, **feedable_dict})

class FeedableRunCarliniWagnerL2(CleverhansAttackFeedableRunMixin, CarliniWagnerL2):
	pass


class FeedableRunBasicIterativeMethod(CleverhansAttackFeedableRunMixin, BasicIterativeMethod):
	pass


class CW(Attack):
	def __init__(self, model, sess, input_ph, num_classes, source_samples = 2, binary_search_steps = 5, cw_learning_rate = 5e-3, confidence = 0, attack_iterations = 1000, attack_initial_const = 1e-2):
		super(Attack, self).__init__()
		
		self.model = model
		self.sess = sess

		self.x = input_ph
		self.y = Input(shape=(num_classes,), dtype = 'float32')

		abort_early = True
		self.cw = FeedableRunCarliniWagnerL2(self.model, sess=self.sess)
		self.cw_params = {
			'binary_search_steps': binary_search_steps,
			"y": None,
			'abort_early': True,
			'max_iterations': attack_iterations,
			'learning_rate': cw_learning_rate ,
			'batch_size': source_samples,
			'initial_const': attack_initial_const ,
			'confidence': confidence,
			'clip_min': 0.0,
		}

	def attack(self, x, y = None, feedable_dict={}):
		# print(self.cw_params)
		adv = self.cw.generate_np(x, **self.cw_params, feedable_dict=feedable_dict)

		if y:
			eval_params = {'batch_size': 100}
			preds = self.model.get_logits(self.x)
			acc = model_eval(self.sess, self.x, self.y, preds, adv, y, args=eval_params)
			adv_success = 1 - acc
			print('The adversarial success rate is {}.'.format(adv_success))

		return adv


# added by AUTHOR according to the description in the paper
class BIM(Attack):
	def __init__(self, model, sess, input_ph, num_classes, epsilon=0.03, learning_rate = 5e-3, attack_iterations = 1000, random_init=True):
		super(Attack, self).__init__()

		self.model = model
		self.sess = sess

		self.x = input_ph
		self.y = Input(shape=(num_classes,), dtype = 'float32')

		self.bim = FeedableRunBasicIterativeMethod(self.model, sess=self.sess)
		self.bim_params = {
				"y": None,
				'nb_iter': attack_iterations,
				'eps_iter': learning_rate,
				'eps': epsilon,
				'rand_init': random_init,
				'clip_min': 0.0,
				'clip_max': 1.0,
		}

	def attack(self, x, y = None, feedable_dict={}):
		# print(self.bim_params)
		adv = self.bim.generate_np(x, **self.bim_params, feedable_dict=feedable_dict)

		if y:
			eval_params = {'batch_size': 100}
			preds = self.model.get_logits(self.x)
			acc = model_eval(self.sess, self.x, self.y, preds, adv, y, args=eval_params)
			adv_success = 1 - acc
			print('The adversarial success rate is {}.'.format(adv_success))

		return adv


class FMA(Attack):
	def __init__(self, raw_model, model, sess, input_ph, num_classes, target_samples,
			reference,
			features, epsilon=0.03, num_random_features=1000,
			learning_rate = 5e-3, attack_iterations = 1000, random_init=True,
		verbose=False):
		super(Attack, self).__init__()

		self.raw_model = raw_model
		self.model = model
		self.sess = sess

		self.reference = reference
		self.features = features

		assert len(target_samples) == num_classes, (len(target_samples), num_classes)
		self.target_samples = target_samples

		self.x = input_ph
		self.y = Input(shape=(num_classes,), dtype = 'float32')

		self.logits = model.get_logits(input_ph)

		self.epsilon = epsilon
		self.learning_rate = learning_rate
		self.attack_iterations = attack_iterations
		self.random_init = random_init

		self.all_features = tf.concat(features, 1)
		num_random_features = min(num_random_features, self.all_features.shape[1].value)
		self.num_random_features = num_random_features
		self.feature_indices_ph = tf.placeholder(tf.int32, shape=(num_random_features,))
		self.target_features_ph = tf.placeholder(tf.float32,
																				shape=self.all_features.shape)

		self.loss = tf.nn.l2_loss(tf.gather(self.all_features -
												 tf.expand_dims(self.target_features_ph, 0), self.feature_indices_ph, axis=1))
		self.gradient = tf.gradients(self.loss, self.x)[0]

		self.verbose = verbose

	def attack(self, x, y = None, feedable_dict={}):
		assert len(x) == 1, "attack can only process a single sample at a time"
		# print(self.bim_params)

		y = self.sess.run(self.logits, {self.x: x, **feedable_dict}).argmax(-1)[0]
		x_target = self.target_samples[(y + 1) % 10]

		from ml_loo import loo_ml_instance
		target_features = loo_ml_instance(x_target, self.reference, self.raw_model, self.features,
																		 batch_size=3100)[:, :-1]

		if not self.random_init:
			x_adv = x
		else:
			x_adv = np.clip(x + np.random.uniform(-self.epsilon, +self.epsilon, x.shape), 0, 1)

		for i in range(self.attack_iterations):
			feature_indices = np.random.choice(
					np.arange(self.all_features.shape[-1].value),
					self.num_random_features)
			loss_value, logits_value, gradient_value = self.sess.run(
					(self.loss, self.logits, self.gradient),
					{
							self.x: x_adv,
						  self.target_features_ph: target_features,
							self.feature_indices_ph: feature_indices,
							**feedable_dict
					}
			)
			gradient_value = np.sign(gradient_value)
			x_adv -= self.learning_rate * gradient_value
			delta = np.clip(x_adv - x, -self.epsilon, +self.epsilon)
			x_adv = np.clip(x + delta, 0, 1)

			if self.verbose:
				print(loss_value, logits_value)

		return x_adv
