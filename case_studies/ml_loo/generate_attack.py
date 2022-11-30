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
from build_model import ImageModel 
from load_data import ImageData, split_data
import pickle as pkl
from keras.utils import to_categorical
from attack_model import Attack, CW, BIM
import scipy


if __name__ == '__main__':
	import argparse
	parser = argparse.ArgumentParser()

	parser.add_argument('--dataset_name', type = str, 
		choices = ['cifar10'], 
		default = 'cifar10')

	parser.add_argument('--model_name', type = str, 
		choices = ['resnet'], 
		default = 'resnet') 

	parser.add_argument('--data_sample', type = str, 
		choices = ['x_train', 'x_val', 'x_val200'], 
		default = 'x_val200')

	parser.add_argument(
			'--attack',
			type = str,
			choices = ['cw', 'bim', 'bim2'],
			default = 'cw'
	)

	args = parser.parse_args()
	dict_a = vars(args) 
	data_model = args.dataset_name + args.model_name

	if data_model not in os.listdir('./'):	
		os.mkdir(data_model)
	if 'results' not in os.listdir('./{}'.format(data_model)):
		os.mkdir('{}/results'.format(data_model))
	if 'models' not in os.listdir(data_model):
		os.mkdir('{}/models'.format(data_model))
	if 'data' not in os.listdir(data_model):
		os.mkdir('{}/data'.format(data_model))
	if 'figs' not in os.listdir(data_model):
		os.mkdir('{}/figs'.format(data_model))

	print('Loading dataset...') 
	dataset = ImageData(args.dataset_name)
	model = ImageModel(args.model_name, args.dataset_name, train = False, load = True)

	if args.dataset_name == 'cifar10':
		X_train, Y_train, X_test, Y_test = split_data(dataset.x_val,
			dataset.y_val, model, num_classes = 10, 
			split_rate = 0.8, sample_per_class = 1000)


	print('Sanity checking...')
	data_sample = X_test
	print('data_sample.shape', data_sample.shape)
	print('X_train.shape', X_train.shape)

	pred_test = model.predict(dataset.x_val)
	def cross_entropy(predictions, targets, epsilon=1e-12):
		predictions = np.clip(predictions, epsilon, 1. - epsilon)
		N = predictions.shape[0]
		ce = -np.sum(targets*np.log(predictions+1e-9))/N
		return ce

	ce = cross_entropy(pred_test, dataset.y_val, epsilon=1e-12)
	acc = np.mean(np.argmax(pred_test, axis = 1) == np.argmax(dataset.y_val, axis = 1))
	print('The accuracy is {}. The cross entropy is {}.'.format(acc, ce))


	if args.attack == 'cw':
		if args.dataset_name in ['cifar10']:
			if args.model_name == 'resnet':
				attack_model = CW(
					KerasModelWrapper(model.model),
					model.input_ph,
					model.num_classes,
					source_samples = 100,
					binary_search_steps = 5, 
					cw_learning_rate = 1e-2, 
					confidence = 0, 
					attack_iterations = 100, 
					attack_initial_const = 1e-2,
				)
	elif args.attack == "bim":
		if args.dataset_name in ['cifar10']:
			if args.model_name == 'resnet':
				attack_model = BIM(
						KerasModelWrapper(model.model),
						model.sess,
						model.input_ph,
						model.num_classes,
						attack_iterations = 100,
						epsilon=0.03,
						learning_rate=2.5 * 0.03 / 100,
						random_init=True
				)
	elif args.attack == "bim2":
		if args.dataset_name in ['cifar10']:
			if args.model_name == 'resnet':
				attack_model = BIM(
						KerasModelWrapper(model.model),
						model.sess,
						model.input_ph,
						model.num_classes,
						attack_iterations = 10,
						epsilon=0.03,
						learning_rate=2.5 * 0.03 / 10,
						random_init=True
				)


	###################################################
	# filter data samples with correct predictions by model and successsful attacks
	###################################################

	data_types = ['train', 'test']
	data = {'train': (X_train, Y_train), 'test': (X_test, Y_test)}
	if args.data_sample == 'x_val200':
		num_samples = {'train': 800, 'test': 200}

	for data_type in data_types:
		x, y = data[data_type]
		print('x.shape', x.shape)
		print('y.shape', y.shape)
		num_successes = 0
		oris = []
		perturbeds = []


		batch_size = int(np.minimum(100, num_samples[data_type]))
		cur_batch = 0
		conf = 15
		epsilon = 0
		while num_successes < num_samples[data_type]:
			batch_x, batch_y = x[cur_batch * batch_size:(cur_batch+1) * batch_size], y[cur_batch * batch_size:(cur_batch+1) * batch_size]

			print('batch_x', batch_x.shape)
			x_adv = attack_model.attack(batch_x)
	
			print('x_adv', x_adv.shape)
			if x_adv.shape[0] == 0:
				continue
			x_adv_labels = np.argmax(model.predict(x_adv), axis = -1)
			index_filter = (x_adv_labels != np.argmax(batch_y, axis = 1))
			ori = batch_x[index_filter]
			perturbed = x_adv[index_filter]
			print('Success rate', perturbed.shape[0] / len(x_adv))
			oris.append(ori)
			perturbeds.append(perturbed)

			cur_batch += 1
			num_successes += len(ori)
			print('Number of successsful samples is {}'.format(num_successes))

		oris = np.concatenate(oris, axis = 0)
		perturbeds = np.concatenate(perturbeds, axis = 0)

		oris = oris[:num_samples[data_type]]
		perturbeds = perturbeds[:num_samples[data_type]]

		print('oris.shape', oris.shape)
		print('perturbeds.shape', perturbeds.shape)

		np.save('{}/data/{}{}_{}_{}.npy'.format(
			data_model, 
			args.data_sample,
			'' if data_type == 'test' else '_train',
			args.attack, 
			'ori'), 
			oris)
		np.save('{}/data/{}{}_adv_{}_{}.npy'.format(
			data_model, 
			args.data_sample,
			'' if data_type == 'test' else '_train',
			args.attack, 
			'ori'), 
			perturbeds)

