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
import time
import logging
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
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
from attack_model import Attack, CW
import scipy
from ml_loo import generate_ml_loo_features


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

	parser.add_argument("--batch-size", default=500, type=int)

	parser.add_argument(
		'--det', 
		type = str, 
		choices = ['ml_loo'], 
		default = 'ml_loo'
	)

	args = parser.parse_args()
	dict_a = vars(args) 
	data_model = args.dataset_name + args.model_name

	print('Loading dataset...') 
	dataset = ImageData(args.dataset_name)
	model = ImageModel(args.model_name, args.dataset_name, train = False, load = True)

	###########################################################
	# Loading original, adversarial and noisy samples
	###########################################################
	
	print('Loading original, adversarial and noisy samples...')
	X_test = np.load('{}/data/{}_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	X_test_adv = np.load('{}/data/{}_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	X_train = np.load('{}/data/{}_train_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	X_train_adv = np.load('{}/data/{}_train_adv_{}_{}.npy'.format(data_model, args.data_sample, args.attack, 'ori'))

	Y_test = model.predict(X_test)
	print("X_test_adv: ", X_test_adv.shape)

	
	x = {
		'train': {
			'original': X_train, 
			'adv': X_train_adv, 
			},
		'test': {
			'original': X_test, 
			'adv': X_test_adv, 
			},
	}
	#################################################################
	# Extracting features for original, adversarial and noisy samples
	#################################################################
	cat = {'original':'ori', 'adv':'adv', 'noisy':'noisy'}
	dt = {'train':'train', 'test':'test'}

	if args.det in ['ml_loo']:
		if args.model_name == 'resnet':
			interested_layers = [14,24,35,45,56,67,70]

		print('extracting layers ', interested_layers)
		reference = - dataset.x_train_mean

		combined_features = generate_ml_loo_features(args, data_model, reference, model, x, interested_layers, batch_size=args.batch_size)

		for data_type in ['test', 'train']:
			for category in ['original', 'adv']:
				np.save('{}/data/{}_{}_{}_{}_{}.npy'.format(
					data_model,
					args.data_sample,
					dt[data_type],
					cat[category],
					args.attack, 
					args.det), 
					combined_features[data_type][category])

	