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

try:
	import cPickle as pickle 
except:
	import _pickle as pickle
import logging
import os
import numpy as np
import time
import os
import math
import pickle as pkl 

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  
from matplotlib.patches import Rectangle 
from sklearn.preprocessing import scale, MinMaxScaler, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.stats import pearsonr
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score


color_dict = {
	'ml_loo': 'red',
}

linestyles = {
	'ml_loo': '-',
}

labels = {
	'ml_loo': 'ML-LOO',
}

labels_attack = { 
	'cw': 'C&W',
	'bim': 'PGD',
	'bim2': 'PGD (fewer steps)',
}

labels_data = {
	'cifar10': 'CIFAR-10',
}

labels_model = {
	'resnet': 'ResNet',
}


def load_data(args, attack, det, magnitude = 0.0):
	x, y = {}, {}
	for data_type in ['train', 'test']:
		if det == 'ml_loo':
			data_ori = np.load('{}/data/{}_{}_{}_{}_{}.npy'.format(
				args.data_model,
				args.data_sample,
				data_type,
				'ori',
				attack, 
				'ml_loo'))

			data_adv = np.load('{}/data/{}_{}_{}_{}_{}.npy'.format(
				args.data_model,
				args.data_sample,
				data_type,
				'adv',
				attack, 
				'ml_loo'))
			d = len(data_ori)
			print('detect using {}'.format(det))
			print('using adv only')
			print('data_ori', data_ori.shape) 
			# Use only IQR features (5th dimension).
			data_ori = data_ori[:, [5], :]
			data_adv = data_adv[:, [5], :]
			data_ori = data_ori.reshape(d, -1)
			data_adv = data_adv.reshape(d, -1)
			# (200, 1)

		d = len(data_ori)
		x[data_type] = np.vstack([data_ori, data_adv])
		y[data_type] = np.concatenate((np.zeros(d), np.ones(d)))


	idx_train = np.random.permutation(len(x['train']))
	x['train'] = x['train'][idx_train]  
	y['train'] = y['train'][idx_train]
	return x, y


def train_and_evaluate(args, detections, attack, fpr_upper = 1.0):
	plt.figure(figsize = (10, 8))
	font = {'weight': 'bold', 'size': 16}
	matplotlib.rc('font', **font)

	auc_dict = {}
	tpr1 = {}
	tpr5 = {}
	tpr10 = {}

	for det in detections:
		# Load data
		x, y = load_data(args, attack, det)
		x_train, y_train = x['train'], y['train']
		x_test, y_test = x['test'], y['test']

		x_train = x_train.reshape(len(x_train), -1)
		x_test = x_test.reshape(len(x_test), -1)
		# Train
		# TODO: verify this. max_iter=10000 was added by AUTHOR/zimmerrol
		lr = LogisticRegressionCV(n_jobs=-1, max_iter=10000).fit(x_train, y_train)

		with open(f"{args.data_model}/models/{det}_{attack}_lr.pkl", "wb") as f:
			pickle.dump(lr, f)

		# Predict
		pred = lr.predict_proba(x_test)[:, 1] 
		pred = lr.predict_proba(x_test)[:, 1] 
		# Evaluate.
		fpr, tpr, thresholds = roc_curve(y_test, pred, drop_intermediate=False)
		def find_nearest(array, value):
			array = np.asarray(array)
			idx = (np.abs(array - value)).argmin()
			return idx
		auc_dict[det] = auc(fpr, tpr)
		tpr1[det] = tpr[find_nearest(fpr, 0.01)]
		tpr5[det] = tpr[find_nearest(fpr, 0.05)]
		tpr10[det] = tpr[find_nearest(fpr, 0.10)]
		plt.plot(
			fpr, tpr,
			label="{0} (AUC: {1:0.3f})".format(labels[det], auc(fpr, tpr)),
			color=color_dict[det], 
			linestyle=linestyles[det], 
			linewidth=4)
		print("Threshold for FPR1:", thresholds[find_nearest(fpr, 0.01)])
		print("Threshold for FPR5:", thresholds[find_nearest(fpr, 0.05)])
		print("Threshold for FPR10:", thresholds[find_nearest(fpr, 0.10)])

	# 0.2 was fpr_upper before
	plt.xlim([0.0, 0.2])
	plt.ylim([0.0, 1.0])
	plt.xlabel('False Positive Rate', fontsize = 32)
	plt.ylabel('True Positive Rate', fontsize = 32)
	plt.title('{} ({}, {})'.format(labels_attack[attack], labels_data[args.dataset_name], labels_model[args.model_name]), fontsize = 32) 
	plt.legend(loc="lower right", fontsize = 22)
	plt.show()
	figure_name = '{}/figs/mad_transfer_roc_{}_{}_{}.pdf'.format(args.data_model, args.data_sample, attack, attack)
	plt.savefig(figure_name)


	return auc_dict, tpr1, tpr5, tpr10 


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
		choices = ['x_val200'], 
		default = 'x_val200')
	parser.add_argument(
			'--attack',
			type = str,
			choices = ['cw', 'bim', 'bim2'],
			default = 'cw'
	)

	args = parser.parse_args()
	dict_a = vars(args) 
	args.data_model = args.dataset_name + args.model_name

	train_and_evaluate(args, ['ml_loo'], args.attack, fpr_upper = 1.0)







