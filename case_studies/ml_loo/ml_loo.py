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

from __future__ import absolute_import, division, print_function 

import numpy as np
import tensorflow as tf
import os
from keras.utils import to_categorical
import math
import time 
import numpy as np 
import sys
import os
import math
from build_model import ImageModel 
from load_data import ImageData, split_data
import pickle as pkl
from keras.models import Model
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist
import time
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import precision_recall_curve, roc_curve, auc, average_precision_score
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew
from scipy.spatial.distance import pdist


def con(score):
  # score (n, d)
  score = score.reshape(len(score), -1)
  score_mean = np.mean(score, -1, keepdims = True)
  c_score = score - score_mean
  c_score = np.abs(c_score)
  return np.mean(c_score, axis = -1)


def mad(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    median = np.median(d)
    abs_dev = np.abs(d - median)
    med_abs_dev = np.median(abs_dev)
    pd.append(med_abs_dev)
  pd = np.array(pd)
  return pd


def med_pdist(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    k = np.median(pdist(d.reshape(-1,1)))
    pd.append(k)
  pd = np.array(pd)
  return pd


def pd(score):
  pd = []
  for i in range(len(score)):
    d = score[i]
    k = np.mean(pdist(d.reshape(-1,1)))
    pd.append(k)
  pd = np.array(pd)
  return pd


def neg_kurtosis(score):
  k = []
  for i in range(len(score)):
    di = score[i]
    ki = kurtosis(di, nan_policy = 'raise')
    k.append(ki)
  k = np.array(k)
  return -k


def quantile(score):
    # score (n, d)
  score = score.reshape(len(score), -1)
  score_75 = np.percentile(score, 75, -1)
  score_25 = np.percentile(score, 25, -1)
  score_qt = score_75 - score_25
  return score_qt


def calculate(score, stat_name):
	if stat_name == 'variance':
		results = np.var(score, axis = -1)
	elif stat_name == 'std':
		results = np.std(score, axis = -1)
	elif stat_name == 'pdist':
		results = pd(score)
	elif stat_name == 'con':
		results = con(score)
	elif stat_name == 'med_pdist':
		results = med_pdist(score)
	elif stat_name == 'kurtosis':
		results = neg_kurtosis(score)
	elif stat_name == 'skewness':
		results = -skew(score, axis = -1)
	elif stat_name == 'quantile':
		results = quantile(score)
	elif stat_name == 'mad':
		results = mad(score)
	#print('results.shape', results.shape)
	return results


def collect_layers(model, interested_layers):
	if model.framework == 'keras':
		outputs = [layer.output for layer in model.layers]
	elif model.framework == 'tensorflow':
		outputs = model.layers

	outputs = [output for i, output in enumerate(outputs) if i in interested_layers]
	#print(outputs)
	features = []
	for output in outputs:
		#print(output)
		if len(output.get_shape())== 4:
			features.append(
				tf.reduce_mean(output, axis = (1, 2))
			)
		else:
			features.append(output)
	return features

			
def evaluate_features(x, model, features, batch_size=500):
	x = np.array(x)
	if len(x.shape) == 3:
		_x = np.expand_dims(x, 0) 
	else:
		_x = x
	num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))

	outs = []
	for i in range(num_iters):
		x_batch = _x[i * batch_size: (i+1) * batch_size]
		out = model.sess.run(features, 
			feed_dict = {model.input_ph: x_batch})

		outs.append(out)

	num_layers = len(outs[0])
	outputs = []
	for l in range(num_layers):
		outputs.append(np.concatenate([outs[s][l] for s in range(len(outs))]))

	# (3073, 64)
	# (3073, 64)
	# (3073, 128)
	# (3073, 128)
	# (3073, 256)
	# (3073, 256)
	# (3073, 10)
	# (3073, 1)
	outputs = np.concatenate(outputs, axis = 1)
	prob = outputs[:,-model.num_classes:]
	label = np.argmax(prob[-1])
	#print('outputs', outputs.shape)
	#print('prob[:, label]', np.expand_dims(prob[:, label], axis = 1).shape)
	outputs = np.concatenate([outputs, np.expand_dims(prob[:, label], axis = 1)], axis = 1)

	return outputs


def loo_ml_instance(sample, reference, model, features, batch_size=500):
	h,w,c = sample.shape
	sample = sample.reshape(-1)
	reference = reference.reshape(-1)

	data = []
	st = time.time()
	positions = np.ones((h*w*c + 1, h*w*c), dtype = np.bool)
	for i in range(h*w*c):
		positions[i, i] = False
	
	data = np.where(positions, sample, reference)

	data = data.reshape((-1, h, w, c))
	features_val = evaluate_features(data, model, features, batch_size=batch_size) # (3072+1, 906+1)
	st1 = time.time()

	return features_val


def get_ml_loo_features(model, x, reference, interested_layers, batch_size=3100,
		stat_names=['std', 'variance', 'con', 'kurtosis', 'skewness', 'quantile', 'mad']):
	# copied from generate_ml_loo_features

	features = collect_layers(model, interested_layers)
	all_features = []
	for sample in x:
		features_val = loo_ml_instance(sample, reference, model, features,
																	 batch_size=batch_size)
		features_val = np.transpose(features_val)[:,:-1]
		single_feature = []
		for stat_name in stat_names:
			single_feature.append(calculate(features_val, stat_name))
		single_feature = np.array(single_feature)
		all_features.append(single_feature)
	all_features = np.array(all_features)
	return all_features

def generate_ml_loo_features(args, data_model, reference, model, x, interested_layers, batch_size=500):
	# print(args.attack)
	# x = load_examples(data_model, attack)
	features = collect_layers(model, interested_layers)

	cat = {'original':'ori', 'adv':'adv', 'noisy':'noisy'}
	dt = {'train':'train', 'test':'test'}
	stat_names = ['std', 'variance', 'con', 'kurtosis', 'skewness', 'quantile', 'mad']

	combined_features = {data_type: {} for data_type in ['test', 'train']}
	for data_type in ['test', 'train']:
		print('data_type', data_type)
		for category in ['original', 'adv']:
			print('category', category)
			all_features = []
			for i, sample in enumerate(x[data_type][category]):
				print('Generating ML-LOO for {}th sample...'.format(i))
				features_val = loo_ml_instance(sample, reference, model, features, batch_size=batch_size)
				
				# (3073, 907)
				#print('features_val.shape', features_val.shape)
				features_val = np.transpose(features_val)[:,:-1]
				#print('features_val.shape', features_val.shape)
				# (906, 3073)
				single_feature = []
				for stat_name in stat_names:
					#print('stat_name', stat_name)
					single_feature.append(calculate(features_val, stat_name))

				single_feature = np.array(single_feature)
				#print('single_feature', single_feature.shape)
				# (k, 906)
				all_features.append(single_feature)
			print('all_features', np.array(all_features).shape)
			combined_features[data_type][category] = np.array(all_features)

			np.save('{}/data/{}_{}_{}_{}_{}.npy'.format(
				data_model,
				args.data_sample,
				dt[data_type],
				cat[category],
				args.attack, 
				args.det), 
				combined_features[data_type][category])

	return combined_features


def compute_stat_single_layer(output):
	# l2dist = pdist(output)
	# l1dist = pdist(output, 'minkowski', p = 1)
	# sl2dist = pdist(X, 'seuclidean')
	variance = np.sum(np.var(output, axis = 0))
	# on = np.sum(np.linalg.norm(output, ord = 1, axis = 0))
	con = np.sum(np.linalg.norm(output - np.mean(output, axis = 0), ord = 1, axis = 0))

	return variance, con


def load_features(data_model, attacks):
	def softmax(x, axis):
		"""Compute softmax values for each sets of scores in x."""
		e_x = np.exp(x - np.max(x, axis = axis, keepdims = True))
		return e_x / e_x.sum(axis=axis, keepdims = True) # only difference      

	cat = {'original':'', 'adv':'_adv', 'noisy':'_noisy'}
	dt = {'train':'_train', 'test':''}
	features = {attack: {'train': {}, 'test': {}} for attack in attacks}

	normalizer = {}
	for attack in attacks:
		for data_type in ['train', 'test']:
			for category in ['original', 'adv']:
				print('Loading data...')
				feature = np.load('{}/data/{}{}{}_{}_{}.npy'.format(data_model,'x_val200',  
					dt[data_type], 
					cat[category], 
					attack, 
					'ml_loo')) # [n, 3073, ...]
				n = len(feature)
				print('Processing...')
				nums = [0,64,64,128,128,256,256,10]
				splits = np.cumsum(nums) # [0,64,128,...]
				processed = []
				for j, s in enumerate(splits):
					if j < len(splits) - 1:
						separated = feature[:, :-1, s:splits[j+1]]

						if j == len(splits) - 2:
							separated = softmax(separated, axis = -1)
							
						dist = np.var(separated, axis = 1) # [n, ...]
						if data_type == 'train' and category == 'original' and attack == 'linfpgd':
							avg_dist = np.mean(dist, axis = 0)
							normalizer[j] = avg_dist

						# dist /= normalizer[j]
						dist = np.sqrt(dist)

						# max_dist = np.max(dist, axis = -1)
						print(np.mean(dist))
						processed.append(dist.T)

				processed = np.concatenate(processed, axis = 0).T
				# processed = np.concatenate(processed, axis = )


				print(processed.shape)

				features[attack][data_type][category] = processed

	return features

