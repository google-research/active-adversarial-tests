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
import tensorflow as tf
import numpy as np
import os
from keras.layers import Flatten, Conv2D, MaxPooling2D, Conv2DTranspose, UpSampling2D, Convolution2D, BatchNormalization, Dense, Dropout, Activation, Embedding, Conv1D, Input, GlobalMaxPooling1D, Multiply, Lambda, Permute, GlobalAveragePooling2D 
from keras.preprocessing import sequence
from keras.datasets import imdb, mnist
from keras.callbacks import ModelCheckpoint
from keras.models import Model, Sequential
from keras.objectives import binary_crossentropy
from keras.metrics import binary_accuracy as accuracy
from keras.optimizers import RMSprop
from keras import backend as K  
from keras import optimizers
import math


def construct_original_network(dataset_name, model_name, train): 
	data_model = dataset_name + model_name
	if dataset_name == 'mnist': 
		input_size = 28
		num_classes = 10
		channel = 1


	elif dataset_name == 'cifar10':
		# Define the model 
		input_size = 32
		num_classes = 10
		channel = 3

	elif dataset_name == 'cifar100':
		# Define the model 
		input_size = 32
		num_classes = 100
		channel = 3		

	if model_name == 'scnn':
		image_ph = Input(shape=(input_size,input_size,channel),dtype = 'float32')
		net = Convolution2D(32, kernel_size=(5, 5),padding = 'same',
				 activation='relu', name = 'conv1')(image_ph)
		net = MaxPooling2D(pool_size=(2, 2))(net)
		net = Convolution2D(64, (5, 5),padding = 'same',
				 activation='relu', name = 'conv2')(net)
		net = MaxPooling2D(pool_size=(2, 2))(net) 

		net = Flatten()(net)
		net = Dense(1024, activation='relu',name='fc1')(net) 
		net = Dense(num_classes, activation='softmax',name='fc2')(net) 
		preds = Activation('softmax')(net) 
		model = Model(image_ph, preds)

		model.compile(loss='categorical_crossentropy',
					  optimizer='adam',
					  metrics=['acc']) 

	elif model_name == 'cnn':
		image_ph = Input(shape=(input_size,input_size,channel),dtype = 'float32')
		net = Convolution2D(48, (3,3), padding='same', input_shape=(32, 32, 3))(image_ph)
		net = Activation('relu')(net)
		net = Convolution2D(48, (3, 3))(net)
		net = Activation('relu')(net)
		net = MaxPooling2D(pool_size=(2, 2))(net)
		net = Dropout(0.25)(net)
		net = Convolution2D(96, (3,3), padding='same')(net)
		net = Activation('relu')(net)
		net = Convolution2D(96, (3, 3))(net)
		net = Activation('relu')(net)
		net = MaxPooling2D(pool_size=(2, 2))(net)
		net = Dropout(0.25)(net)
		net = Convolution2D(192, (3,3), padding='same')(net)
		net = Activation('relu')(net)
		net = Convolution2D(192, (3, 3))(net)
		net = Activation('relu')(net)
		net = MaxPooling2D(pool_size=(2, 2))(net)
		net = Dropout(0.25)(net)
		net = Flatten()(net)
		net = Dense(512)(net)
		net = Activation('relu')(net)
		net = Dropout(0.5)(net)
		net = Dense(256)(net)
		net = Activation('relu')(net)
		net = Dropout(0.5)(net)
		net = Dense(num_classes, activation=None)(net)
		preds = Activation('softmax')(net)

		model = Model(image_ph, preds)
		sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

		model.compile(loss='categorical_crossentropy',
						  optimizer=sgd,
						  metrics=['acc'])

		# Compile the model
	elif model_name == 'fc':
		image_ph = Input(shape=(input_size,input_size,channel),dtype = 'float32')
		net = Flatten()(image_ph)
		net = Dense(256)(net)
		net = Activation('relu')(net)
		
		net = Dense(256)(net)
		net = Activation('relu')(net)

		net = Dense(256)(net)
		net = Activation('relu')(net)
		
		preds = Dense(num_classes, activation='softmax')(net)

		model = Model(image_ph, preds)
		sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

		model.compile(loss='categorical_crossentropy',
						  optimizer=sgd,
						  metrics=['acc'])

	elif model_name == 'resnet':
		from resnet import resnet_v2, lr_schedule,  lr_schedule_sgd
		
		model, image_ph, preds = resnet_v2(input_shape=(input_size, input_size, channel), depth=20, num_classes = num_classes)

		optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)


		model.compile(loss='categorical_crossentropy',
				  optimizer=optimizer,
				  metrics=['accuracy'])

	elif model_name == 'densenet':
		from densenet import DenseNet 
		nb_filter = -1#12 if dataset_name == 'cifar100' else -1
		
		image_ph = Input(shape=(input_size,input_size,channel),dtype = 'float32')
		model, preds = DenseNet((input_size,input_size,channel), 
			classes=num_classes, depth=40, nb_dense_block=3, growth_rate=12, nb_filter=nb_filter, dropout_rate=0.0, weights=None, input_tensor = image_ph)
			
		optimizer = optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)


		model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=["accuracy"])

	grads = []
	for c in range(num_classes):
		grads.append(tf.gradients(preds[:,c], image_ph))

	grads = tf.concat(grads, axis = 0)
	approxs = grads * tf.expand_dims(image_ph, 0)

	logits = [layer.output for layer in model.layers][-2]
	print(logits)
		
	sess = K.get_session()

	return image_ph, preds, grads, approxs, sess, model, num_classes, logits


class ImageModel():
	def __init__(self, model_name, dataset_name, train = False, load = False, **kwargs):
		self.model_name = model_name
		self.dataset_name = dataset_name
		self.data_model = dataset_name + model_name
		self.framework = 'keras'

		# if not train:
			# K.set_learning_phase(0)

		print('Constructing network...')
		self.input_ph, self.preds, self.grads, self.approxs, self.sess, self.model, self.num_classes, self.logits = construct_original_network(self.dataset_name, self.model_name, train = train)


		self.layers = self.model.layers
		self.last_hidden_layer = self.model.layers[-3]

		self.y_ph = tf.placeholder(tf.float32, shape = [None, self.num_classes])
		if load:
			if load == True:
				print('Loading model weights...')
				self.model.load_weights('{}/models/original.hdf5'.format(self.data_model), 
					by_name=True)
			elif load != False:
				self.model.load_weights('{}/models/{}.hdf5'.format(self.data_model, load), 
					by_name=True)

		self.pred_counter = 0

	def train(self, dataset): 
		print('Training...')
		if self.dataset_name == 'mnist':
			assert self.model_name in ['cnn', 'scnn']
			data_model = self.dataset_name + self.model_name
			filepath="{}/models/original.hdf5".format(data_model)
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
				verbose=1, save_best_only=True, mode='max')
			callbacks_list = [checkpoint]
			history = self.model.fit(dataset.x_train, dataset.y_train, 
				validation_data=(dataset.x_val, dataset.y_val),
				callbacks = callbacks_list, 
				epochs=100, batch_size=128)
			# print(history.history)
		elif self.dataset_name in ['cifar10', 'cifar100']:
			from keras.preprocessing.image import ImageDataGenerator

			if self.model_name == 'cnn':
				datagen = ImageDataGenerator(zoom_range=0.2, horizontal_flip=True)
				# zoom 0.2
				datagen = create_resnet_generator(dataset.x_train)
				callbacks_list = []
				batch_size = 128
				num_epochs = 200

			elif self.model_name in ['resnet', 'densenet']:
				from resnet import lr_schedule, create_resnet_generator
				from keras.callbacks import LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
				# zoom 0.2 horizontal_filp always True. change optimizer to sgd, and batch_size to 128. 
				datagen = ImageDataGenerator(rotation_range=15,
                               width_shift_range=5./32,
                               height_shift_range=5./32,
                               horizontal_flip = True,
                               zoom_range = 0.2)

				datagen.fit(dataset.x_train, seed=0)

				from resnet import lr_schedule_sgd
				from keras.callbacks import LearningRateScheduler
				lr_scheduler = LearningRateScheduler(lr_schedule_sgd)
				callbacks_list = [lr_scheduler]
				batch_size = 128 if self.dataset_name == 'cifar10' else 64
				num_epochs = 200

			filepath="{}/models/original.hdf5".format(self.data_model)
			checkpoint = ModelCheckpoint(filepath, monitor='val_acc', 
				verbose=1, save_best_only=True, mode='max')
			callbacks_list.append(checkpoint)

			
			model_info = self.model.fit_generator(datagen.flow(dataset.x_train, 
				dataset.y_train, batch_size = batch_size),
				epochs = num_epochs,
				steps_per_epoch = dataset.x_train.shape[0] // batch_size,
				callbacks = callbacks_list, 
				validation_data = (dataset.x_val, dataset.y_val), 
				verbose = 2,
				workers = 4)

	def adv_train(self, dataset, attack_name):
		from cleverhans.attacks import FastGradientMethod, ProjectedGradientDescent
		from cleverhans.utils_keras import KerasModelWrapper
		from cleverhans.loss import CrossEntropy
		from cleverhans.train import train
		from cleverhans.utils_tf import model_eval
		import time, datetime

		if attack_name == 'fgsm' and self.dataset_name == 'mnist':
			wrap = KerasModelWrapper(self.model)
			params = {'eps': 0.3,
				'clip_min': -1.,
				'clip_max': 1.}

			attacker = FastGradientMethod(wrap, sess=self.sess)
			def attack(x):
				return attacker.generate(x, **params)

			preds_adv = self.model(attack(self.input_ph))
			loss = CrossEntropy(wrap, smoothing=0.1, attack=attack)

			y_ph = tf.placeholder(tf.float32, shape = (None, self.num_classes))

			def evaluate():
				# Accuracy of adversarially trained model on legitimate test inputs
				eval_params = {'batch_size': 128}
				accuracy = model_eval(self.sess, self.input_ph, y_ph, self.preds, dataset.x_val, dataset.y_val, args=eval_params)
				print('Test accuracy on legitimate examples: %0.4f' % accuracy)

				# Accuracy of the adversarially trained model on adversarial examples
				accuracy = model_eval(self.sess, self.input_ph, y_ph, preds_adv, dataset.x_val, dataset.y_val, args=eval_params)
				print('Test accuracy on adversarial examples: %0.4f' % accuracy)

			# if self.dataset_name == 'mnist':
			train_params = {
				'nb_epochs': 20,
				'batch_size': 128,
				'learning_rate': 0.001,
				'train_dir': '{}/models'.format(self.data_model),
				'filename': 'adv.cpkt'
			}

			# Perform and evaluate adversarial training
			train(self.sess, loss, dataset.x_train, dataset.y_train, evaluate=evaluate,
				args=train_params, rng=np.random.RandomState([2017, 8, 30]))

			self.model.save_weights('{}/models/{}.hdf5'.format(self.data_model, 'adv-{}'.format(attack_name)))

		elif attack_name == 'pgd':
			if self.dataset_name == 'mnist':
				params = {'eps': 0.1,
							# 'clip_min': -1.0,
							# 'clip_max': 1.0,
							'eps_iter': 0.01,
							'nb_iter': 20,
							'epochs': 100,
							'batch_size': 50,
							}
			elif self.dataset_name == 'cifar10':
				params = {'eps': 8.0 / 255 * 2,
							# 'clip_min': -1.0,
							# 'clip_max': 1.0,
							'eps_iter': 2.0 / 255 * 2,
							'nb_iter': 10,#10,#1,
							'epochs': 200,
							'batch_size': 128,
							}				

			# attacker = ProjectedGradientDescent(wrap, sess=self.sess)

			# import foolbox
			# from foolbox.attacks import ProjectedGradientDescentAttack
			from attack_model import LinfPGDAttack
			# Main training loop
			# fmodel = foolbox.models.KerasModel(self.model, bounds=(-1, 1), preprocessing=(0, 1))
			attacker = LinfPGDAttack(self, params['eps'], k = params['nb_iter'], a = params['eps_iter'], clip_min = dataset.clip_min, clip_max = dataset.clip_max, 
				random_start = True, loss_func = 'xent')

			def attack(x, y):
				# return attacker(x, label=label, unpack=True, binary_search=False, epsilon=params['eps'], stepsize=params['eps_iter'], 
				# 	iterations=params['nb_iter'], 
				# 	random_start=False, return_early=True)
				return attacker.attack(x, np.argmax(y, axis = -1))

			from resnet import lr_schedule, create_resnet_generator,  lr_schedule_sgd
			from keras.preprocessing.image import ImageDataGenerator

			# datagen = create_resnet_generator(dataset.x_train)
			datagen = ImageDataGenerator(rotation_range=15,
				width_shift_range=5./32,
				height_shift_range=5./32,
				horizontal_flip = True,
				zoom_range = 0.2)

			datagen.fit(dataset.x_train, seed=0)

			xent = tf.reduce_mean(K.categorical_crossentropy(self.y_ph, self.preds), name='y_xent')


			global_step = tf.train.get_or_create_global_step()

			if self.dataset_name == 'cifar10':
				momentum = 0.9
				weight_decay = 0.0002
				costs = []
				print('number of trainable variables: ',len(tf.trainable_variables()))
				for var in tf.trainable_variables():
					if 'kernel' in var.name:
						costs.append(tf.nn.l2_loss(var))
				penalty = tf.add_n(costs)

				loss = xent + weight_decay * penalty
			elif self.dataset_name == 'mnist':
				loss = xent


			if self.dataset_name == 'cifar10':
				boundaries = [40000,60000]
				values = [0.1,0.01,0.001]
				learning_rate = tf.train.piecewise_constant(
					tf.cast(global_step, tf.int32),
					boundaries,
					values)
				optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
			elif self.dataset_name == 'mnist':
				boundaries = [50000]
				values = [1e-3,1e-4]
				learning_rate = tf.train.piecewise_constant(
					tf.cast(global_step, tf.int32),
					boundaries,
					values)
				optimizer = tf.train.AdamOptimizer(learning_rate)

			train_step = optimizer.minimize(loss, global_step=global_step)


			accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(self.preds, 1), 
				tf.argmax(self.y_ph, 1)), tf.float32))

			num_output_steps = 100 # 100
			num_checkpoint_steps = 1000
			batch_size = params['batch_size']
			ii = 0
			epochs = params['epochs']

			for e in range(epochs):
				num_batches = 0
				for x_batch, y_batch in datagen.flow(dataset.x_train, dataset.y_train, batch_size=batch_size):

					# Compute Adversarial Perturbations
					start = time.time()

					x_batch_adv = attack(x_batch, y_batch)

					nat_dict = {self.input_ph: x_batch,
					            self.y_ph: y_batch}

					adv_dict = {self.input_ph: x_batch_adv,
					            self.y_ph: y_batch}
					eval_dict = {self.input_ph: dataset.x_train[:1000],
					            self.y_ph: dataset.y_train[:1000]}
					val_dict = {self.input_ph: dataset.x_val[:1000],
					            self.y_ph: dataset.y_val[:1000]}
					# Output to stdout
					if ii % num_output_steps == 0:
						nat_acc = self.sess.run(accuracy, feed_dict=eval_dict)
						val_acc = self.sess.run(accuracy, feed_dict=val_dict)
						adv_acc = self.sess.run(accuracy, feed_dict=adv_dict)

						print('Step {} '.format(ii))
						print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
						print('    validation accuracy {:.4}%'.format(val_acc * 100))
						print('    training adv accuracy {:.4}%'.format(adv_acc * 100))

						if ii != 0:
							print('    {} examples per second'.format(
							num_output_steps * batch_size / training_time))
							training_time = 0.0


					# Write a checkpoint
					if ii % num_checkpoint_steps == 0:
						self.model.save_weights('{}/models/adv-{}-{}.hdf5'.format(self.data_model, attack_name, ii))

					# Actual training step
					
					_ = self.sess.run(train_step, feed_dict=adv_dict)
					# print(step)
					end = time.time()
					training_time = end - start
					ii += 1
					num_batches += 1

					if num_batches >= len(dataset.x_train) / batch_size:
						break

			self.model.save_weights('{}/models/adv-{}.hdf5'.format(self.data_model, attack_name))



	def predict(self, x, verbose=0, batch_size = 500, logits = False):
		x = np.array(x)
		if len(x.shape) == 3:
			_x = np.expand_dims(x, 0) 
		else:
			_x = x

		self.pred_counter += len(_x)
			
		if not logits:
			prob = self.model.predict(_x, batch_size = batch_size, 
			verbose = verbose)
		else:
			num_iters = int(math.ceil(len(_x) * 1.0 / batch_size))
			probs = []
			for i in range(num_iters):
				# print('{} samples predicted'.format(i * batch_size))
				x_batch = _x[i * batch_size: (i+1) * batch_size]

				prob = self.sess.run(self.logits, 
					feed_dict = {self.input_ph: x_batch})

				probs.append(prob)
				
			prob = np.concatenate(probs, axis = 0)

		if len(x.shape) == 3:
			prob = prob.reshape(-1)

		return prob

	def compute_saliency(self, x, saliency_type = 'gradient'):
		x = np.array(x)
		if self.dataset_name in ['mnist', 'cifar10', 'cifar100']:
			batchsize = 128 #if self.data in ['imdbcnn','imdblstm'] else 20
			num_iters = int(math.ceil(len(x) * 1.0 / batchsize))
			approxs_val = []

			for i in range(num_iters): 
				batch_data = x[i * batchsize: (i+1) * batchsize]

				if saliency_type == 'gradient':
					approxs = self.grads

				elif saliency_type == 'taylor':
					approxs = self.approxs

				batch_approxs = self.sess.run(approxs, feed_dict = {self.input_ph: batch_data}) 
				# [num_classes, batchsize, h, w, c]
				approxs_val.append(batch_approxs) 

			approxs_val = np.concatenate(approxs_val, axis = 1)
			# [num_classes, num_data, h, w, c]

			pred_val = self.predict(x)
			
			class_specific_scores = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]
			# [num_data, h, w, c]

			return class_specific_scores

	def compute_ig(self, x, reference):
		x = np.array(x)
		if self.dataset_name in ['mnist', 'cifar10', 'cifar100']:
			batchsize = 1
			steps = 50

			pred_vals = self.predict(x)
			class_specific_scores = []

			num_iters = int(math.ceil(len(x) * 1.0 / batchsize))
			for i in range(num_iters): 
				batch_data = x[i * batchsize: (i+1) * batchsize] 
				_, h, w, c = batch_data.shape
				step_batch = [batch_data * float(s) / steps + reference * (1 - float(s) / steps) for s in range(1, steps+1)] 
				# [steps,batchsize, h, w, c]

				step_batch = np.reshape(step_batch, 
						[-1, h, w, c])
				# [steps * batchsize, h, w, c]

				batch_grads = self.sess.run(self.grads,
					feed_dict = {self.input_ph: step_batch})
				# [num_classes, steps * batchsize, h, w, c]
				num_classes, _, h, w, c = batch_grads.shape
				grads_val = np.mean(batch_grads.reshape([num_classes, steps, -1, h, w, c]), axis = 1)
				approxs_val = grads_val * (batch_data - reference)
				# [num_classes, batchsize, h, w, c]

				pred_val = pred_vals[i * batchsize: (i+1) * batchsize]
				class_specific_score = approxs_val[np.argmax(pred_val, axis = 1), range(len(pred_val))]
				# [batchsize, h, w, c]

				# [batchsize, maxlen]
				class_specific_scores.append(class_specific_score)
 
			# [num_data, length]
			return np.concatenate(class_specific_scores, axis = 0)















