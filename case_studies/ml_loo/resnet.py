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

"""Trains a ResNet on the CIFAR10 dataset.

ResNet v1
[a] Deep Residual Learning for Image Recognition
https://arxiv.org/pdf/1512.03385.pdf

ResNet v2
[b] Identity Mappings in Deep Residual Networks
https://arxiv.org/pdf/1603.05027.pdf
"""

from __future__ import print_function
import keras
from keras.layers import Dense, Conv2D, BatchNormalization, Activation
from keras.layers import AveragePooling2D, Input, Flatten
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras import backend as K
from keras.models import Model
from keras.datasets import cifar10
import numpy as np
import os



def lr_schedule(epoch):
	"""Learning Rate Schedule

	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.

	# Arguments
		epoch (int): The number of epochs

	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-3
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 80:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr


def lr_schedule_cifar100(epoch):
	"""Learning Rate Schedule

	Learning rate is scheduled to be reduced after 80, 120, 160, 180 epochs.
	Called automatically every epoch as part of callbacks during training.

	# Arguments
		epoch (int): The number of epochs

	# Returns
		lr (float32): learning rate
	"""
	lr = 1e-4
	if epoch > 180:
		lr *= 0.5e-3
	elif epoch > 160:
		lr *= 1e-3
	elif epoch > 120:
		lr *= 1e-2
	elif epoch > 80:
		lr *= 1e-1
	print('Learning rate: ', lr)
	return lr

def lr_schedule_sgd(epoch):
	decay = epoch >= 122 and 2 or epoch >= 81 and 1 or 0
	lr = 1e-1 * 0.1 ** decay
	print('Learning rate: ', lr)
	return lr	

def resnet_layer(inputs,
				 num_filters=16,
				 kernel_size=3,
				 strides=1,
				 activation='relu',
				 batch_normalization=True,
				 conv_first=True):
	"""2D Convolution-Batch Normalization-Activation stack builder

	# Arguments
		inputs (tensor): input tensor from input image or previous layer
		num_filters (int): Conv2D number of filters
		kernel_size (int): Conv2D square kernel dimensions
		strides (int): Conv2D square stride dimensions
		activation (string): activation name
		batch_normalization (bool): whether to include batch normalization
		conv_first (bool): conv-bn-activation (True) or
			bn-activation-conv (False)

	# Returns
		x (tensor): tensor as input to the next layer
	"""
	conv = Conv2D(num_filters,
				  kernel_size=kernel_size,
				  strides=strides,
				  padding='same',
				  kernel_initializer='he_normal',
				  kernel_regularizer=l2(1e-4))

	x = inputs
	if conv_first:
		x = conv(x)
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
	else:
		if batch_normalization:
			x = BatchNormalization()(x)
		if activation is not None:
			x = Activation(activation)(x)
		x = conv(x)
	return x


def resnet_v2(input_shape, depth, num_classes=10):
	"""ResNet Version 2 Model builder [b]

	Stacks of (1 x 1)-(3 x 3)-(1 x 1) BN-ReLU-Conv2D or also known as
	bottleneck layer
	First shortcut connection per layer is 1 x 1 Conv2D.
	Second and onwards shortcut connection is identity.
	At the beginning of each stage, the feature map size is halved (downsampled)
	by a convolutional layer with strides=2, while the number of filter maps is
	doubled. Within each stage, the layers have the same number filters and the
	same filter map sizes.
	Features maps sizes:
	conv1  : 32x32,  16
	stage 0: 32x32,  64
	stage 1: 16x16, 128
	stage 2:  8x8,  256

	# Arguments
		input_shape (tensor): shape of input image tensor
		depth (int): number of core convolutional layers
		num_classes (int): number of classes (CIFAR10 has 10)

	# Returns
		model (Model): Keras model instance
	"""
	if (depth - 2) % 9 != 0:
		raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')
	# Start model definition.
	num_filters_in = 16
	num_res_blocks = int((depth - 2) / 9)

	inputs = Input(shape=input_shape)
	# v2 performs Conv2D with BN-ReLU on input before splitting into 2 paths
	x = resnet_layer(inputs=inputs,
					 num_filters=num_filters_in,
					 conv_first=True)

	# Instantiate the stack of residual units
	for stage in range(3):
		for res_block in range(num_res_blocks):
			activation = 'relu'
			batch_normalization = True
			strides = 1
			if stage == 0:
				num_filters_out = num_filters_in * 4
				if res_block == 0:  # first layer and first stage
					activation = None
					batch_normalization = False
			else:
				num_filters_out = num_filters_in * 2
				if res_block == 0:  # first layer but not first stage
					strides = 2    # downsample

			# bottleneck residual unit
			y = resnet_layer(inputs=x,
							 num_filters=num_filters_in,
							 kernel_size=1,
							 strides=strides,
							 activation=activation,
							 batch_normalization=batch_normalization,
							 conv_first=False)

			y = resnet_layer(inputs=y,
							 num_filters=num_filters_in,
							 conv_first=False)
			y = resnet_layer(inputs=y,
							 num_filters=num_filters_out,
							 kernel_size=1,
							 conv_first=False)
			if res_block == 0:
				# linear projection residual shortcut connection to match
				# changed dims
				x = resnet_layer(inputs=x,
								 num_filters=num_filters_out,
								 kernel_size=1,
								 strides=strides,
								 activation=None,
								 batch_normalization=False)
				
			x = keras.layers.add([x, y])

		num_filters_in = num_filters_out

	# Add classifier on top.
	# v2 has BN-ReLU before Pooling
	x = BatchNormalization()(x)
	x = Activation('relu')(x)
	pool_size = int(x.get_shape()[1])
	x = AveragePooling2D(pool_size=pool_size)(x)
	y = Flatten()(x)
	outputs = Dense(num_classes,
					activation=None,
					kernel_initializer='he_normal')(y)

	outputs = Activation('softmax')(outputs)

	# Instantiate model.
	model = Model(inputs=inputs, outputs=outputs)
	return model, inputs, outputs

	

def create_resnet_generator(x_train):
	# This will do preprocessing and realtime data augmentation:
	datagen = ImageDataGenerator(
		# set input mean to 0 over the dataset
		featurewise_center=False,
		# set each sample mean to 0
		samplewise_center=False,
		# divide inputs by std of dataset
		featurewise_std_normalization=False,
		# divide each input by its std
		samplewise_std_normalization=False,
		# apply ZCA whitening
		zca_whitening=False,
		# epsilon for ZCA whitening
		zca_epsilon=1e-06,
		# randomly rotate images in the range (deg 0 to 180)
		rotation_range=0,
		# randomly shift images horizontally
		width_shift_range=0.1,
		# randomly shift images vertically
		height_shift_range=0.1,
		# set range for random shear
		shear_range=0.,
		# set range for random zoom
		zoom_range=0.,
		# set range for random channel shifts
		channel_shift_range=0.,
		# set mode for filling points outside the input boundaries
		fill_mode='nearest',
		# value used for fill_mode = "constant"
		cval=0.,
		# randomly flip images
		horizontal_flip=True,
		# randomly flip images
		vertical_flip=False,
		# set rescaling factor (applied before any other transformation)
		rescale=None,
		# set function that will be applied on each input
		preprocessing_function=None,
		# image data format, either "channels_first" or "channels_last"
		data_format=None,
		# fraction of images reserved for validation (strictly between 0 and 1)
		validation_split=0.0)

	# Compute quantities required for featurewise normalization
	# (std, mean, and principal components if ZCA whitening is applied).
	datagen.fit(x_train)
	return datagen


