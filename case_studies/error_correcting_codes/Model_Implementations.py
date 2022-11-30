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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Full implementation of all methods of Abstract class "Model"
"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Dropout, Multiply, Lambda, Input, Dense, Conv2D, MaxPooling2D, Flatten, Activation, UpSampling2D, Concatenate, GaussianNoise
from tensorflow.keras.utils import plot_model
from tensorflow.keras import metrics, regularizers, optimizers
from tensorflow.keras.models import Model as KerasModel
from Model import Model
from tensorflow.keras import losses, metrics
from ClassBlender import ClassBlender
from DataAugmenter import DataAugmenter



#Full architectural definition for all "baseline" models used in the paper
def defineModelBaseline(self):
        outputs=[]
        self.penultimate = []
        self.penultimate2 = []
            
        x = self.input      
       
        x = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x)
        if (self.TRAIN_FLAG==1):
            if self.params_dict['DATA_AUGMENTATION_FLAG']>0:
                x = DataAugmenter(self.params_dict['batch_size'])(x)
            x = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x)  
            
        x = Lambda(lambda x:  tf.clip_by_value(x,-0.5,0.5))(x) 
        
        for rep in np.arange(self.params_dict['model_rep']):
            x = Conv2D(self.params_dict['num_filters_std'][0], (5,5), activation='elu', padding='same', kernel_regularizer=regularizers.l2(self.params_dict['weight_decay']))(x)
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)
            
        x = Conv2D(self.params_dict['num_filters_std'][0], (3,3), strides=(2,2), activation='elu', padding='same')(x)  

        for rep in np.arange(self.params_dict['model_rep']):
            x = Conv2D(self.params_dict['num_filters_std'][1], (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(self.params_dict['weight_decay']))(x)
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)

        x = Conv2D(self.params_dict['num_filters_std'][1], (3,3), strides=(2,2), activation='elu', padding='same')(x)         
        x_=x
        
        for rep in np.arange(self.params_dict['model_rep']):
            x_ = Conv2D(self.params_dict['num_filters_std'][2], (3, 3), activation='elu', padding='same', kernel_regularizer=regularizers.l2(self.params_dict['weight_decay']))(x_)        
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x_ = BatchNormalization()(x_)

        x_ = Conv2D(self.params_dict['num_filters_std'][2], (3,3), strides=(2,2), activation='elu', padding='same')(x_)  

        x_ = Flatten()(x_)
        
        x_ = Dense(128, activation='elu')(x_)  
        x_ = Dense(64, activation='elu')(x_)         
        x0 = Dense(64, activation='linear')(x_) 
        x1 = Dense(self.params_dict['M'].shape[1], activation='linear', kernel_regularizer=regularizers.l2(0.0))(x0)
                
        outputs = [x1]
        self.model = KerasModel(inputs=self.input, outputs=outputs)
        print(self.model.summary())
        plot_model(self.model, to_file=self.params_dict['model_path'] + '/' + self.params_dict['name'] + '.png')    

        return outputs  
        




class Model_Softmax_Baseline(Model):
    
    def __init__(self, data_dict, params_dict):
        super(Model_Softmax_Baseline, self).__init__(data_dict, params_dict)


    def defineModel(self):
        return defineModelBaseline(self)



    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):
            loss = tf.keras.backend.categorical_crossentropy(y_true, y_pred, from_logits=True)
            return loss 
        return loss_fn
    
    
    def defineMetric(self):
        return [metrics.categorical_accuracy]














class Model_Logistic_Baseline(Model):
    
    def __init__(self, data_dict, params_dict):
        super(Model_Logistic_Baseline, self).__init__(data_dict, params_dict)


    def defineModel(self):
        return defineModelBaseline(self)


        
    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):  
            loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
            return loss 
        return loss_fn
    

    
    
    def defineMetric(self): 
        def sigmoid_pred(y_true, y_pred):
            
            corr = tf.to_float((y_pred*(2*y_true-1))>0)
            return tf.reduce_mean(corr)
        
        return [sigmoid_pred]











          
class Model_Tanh_Baseline(Model):
    
    def __init__(self, data_dict, params_dict):
        super(Model_Tanh_Baseline, self).__init__(data_dict, params_dict)



    def defineModel(self):
        return defineModelBaseline(self)


        
    def defineLoss(self, idx):     
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss   
            
        return hinge_loss
    

    
    
    def defineMetric(self):
        def tanh_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        return [tanh_pred]









        
class Model_Logistic_Ensemble(Model):
    
    def __init__(self, data_dict, params_dict):
        super(Model_Logistic_Ensemble, self).__init__(data_dict, params_dict)
    
    def defineLoss(self, idx):
        def loss_fn(y_true, y_pred):  
            loss = tf.keras.backend.binary_crossentropy(y_true, y_pred, from_logits=True)
            return loss 
        return loss_fn
    
        
    
    def defineMetric(self): 
        def sigmoid_pred(y_true, y_pred):
            
            corr = tf.to_float((y_pred*(2*y_true-1))>0)
            return tf.reduce_mean(corr)
        
        return [sigmoid_pred]







class Model_Tanh_Ensemble(Model):
    
    def __init__(self, data_dict, params_dict):
        super(Model_Tanh_Ensemble, self).__init__(data_dict, params_dict)


              
    def defineLoss(self, idx):
        
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss   
        
        return hinge_loss
        
    
    
    def defineMetric(self):
        def hinge_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        return [hinge_pred]
          
    