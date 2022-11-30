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
This defines a general "Model", i.e. architecture and decoding operations. It is an abstract base class for all models, 
e.g. the baseline softmax model or the ensemble Tanh model
"""
import tensorflow as tf
from utils_keras import KerasModelWrapper as CleverHansKerasModelWrapper

from tensorflow.keras.layers import BatchNormalization, Dropout, Lambda, Input, Dense, Conv2D, Flatten, Activation, Concatenate, GaussianNoise
from tensorflow.keras.utils import plot_model
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.models import Model as KerasModel

import pickle
import numpy as np
from ClassBlender import ClassBlender
from DataAugmenter import DataAugmenter
from Clipper import Clipper
from Grayscaler import Grayscaler



class WeightsSaver(Callback):
    def __init__(self, N):
        self.N = N
        self.epoch = 0
        
    def specifyFilePath(self, path):
        self.full_path = path #full path to file, including file name
        
    def on_epoch_end(self, epoch, logs={}):
       if self.epoch % self.N == 0:
            print("SAVING WEIGHTS")
            w= self.model.get_weights()
            pklfile= self.full_path + '_' + str(self.epoch) + '.pkl'
            fpkl= open(pklfile, 'wb')        
            pickle.dump(w, fpkl)
            fpkl.close()
       self.epoch += 1



#Abstract base class for all model classes
class Model(object):
    
    def __init__(self, data_dict, params_dict):
        self.data_dict = data_dict
        self.params_dict = params_dict
        self.input = Input(shape=self.params_dict['inp_shape'], name='input') 
        self.TRAIN_FLAG=1
        if len(data_dict) > 0:
            self.encodeData()
        else:
            import warnings
            warnings.warn("no data passed; cannot encode it")
        
        
    #map categorical class labels (numbers) to encoded (e.g., one hot or ECOC) vectors
    def encodeData(self):
        self.Y_train = np.zeros((self.data_dict['X_train'].shape[0], self.params_dict['M'].shape[1]))
        self.Y_test = np.zeros((self.data_dict['X_test'].shape[0], self.params_dict['M'].shape[1]))
        for k in np.arange(self.params_dict['M'].shape[1]):
            self.Y_train[:,k] = self.params_dict['M'][self.data_dict['Y_train_cat'], k]
            self.Y_test[:,k] = self.params_dict['M'][self.data_dict['Y_test_cat'], k]
            


    #define the neural network
    def defineModel(self):
        
        outputs=[]
        self.penultimate = []
        self.penultimate2 = []

        features = []
        
        n = int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
        for k in np.arange(0,self.params_dict['num_chunks']):
            
            x = self.input

            if self.params_dict.get('zero_one_input', False):
                x = x - 0.5
            
            if self.params_dict['inp_shape'][2]>1:
                x_gs = Grayscaler()(x)
            else:
                x_gs = x

            if (self.TRAIN_FLAG==1):
                x = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x)
                x_gs = GaussianNoise(self.params_dict['noise_stddev'], input_shape=self.params_dict['inp_shape'])(x_gs)

                if self.params_dict['DATA_AUGMENTATION_FLAG']>0:
                    x = DataAugmenter(self.params_dict['batch_size'])(x)
                    x_gs = DataAugmenter(self.params_dict['batch_size'])(x_gs)

                x = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x)  
                x_gs = ClassBlender(self.params_dict['blend_factor'], self.params_dict['batch_size'])(x_gs)  
            
                 
            #x = Lambda(lambda x: x-0.5)(x) 

            x = Clipper()(x)
            x_gs = Clipper()(x_gs)

            # TODO: verify that this modifcation makes sense

            # Added trainable=self.TRAIN_FLAG==1 for all batchnorm layers to make
            # sure they stay fixed during eval (modification by AUTHOR)

            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][0], (5,5), activation='elu', padding='same')(x)          
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            

            x = Conv2D(self.params_dict['num_filters_ens'][0], (3,3), strides=(2,2), activation='elu', padding='same')(x)
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)


            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][1], (3, 3), activation='elu', padding='same')(x)
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            
            x = Conv2D(self.params_dict['num_filters_ens'][1], (3,3), strides=(2,2), activation='elu', padding='same')(x)  
            if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                x = BatchNormalization()(x)
            
            for rep in np.arange(self.params_dict['model_rep']):
                x = Conv2D(self.params_dict['num_filters_ens'][2], (3, 3), activation='elu', padding='same')(x)
                if self.params_dict['BATCH_NORMALIZATION_FLAG']>0:
                    x = BatchNormalization()(x)
            
            
            x = Conv2D(self.params_dict['num_filters_ens'][2], (3,3), strides=(2,2), activation='elu', padding='same')(x)  
            #x = BatchNormalization()(x)


                        
            pens = []
            out=[]
            for k2 in np.arange(n):
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (5, 5), strides=(2,2), activation='elu', padding='same')(x_gs)
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (3, 3), strides=(2,2), activation='elu', padding='same')(x0)
                x0 = Conv2D(self.params_dict['num_filters_ens_2'], (3, 3), strides=(2,2), activation='elu', padding='same')(x0)

                x_= Concatenate()([x0, x])
            
                x_ = Conv2D(self.params_dict['num_filters_ens_2'], (2, 2), activation='elu', padding='same')(x_)                    
                    
                x_ = Conv2D(self.params_dict['num_filters_ens_2'], (2, 2), activation='elu', padding='same')(x_)

                x_ = Flatten()(x_)

                features.append(x_)

                x_ = Dense(16, activation='elu')(x_) 
                x_ = Dense(8, activation='elu')(x_) 
                x_ = Dense(4, activation='elu')(x_)
                x0 = Dense(2, activation='linear')(x_)

                pens += [x0]                

                x1 = Dense(1, activation='linear', name='w_'+str(k)+'_'+str(k2)+'_'+self.params_dict['name'], kernel_regularizer=regularizers.l2(0.0))(x0) 
                out += [x1]
                
            self.penultimate += [pens]
            
            if len(pens) > 1:
                self.penultimate2 += [Concatenate()(pens)]
            else:
                self.penultimate2 += pens

            if len(out)>1:
                outputs += [Concatenate()(out)]
            else:
                outputs += out

        self.features = features

        self.model = KerasModel(inputs=self.input, outputs=outputs)
        # print(self.model.summary())
        #plot_model(self.model, to_file=self.params_dict['model_path'] + '/' + self.params_dict['name'] + '.png')

        return outputs 
    
            
    def defineLoss(self):
        error = "Sub-classes must implement defineLoss."
        raise NotImplementedError(error)
    
    
    def defineMetric(self):
        error = "Sub-classes must implement defineMetric."
        raise NotImplementedError(error)
    
        
    def trainModel(self):
        opt = Adam(lr=self.params_dict['lr'])
        
        self.model.compile(optimizer=opt, loss=[self.defineLoss(k) for k in np.arange(self.params_dict['num_chunks'])], metrics=self.defineMetric())
        WS = WeightsSaver(self.params_dict['weight_save_freq'])
        WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name'])
        
        Y_train_list=[]
        Y_test_list=[]

        start = 0
        for k in np.arange(self.params_dict['num_chunks']):
            end = start + int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
            Y_train_list += [self.Y_train[:,start:end]]
            Y_test_list += [self.Y_test[:,start:end]]
            start=end
            
        
        self.model.fit(self.data_dict['X_train'], Y_train_list,
                        epochs=self.params_dict['epochs'],
                        batch_size=self.params_dict['batch_size'],
                        shuffle=True,
                        validation_data=[self.data_dict['X_test'], Y_test_list],
                        callbacks=[WS])
        
        

        self.saveModel()
        
        
        
        
    def resumeTrainModel(self):
    
        Y_train_list=[]
        Y_test_list=[]

        start = 0
        for k in np.arange(self.params_dict['num_chunks']):
            end = start + int(self.params_dict['M'].shape[1]/self.params_dict['num_chunks'])
            Y_train_list += [self.Y_train[:,start:end]]
            Y_test_list += [self.Y_test[:,start:end]]
            start=end
        
        def hinge_loss(y_true, y_pred):
            loss = tf.reduce_mean(tf.maximum(1.0-y_true*y_pred, 0))
            return loss  
        
        def hinge_pred(y_true, y_pred):
            corr = tf.to_float((y_pred*y_true)>0)
            return tf.reduce_mean(corr)
        
        self.model = load_model(self.params_dict['model_path'] + self.params_dict['name'] + '_final.h5', custom_objects={'DataAugmenter':DataAugmenter, 'ClassBlender':ClassBlender, 'Clipper':Clipper, 'Grayscaler':Grayscaler, 'hinge_loss':hinge_loss, 'hinge_pred':hinge_pred})
        WS = WeightsSaver(self.params_dict['weight_save_freq'])
        WS.specifyFilePath(self.params_dict['model_path'] + self.params_dict['name'])
          
        
        self.model.fit(self.data_dict['X_train'], Y_train_list,
                        epochs=self.params_dict['epochs'],
                        batch_size=self.params_dict['batch_size'],
                        shuffle=True,
                        validation_data=[self.data_dict['X_test'], Y_test_list],
                        callbacks=[WS])
        
        

        self.saveModel()
        
                
        
    #this function takes the output of the NN and maps into logits (which will be passed into softmax to give a prob. dist.)
    #It effectively does a Hamming decoding by taking the inner product of the output with each column of the coding matrix (M)
    #obviously, the better the match, the larger the dot product is between the output and a given row
    #it is simply a log ReLU on the output
    def outputDecoder(self, x, M=None):
        if M is None:
            M = self.params_dict['M']
        mat1 = tf.matmul(x, M, transpose_b=True)

        if not self.params_dict['adaptive_attack']:
            mat1 = tf.maximum(mat1, 0)
            mat1 = tf.log(mat1+1e-6) #floor negative values
        return mat1


    def defineBinarizedModel(self):
        assert hasattr(self, "penultimate2"), "model needs to be defined first"

        readouts = []
        individual_logits = []
        for k in range(len(self.features)):
            readout = Dense(1, activation='linear',
                            name='binarized_readout_' + str(k),
                            kernel_regularizer=regularizers.l2(0.0)
                            )
            logit = readout(self.features[k])
            logit = Lambda(self.params_dict['output_activation'])(logit)
            readouts.append(readout)
            individual_logits.append(logit)

        if len(individual_logits)>1:
            logits = Concatenate()(individual_logits)
        else: #if only a single chunk
            logits = individual_logits[0]
        M = np.stack([np.ones(logits.shape[-1]), -np.ones(logits.shape[-1])], 0).astype(np.float32)
        logits = Lambda(
            lambda x: self.outputDecoder(
            x,
            M=M
        ))(logits)

        probs = Activation('softmax')(logits) #return probs

        self.binarized_logit = logits
        self.binarized_probs = probs
        self.binarized_readouts = readouts

        self.model_binarized = KerasModel(inputs=self.input, outputs=self.binarized_probs)


    def defineFullModel(self):
        self.TRAIN_FLAG=0
        outputs = self.defineModel()

        if len(outputs)>1:
            self.raw_output = Concatenate()(outputs)
        else: #if only a single chunk
            self.raw_output = outputs[0]
            
        #pass output logits through activation
        for idx,o in enumerate(outputs):
            outputs[idx] = Lambda(self.params_dict['output_activation'])(o)
            
        if len(outputs)>1:
            x = Concatenate()(outputs)
        else: #if only a single chunk
            x = outputs[0]

        x = Lambda(self.outputDecoder)(x) #logits
        logits = x
        x = Activation('softmax')(x) #return probs

        self.logits = logits
        self.probabilities = x
        
        if self.params_dict['base_model'] == None:
            self.model_full = KerasModel(inputs=self.input, outputs=x)
        else:
            self.model_full = KerasModel(inputs=self.params_dict['base_model'].input, outputs=x)


    #CleverHans model wrapper; returns a model that CH can attack  
    def modelCH(self):
        return CleverHansKerasModelWrapper(self.model_full)

    def modelBinarizedCH(self):
        return CleverHansKerasModelWrapper(self.model_binarized)
       
        
    def saveModel(self):
        w= self.model.get_weights()
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        fpkl= open(pklfile, 'wb')        
        pickle.dump(w, fpkl)
        fpkl.close()
        self.model.save(self.params_dict['model_path'] + self.params_dict['name'] + '_final.h5')

    
    
    def loadModel(self):
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        f= open(pklfile, 'rb')
        weigh= pickle.load(f);  
        f.close();
        self.defineModel()
        self.model.set_weights(weigh)
        
        
    def loadFullModel(self):
        pklfile= self.params_dict['model_path'] + self.params_dict['name'] + '_final.pkl'
        f= open(pklfile, 'rb')
        weigh= pickle.load(f);  
        f.close();
        self.defineFullModel()
        self.model_full.set_weights(weigh)
        
        
    def predict(self, X):
        return self.model_full(X)
