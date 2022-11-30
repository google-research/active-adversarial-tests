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
A pure TensorFlow implementation of a neural network. This can be
used as a drop-in replacement for a Keras model.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf
from modified_cleverhans.model import Model

BN_EPSILON = 1e-5

## For alexnet's local response normalization 
RADIUS = 2; ALPHA = 2E-05; BETA = 0.75; BIAS = 1.0 # values copied from myalexnet_forward_newtf.py

@tf.RegisterGradient("QuantizeGrad")
def quantize_grad(op, grad):
    return tf.clip_by_value(tf.identity(grad), -1, 1)


def hard_sigmoid(x):
    return tf.cast(tf.clip_by_value((x + 1.) / 2., 0., 1.), tf.float32)


class MLP(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, layers, input_shape):
        super(MLP, self).__init__()

        self.layer_names = []
        self.layers = layers
        self.input_shape = input_shape
        if isinstance(layers[-1], Softmax):
            layers[-1].name = 'probs'
            layers[-2].name = 'logits'
        else:
            layers[-1].name = 'logits'
        for i, layer in enumerate(self.layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()
        print(self.layer_names)

    def fprop(self, x, reuse, set_ref=False):
        states = []
        for layer in self.layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)
        states = dict(zip(self.get_layer_names(), states))
        return states

# special distilled model class consisting of a teacher and a student model
class distilledModel(Model):
    """
    An example of a bare bones multilayer perceptron (MLP) class.
    """

    def __init__(self, teacher_layers, student_layers, input_shape):
        super(distilledModel, self).__init__()

        self.layer_names = []
        self.teacher_layers = teacher_layers
        self.student_layers = student_layers
        self.input_shape = input_shape
        original_input_shape = input_shape
        if isinstance(teacher_layers[-1], Softmax):
            teacher_layers[-1].name = 'teacher_probs'
            teacher_layers[-2].name = 'teacher_logits'
        else:
            layers[-1].name = 'teacher_logits'
        for i, layer in enumerate(self.teacher_layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()
        
        input_shape = original_input_shape
        if isinstance(student_layers[-1], Softmax):
            student_layers[-1].name = 'probs'
            student_layers[-2].name = 'logits'
        else:
            student_layers[-1].name = 'logits'
        for i, layer in enumerate(self.student_layers):
            if hasattr(layer, 'name'):
                name = layer.name
            else:
                name = layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()

        print(self.layer_names)

    def fprop(self, x, reuse, set_ref=False):
        states = []
        original_x = x
        for layer in self.teacher_layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)
        x = original_x
        num_student_layers = len(self.student_layers)
        layer_count = 0
        for layer in self.student_layers:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)
            layer_count = layer_count + 1
        states = dict(zip(self.get_layer_names(), states))
        return states

# ensembleThreeModel class build on Model class that forms ensemble of three models
class ensembleThreeModel(Model):
    """
    An example ensemble model.
    """

    def __init__(self, layers1, layers2, layers3, input_shape, num_classes): #layers1: layers of model1, layers2: layers of model2
        super(ensembleThreeModel, self).__init__()

        self.layer_names = []
        self.layers1 = layers1
        self.layers2 = layers2
        self.layers3 = layers3
        self.input_shape = input_shape
        self.num_classes = num_classes
        original_input_shape = input_shape
        if isinstance(layers1[-1], Softmax):
            layers1[-1].name = 'probs'
            layers1[-2].name = 'logits'
        else:
            layers1[-1].name = 'logits'
        # First model
        for i, layer in enumerate(self.layers1):
            if hasattr(layer, 'name'):
                if layer.name == 'probs' or layer.name == 'logits': 
                    name = layer.name
                else: 
                    name = 'Model1_' + layer.name
            else:
                name = 'Model1_' + layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()
        
        input_shape = original_input_shape
        # Second model
        if isinstance(layers2[-1], Softmax):
            layers2[-1].name = 'probs'
            layers2[-2].name = 'logits'
        else:
            layers2[-1].name = 'logits'
        for i, layer in enumerate(self.layers2):
            if hasattr(layer, 'name'):
                if layer.name == 'probs' or layer.name == 'logits': 
                    name = layer.name
                else:
                    name = 'Model2_' + layer.name
            else:
                name = 'Model2_' + layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()
        input_shape = original_input_shape
        # Third model
        if isinstance(layers3[-1], Softmax):
            layers3[-1].name = 'probs'
            layers3[-2].name = 'logits'
        else:
            layers3[-1].name = 'logits'
        for i, layer in enumerate(self.layers3):
            if hasattr(layer, 'name'):
                if layer.name == 'probs' or layer.name == 'logits': 
                    name = layer.name
                else:
                    name = 'Model3_' + layer.name
            else:
                name = 'Model3_' + layer.__class__.__name__ + str(i)
            self.layer_names.append(name)

            layer.set_input_shape(input_shape, False)
            input_shape = layer.get_output_shape()
        self.layer_names.append('combined_features')
        self.layer_names.append('combined_logits')

        combined_layer_name = 'combined' ## Gives the final class prediction based on max voting
        self.layer_names.append(combined_layer_name)
        combinedCorrectProb_layer_name = 'combinedAvgCorrectProb' ## Gives average probability values of the models that decided the final prediction
        self.layer_names.append(combinedCorrectProb_layer_name)
        combinedProb_layer_name = 'combinedAvgProb' ## Gives average probability values of all the models 
        self.layer_names.append(combinedProb_layer_name)
        
        print(self.layer_names)

    def fprop(self, x, reuse, set_ref=False):
        states = []
        original_x = x
        for layer in self.layers1:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)

        output1 = states[-1]
        features1 = states[-3]
        x = original_x
        for layer in self.layers2:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)

        features2 = states[-3]
        output2 = states[-1]
        x = original_x
        for layer in self.layers3:
            if set_ref:
                layer.ref = x
            x = layer.fprop(x, reuse)
            assert x is not None
            states.append(x)
        output3 = states[-1]
        features3 = states[-3]

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
        rand_idx = np.random.random_integers(3)
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
            rand_idx = np.random.random_integers(3)
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

class Layer(object):

    def get_output_shape(self):
        return self.output_shape


class SimpleLinear(Layer):

    def __init__(self, num_hid):
        self.num_hid = num_hid

    def set_input_shape(self, input_shape, reuse):
        batch_size, dim = input_shape
        self.input_shape = [batch_size, dim]
        self.output_shape = [batch_size, self.num_hid]
        init = tf.random_normal([dim, self.num_hid], dtype=tf.float32)
        init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init), axis=0,
                                                   keep_dims=True))
        self.W = tf.Variable(init)
        self.b = tf.Variable(np.zeros((self.num_hid,)).astype('float32'))

    def fprop(self, x, reuse):
        return tf.matmul(x, self.W) + self.b


class Linear(Layer):

    def __init__(self, num_hid, detail, useBias=False):
        self.__dict__.update(locals())
        # self.num_hid = num_hid

    def set_input_shape(self, input_shape, reuse):

        # with tf.variable_scope(self.scope_name+ 'init', reuse): # this works
        # with black box, but now can't load checkpoints from wb
        # this works with white-box
        with tf.variable_scope(self.detail + self.name + '_init', reuse):

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
            W_summ = tf.summary.histogram('W', values=self.W)
            if self.useBias:
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)

    def fprop(self, x, reuse):

        # with tf.variable_scope(self.scope_name + '_fprop', reuse):
        # this works with white-box
        with tf.variable_scope(self.detail + self.name + '_fprop', reuse):

            x = tf.matmul(x, self.W)  # + self.b
            if self.useBias:
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class HiddenLinear(Layer):

    def __init__(self, num_hid, scope_name, useBias=False):
        self.__dict__.update(locals())

    def set_input_shape(self, input_shape, reuse):

        with tf.variable_scope(self.scope_name+ 'init', reuse):

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
                self.bias =tf.get_variable("b", initializer= bias_init)

            W_summ = tf.summary.histogram('W', values=self.W)

    def fprop(self, x, reuse):

        with tf.variable_scope(self.scope_name + '_fprop', reuse):

            x = tf.matmul(x, self.W)  
            if self.useBias:
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class HiddenLinear_lowprecision(Layer):

    # def __init__(self, num_hid, scope_name):
    def __init__(self, wbits, abits, num_hid, scope_name, useBias=False):
        self.__dict__.update(locals())

    def quantize(self, x, k): ## k= No. of quantized bits
        n = float(2**k-1) ## Max value representable with k bits
        
        @tf.custom_gradient ## Can be used to define a custom gradient function
        def _quantize(x):
            return tf.round(x*n)/n, lambda dy: dy # Second part is the function evaluated during gradient, identity function

        return _quantize(x) 

    def quantizeWt(self, x):
        x = tf.tanh(x) ## Normalizing weights to [-1, 1]
        x = x/tf.reduce_max(abs(x))*0.5 + 0.5 ## Normalizing weights to [0, 1]
        return 2*self.quantize(x, self.wbits) - 1 ## Normalizing back to [0, 1] after quantizing
    
    def quantizeAct(self, x):
        x = tf.clip_by_value(x, 0, 1.0) ## Normalizing activations to [0, 1] --> performed in nonlin(x) function of alexnet-dorefa.py
        return self.quantize(x, self.abits)

    def set_input_shape(self, input_shape, reuse):

        with tf.variable_scope(self.scope_name+ 'init', reuse): # this works
        # with black box, but now can't load checkpoints from wb
        # this works with white-box
        # with tf.variable_scope(self.detail + self.name + '_init', reuse):

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
            
            if (self.wbits < 32):  
                self.W = self.quantizeWt(self.W)
            
            if self.useBias:
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)

            W_summ = tf.summary.histogram('W', values=self.W)

    def fprop(self, x, reuse):

        with tf.variable_scope(self.scope_name + '_fprop', reuse):
        # this works with white-box
        # with tf.variable_scope(self.detail + self.name + '_fprop', reuse):
            if self.abits < 32:
                x = self.quantizeAct(x)

            x = tf.matmul(x, self.W)  # + self.b
            if self.useBias:
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class Conv2DRand(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding, phase, scope_name):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize_rand(self, x, dist):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return 2 * dist(probs=hard_sigmoid(x)).sample() - 1

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    def set_input_shape(self, input_shape, reuse):

        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse):

            init = tf.truncated_normal(
                kernel_shape, stddev=0.2, dtype=tf.float32)
            self.kernels = tf.get_variable("k", initializer=init)
            k_summ = tf.summary.histogram(
                name="k", values=self.kernels)

            from tensorflow.contrib.distributions import MultivariateNormalDiag
            with self.G.gradient_override_map({"MultivariateNormalDiag": "QuantizeGrad"}):
                self.kernels = MultivariateNormalDiag(
                    loc=self.kernels).sample()

            k_rand_summ = tf.summary.histogram(
                name="k_rand", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            dummy_output = self.fprop(dummy_batch, False)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)

    def fprop(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            x = tf.nn.conv2d(x, self.kernels, (1,) +
                             tuple(self.strides) + (1,), self.padding)
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x


class Conv2D(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding, phase, scope_name, useBias=False):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    def set_input_shape(self, input_shape, reuse):

        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse):

            init = tf.truncated_normal(
                kernel_shape, stddev=0.1, dtype=tf.float32)
            init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                           axis=(0, 1, 2)))
            self.kernels = tf.get_variable("k", initializer=init)
            k_summ = tf.summary.histogram(
                name="k", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            # Set output shape using fprop without bias if useBias set 
            if self.useBias:
                dummy_output = self.fprop_withoutbias(dummy_batch, False)
            else: #--default below
                dummy_output = self.fprop(dummy_batch, False)

            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)
            
            if self.useBias:
                self.bias_shape = self.output_shape
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)

    def fprop(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            x = tf.nn.conv2d(x, self.kernels, (1,) +
                             tuple(self.strides) + (1,), self.padding)

            if self.useBias:
                output_shape = tf.shape(x) # Checking output shape before bias
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))
                x = tf.reshape(x, output_shape)

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x
    
    # special function without bias to get output shape
    def fprop_withoutbias(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            x = tf.nn.conv2d(x, self.kernels, (1,) +
                             tuple(self.strides) + (1,), self.padding)
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class Conv2D_lowprecision(Layer):

    def __init__(self, wbits, abits, output_channels, kernel_shape, strides, padding, phase, scope_name, seed=1, useBatchNorm=False, stocRound=False, useBias=False):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize(self, x, k): ## k= No. of quantized bits
        n = float(2**k-1) ## Max value representable with k bits
        
        @tf.custom_gradient ## Can be used to define a custom gradient function
        def _quantize(x):
            if self.stocRound: # If stochastic rounding is set
                xn_int = tf.floor(x*n) # Get integer part
                xn_frac = tf.subtract(x*n, xn_int) # Get fractional part
                xn_frac_rand = tf.distributions.Bernoulli(probs=xn_frac, dtype=tf.float32).sample() # Get random number from bernoulli distribution with prob=fractional part value
                x_q = (xn_int + xn_frac_rand)/n
                
                return x_q, lambda dy: dy # Second part is the function evaluated during gradient, identity function
            else:
                return tf.round(x*n)/n, lambda dy: dy # Second part is the function evaluated during gradient, identity function

        return _quantize(x) 

    def quantizeWt(self, x):
        x = tf.tanh(x) ## Normalizing weights to [-1, 1]
        x = x/tf.reduce_max(abs(x))*0.5 + 0.5 ## Normalizing weights to [0, 1]
        return 2*self.quantize(x, self.wbits) - 1 ## Normalizing back to [0, 1] after quantizing
    
    def quantizeAct(self, x):
        x = tf.clip_by_value(x, 0, 1.0) ## Normalizing activations to [0, 1] --> performed in nonlin(x) function of alexnet-dorefa.py
        return self.quantize(x, self.abits)

    def set_input_shape(self, input_shape, reuse):

        batch_size, rows, cols, input_channels = input_shape
        kernel_shape = tuple(self.kernel_shape) + (input_channels,
                                                   self.output_channels)
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse):

            if self.wbits < 32:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.2, dtype=tf.float32)
                init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                               axis=(0, 1, 2)))
                self.kernels = tf.get_variable("k", initializer=init)
            else:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.1, dtype=tf.float32)
                init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                               axis=(0, 1, 2)))
                self.kernels = tf.get_variable("k", initializer=init)

            if (self.wbits < 32): ## Quantize if no. of bits less than 32
                self.kernels = self.quantizeWt(self.kernels)
                k_bin_summ = tf.summary.histogram(
                    name="k_bin", values=self.kernels)

            k_summ = tf.summary.histogram(
                name="k", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            
            # Set output shape using fprop without bias if useBias set 
            if self.useBias:
                dummy_output = self.fprop_withoutbias(dummy_batch, False)
            else: #--default below
                dummy_output = self.fprop(dummy_batch, False)
            
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)
            
            # setting bias shape
            if self.useBias:
                self.bias_shape = self.output_shape
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)
                if self.wbits < 32: ## Quantize if no. of bits less than 32
                    self.bias =self.quantizeWt(self.bias)

    def fprop(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:
            
            if self.abits < 32:
                if self.useBatchNorm: ## Specifies whether we want to use Batch Normalization or not
                    x = tf.contrib.layers.batch_norm(
                        x, epsilon=BN_EPSILON, is_training=self.phase,
                        reuse=reuse, scope=scope)
                x = self.quantizeAct(x)
            x = tf.nn.conv2d(x, self.kernels, (1,) +
                             tuple(self.strides) + (1,), self.padding)

            if self.useBias:
                output_shape = tf.shape(x) # Checking output shape before bias
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))
                x = tf.reshape(x, output_shape)

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

    # special function without bias to get output shape
    def fprop_withoutbias(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            x = tf.nn.conv2d(x, self.kernels, (1,) +
                             tuple(self.strides) + (1,), self.padding)
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class Conv2DGroup(Layer):

    def __init__(self, output_channels, kernel_shape, strides, padding, phase, scope_name, useBias=False):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize(self, x):
        with self.G.gradient_override_map({"Sign": "QuantizeGrad"}):
            return tf.sign(x)

    def set_input_shape(self, input_shape, reuse):

        self.input_shape = input_shape 
        batch_size, rows, cols, input_channels = input_shape
        self.input_channels = input_channels
        kernel_shape = tuple(self.kernel_shape) + (int(input_channels/2),
                                                   self.output_channels) # as it is 2 groups, input channel dimension is halved
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse):

            init = tf.variance_scaling_initializer(scale=2., dtype=tf.float32)
            self.kernels = tf.get_variable("k", shape=kernel_shape, initializer=init)
            k_summ = tf.summary.histogram(
                name="k", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            if self.useBias:
                dummy_output = self.fprop_withoutbias(dummy_batch, False)
            else: #--default below
                dummy_output = self.fprop(dummy_batch, False)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)

            # setting bias shape
            self.bias_shape = self.output_shape

            # initializing bias
            if self.useBias:
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)


    def fprop(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            ### groupwise convolution
            x1 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.input_channels/2, tf.int32)]), tf.slice(self.kernels, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.output_channels/2, tf.int32)]), (1,) + tuple(self.strides) + (1,), self.padding)
            x2 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, tf.cast(self.input_channels/2, tf.int32)], [-1, -1, -1, -1]), tf.slice(self.kernels, [0, 0, 0, (tf.cast(self.output_channels/2, tf.int32))], [-1, -1, -1, -1]), (1,) + tuple(self.strides) + (1,), self.padding)
            x = tf.concat([x1, x2], 3)
            
            # adding bias
            if self.useBias:
                output_shape = tf.shape(x) # Checking output shape before bias
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))
                if self.padding=="SAME": # Padding same means input and output size equal
                    x = tf.reshape(x, output_shape)

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x
    
    # Special function without bias to get output shape
    def fprop_withoutbias(self, x, reuse):

        # need variable_scope here because the batch_norm layer creates
        # variables internally
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            ### groupwise convolution
            x1 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.input_channels/2, tf.int32)]), tf.slice(self.kernels, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.output_channels/2, tf.int32)]), (1,) + tuple(self.strides) + (1,), self.padding)
            x2 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, tf.cast(self.input_channels/2, tf.int32)], [-1, -1, -1, -1]), tf.slice(self.kernels, [0, 0, 0, (tf.cast(self.output_channels/2, tf.int32))], [-1, -1, -1, -1]), (1,) + tuple(self.strides) + (1,), self.padding)
            x = tf.concat([x1, x2], 3)
            
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class Conv2DGroup_lowprecision(Layer):

    def __init__(self, wbits, abits, output_channels, kernel_shape, strides, padding, phase, scope_name, seed=1, useBatchNorm=False, stocRound=False, useBias=False):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def quantize(self, x, k): ## k= No. of quantized bits
        n = float(2**k-1) ## Max value representable with k bits
        
        @tf.custom_gradient ## Can be used to define a custom gradient function
        def _quantize(x):
            if self.stocRound: # If stochastic rounding is set
                xn_int = tf.floor(x*n) # Get integer part
                xn_frac = tf.subtract(x*n, xn_int) # Get fractional part
                xn_frac_rand = tf.distributions.Bernoulli(probs=xn_frac, dtype=tf.float32).sample() # Get random number from bernoulli distribution with prob=fractional part value
                x_q = (xn_int + xn_frac_rand)/n
                
                return x_q, lambda dy: dy # Second part is the function evaluated during gradient, identity function
            else:
                return tf.round(x*n)/n, lambda dy: dy # Second part is the function evaluated during gradient, identity function

        return _quantize(x) 

    def quantizeWt(self, x):
        x = tf.tanh(x) ## Normalizing weights to [-1, 1]
        x = x/tf.reduce_max(abs(x))*0.5 + 0.5 ## Normalizing weights to [0, 1]
        return 2*self.quantize(x, self.wbits) - 1 ## Normalizing back to [0, 1] after quantizing
    
    def quantizeAct(self, x):
        x = tf.clip_by_value(x, 0, 1.0) ## Normalizing activations to [0, 1] --> performed in nonlin(x) function of alexnet-dorefa.py
        return self.quantize(x, self.abits)

    def set_input_shape(self, input_shape, reuse):

        self.input_shape = input_shape 
        batch_size, rows, cols, input_channels = input_shape
        self.input_channels = input_channels
        kernel_shape = tuple(self.kernel_shape) + (int(input_channels/2),
                                                   self.output_channels) # as it is 2 groups, input channel dimension is halved
        assert len(kernel_shape) == 4
        assert all(isinstance(e, int) for e in kernel_shape), kernel_shape

        with tf.variable_scope(self.scope_name + '_init', reuse):

            if self.wbits < 32:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.2, dtype=tf.float32)
                self.kernels = tf.get_variable("k", initializer=init)
            else:
                init = tf.truncated_normal(
                    kernel_shape, stddev=0.1, dtype=tf.float32)
                init = init / tf.sqrt(1e-7 + tf.reduce_sum(tf.square(init),
                                                           axis=(0, 1, 2)))
                self.kernels = tf.get_variable("k", initializer=init)

            if (self.wbits < 32): ## Quantize if no. of bits less than 32
                self.kernels = self.quantizeWt(self.kernels)
                k_bin_summ = tf.summary.histogram(
                    name="k_bin", values=self.kernels)
            k_summ = tf.summary.histogram(
                name="k", values=self.kernels)

            orig_input_batch_size = input_shape[0]
            input_shape = list(input_shape)
            input_shape[0] = 1
            dummy_batch = tf.zeros(input_shape)
            # Set output shape using fprop without bias if useBias set 
            if self.useBias:
                dummy_output = self.fprop_withoutbias(dummy_batch, False)
            else: #--default below
                dummy_output = self.fprop(dummy_batch, False)
            output_shape = [int(e) for e in dummy_output.get_shape()]
            output_shape[0] = 1
            self.output_shape = tuple(output_shape)

            self.bias_shape = self.output_shape

            if self.useBias:
                bias_init = tf.zeros(self.bias_shape)
                self.bias =tf.get_variable("b", initializer= bias_init)
                if self.wbits < 32: ## Quantize if no. of bits less than 32
                    self.bias =self.quantizeWt(self.bias)


    def fprop(self, x, reuse):
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            if self.abits < 32:
                if self.useBatchNorm: ## Specifies whether we want to use Batch Normalization or not
                    x = tf.contrib.layers.batch_norm(
                        x, epsilon=BN_EPSILON, is_training=self.phase,
                        reuse=reuse, scope=scope)
                x = self.quantizeAct(x)
            ### groupwise convolution
            x1 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.input_channels/2, tf.int32)]), tf.slice(self.kernels, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.output_channels/2, tf.int32)]), (1,) + tuple(self.strides) + (1,), self.padding)
            x2 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, tf.cast(self.input_channels/2, tf.int32)], [-1, -1, -1, -1]), tf.slice(self.kernels, [0, 0, 0, (tf.cast(self.output_channels/2, tf.int32))], [-1, -1, -1, -1]), (1,) + tuple(self.strides) + (1,), self.padding)
            x = tf.concat([x1, x2], 3)
            
            if self.useBias:
                output_shape = tf.shape(x) # Checking output shape before bias
                x = tf.nn.bias_add(tf.contrib.layers.flatten(x), tf.reshape(self.bias, [-1]))
                if self.padding=="SAME": # Padding same means input and output size equal
                    x = tf.reshape(x, output_shape)

            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x
    
    # Special function without bias to get output shape
    def fprop_withoutbias(self, x, reuse):
        with tf.variable_scope(self.scope_name + '_fprop', reuse=reuse) as scope:

            if self.abits < 32:
                if self.useBatchNorm: ## Specifies whether we want to use Batch Normalization or not
                    x = tf.contrib.layers.batch_norm(
                        x, epsilon=BN_EPSILON, is_training=self.phase,
                        reuse=reuse, scope=scope)
                x = self.quantizeAct(x)
            ### groupwise convolution
            x1 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.input_channels/2, tf.int32)]), tf.slice(self.kernels, [0, 0, 0, 0], [-1, -1, -1, tf.cast(self.output_channels/2, tf.int32)]), (1,) + tuple(self.strides) + (1,), self.padding)
            x2 = tf.nn.conv2d(tf.slice(x, [0, 0, 0, tf.cast(self.input_channels/2, tf.int32)], [-1, -1, -1, -1]), tf.slice(self.kernels, [0, 0, 0, (tf.cast(self.output_channels/2, tf.int32))], [-1, -1, -1, -1]), (1,) + tuple(self.strides) + (1,), self.padding)
            x = tf.concat([x1, x2], 3)
            
            a_u, a_v = tf.nn.moments(tf.abs(x), axes=[0], keep_dims=False)
            a_summ = tf.summary.histogram('a', values=x)
            a_u_summ = tf.summary.scalar("a_u", tf.reduce_mean(a_u))
            a_v_summ = tf.summary.scalar("a_v", tf.reduce_mean(a_v))

            return x

class MaxPool(Layer):
    
    def __init__ (self, pool_size, strides):
        self.pool_size = pool_size
        self.strides = strides
        
    def set_input_shape(self, input_shape, reuse):
        self.input_shape = input_shape
        orig_input_batch_size = input_shape[0]
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch, False)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x, reuse):
        return tf.layers.max_pooling2d(x, self.pool_size, self.strides)

class MaxPoolSame(Layer):
    
    def __init__ (self, pool_size, strides):
        self.pool_size = pool_size
        self.strides = strides
        
    def set_input_shape(self, input_shape, reuse):
        self.input_shape = input_shape
        orig_input_batch_size = input_shape[0]
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch, False)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x, reuse):
        return tf.layers.max_pooling2d(x, self.pool_size, self.strides, padding='same')

class AvgPool(Layer):
    
    def __init__ (self, pool_size, strides):
        self.pool_size = pool_size
        self.strides = strides
        
    def set_input_shape(self, input_shape, reuse):
        self.input_shape = input_shape
        orig_input_batch_size = input_shape[0]
        input_shape = list(input_shape)
        input_shape[0] = 1
        dummy_batch = tf.zeros(input_shape)
        dummy_output = self.fprop(dummy_batch, False)
        output_shape = [int(e) for e in dummy_output.get_shape()]
        output_shape[0] = 1
        self.output_shape = tuple(output_shape)

    def fprop(self, x, reuse):
        return tf.layers.average_pooling2d(x, self.pool_size, self.strides) 

class ReLU(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, reuse):
        return tf.nn.relu(x)


class SReLU(Layer):

    def __init__(self, scope_name):
        self.scope_name = scope_name
        pass

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape
        with tf.variable_scope(self.scope_name + '_init', reuse=reuse):
            self.activation_scalar = tf.get_variable(
                "activation_scalar", initializer=0.05, trainable=True)

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, reuse):
        with tf.variable_scope(self.scope_name + '_init', reuse=reuse):
            return tf.nn.relu(x) * self.activation_scalar


class Softmax(Layer):

    def __init__(self, temperature):
        self.temperature = temperature

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x, reuse):
        return tf.nn.softmax(x * self.temperature)


class SoftmaxT1(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def fprop(self, x, reuse):
        return tf.nn.softmax(x)


class Flatten(Layer):

    def __init__(self):
        pass

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        output_width = 1
        for factor in shape[1:]:
            output_width *= factor
        self.output_width = output_width
        self.output_shape = [None, output_width]

    def fprop(self, x, reuse):
        return tf.reshape(x, [-1, self.output_width])

# Local response Norm layer for AlexNet
class LocalNorm(Layer):

    def __init__(self):
        self.__dict__.update(locals())

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, reuse):
        x = tf.nn.local_response_normalization(x,
                                               depth_radius=RADIUS,
                                               alpha=ALPHA,
                                               beta=BETA,
                                               bias=BIAS)
        return x

# BatchNorm layer for low precision alexnet
class BatchNorm(Layer):

    def __init__(self, phase, scope_name, mean=None, variance=None, scale=None, offset=None):
        self.__dict__.update(locals())

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, reuse):
       
        # Batch normalization for the training phase
        if (self.mean is None) and (self.variance is None) and (self.scale is None) and (self.offset is None):
            with tf.variable_scope(self.scope_name, reuse=tf.AUTO_REUSE): # Adding scope here to help in restoring variables, Saves and restores model 
                x = tf.layers.batch_normalization(x, training=self.phase)
        else:
            x = tf.nn.batch_normalization(
                    x, mean=self.mean, variance=self.variance, 
                    scale=self.scale, offset=self.offset, variance_epsilon=BN_EPSILON)
        return x

## dropout layer for alexnet
class DropOut(Layer):

    def __init__(self, keep_prob, phase):
        self.__dict__.update(locals())
        self.G = tf.get_default_graph()
        del self.self

    def set_input_shape(self, shape, reuse):
        self.input_shape = shape
        self.output_shape = shape

    def get_output_shape(self):
        return self.output_shape

    def fprop(self, x, reuse):
        return tf.cond(self.phase, lambda: tf.nn.dropout(x, self.keep_prob), lambda: tf.identity(x)) # Dropout during training phase but not during test phase

######################### full-precision #########################
def make_basic_cnn(phase, temperature, detail, nb_filters=64, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME", phase, detail + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail + 'conv2'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail + 'conv3'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic cnn')
    return model


def make_scaled_rand_cnn(phase, temperature, detail, nb_filters=64, nb_classes=10,
                         input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME", phase, detail + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail + 'conv2'),
              ReLU(),
              Conv2DRand(nb_filters * 2, (5, 5),
                         (1, 1), "VALID", phase, detail + 'conv3'),
              SReLU(detail + 'srelu3_fp'),
              Flatten(),
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic cnn')
    return model

# distilled model
def make_distilled_cnn(phase, temperature, detail1, detail2, nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1)):
    # make one teacher low precision cnn with wbits precision weights and abits activations
    teacher_layers = [Conv2D(nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail1 + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail1 + 'conv2_bin'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail1 + 'conv3_bin'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail1),
              Softmax(temperature)] # Hard probs (default)
    # make one student low precision cnn with wbits precision weights and abits activations
    student_layers = [Conv2D(nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail2 + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail2 + 'conv2'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail2 + 'conv3'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail2),
              Softmax(temperature)] # Hard probs (default)

    model = distilledModel(teacher_layers, student_layers, input_shape)
    print('Finished making a distilled cnn')

    return model

################## low precision version of mnist cnn #################
def make_basic_lowprecision_cnn(phase, temperature, detail, wbits, abits, nb_filters=64, nb_classes=10,
                          input_shape=(None, 28, 28, 1), useBatchNorm=False, stocRound=False):

    layers = [Conv2D_lowprecision(wbits, abits, nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail + 'conv1', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Conv2D_lowprecision(wbits, abits, nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail + 'conv2_bin', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Conv2D_lowprecision(wbits, abits, nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail + 'conv3_bin', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail), 
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic low precision cnn: %d weight bits, %d activation bits' %(wbits, abits))
    return model

# Variant of low precision supporting different precisions for different layers
def make_layerwise_lowprecision_cnn(phase, temperature, detail, wbits, abits, nb_filters=64, 
            nb_classes=10, input_shape=(None, 28, 28, 1), 
            useBatchNorm=False, stocRound=False):
    layers = [Conv2D_lowprecision(wbits[0], abits[0], nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail + 'conv1', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Conv2D_lowprecision(wbits[1], abits[1], nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail + 'conv2_bin', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Conv2D_lowprecision(wbits[2], abits[2], nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail + 'conv3_bin', useBatchNorm=useBatchNorm, stocRound=stocRound),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making layerwise low precision cnn: %d %d %d weight bits, %d %d %d activation bits' %(wbits[0], wbits[1], wbits[2], abits[0], abits[1], abits[2]))
    return model


################## EMPIR version of mnist cnn #################
def make_ensemble_three_cnn(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1), useBatchNorm=False):
    # make one low precision cnn with wbits precision weights and abits activations
    layers1 = [Conv2D_lowprecision(wbits1, abits1, nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail1 + 'conv1', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits1, abits1, nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail1 + 'conv2_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits1, abits1, nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail1 + 'conv3_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail1),
              Softmax(temperature)]
    # make another low precision cnn with wbits precision weights and abits activations
    layers2 = [Conv2D_lowprecision(wbits2, abits2, nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail2 + 'conv1', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits2, abits2, nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail2 + 'conv2_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits2, abits2, nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail2 + 'conv3_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail2),
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and a bits activations
    layers3 = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME", phase, detail3 + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail3 + 'conv2'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail3 + 'conv3'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail3),
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, nb_classes)
    print('Finished making ensemble of three cnns')

    return model

def make_ensemble_three_cnn_layerwise(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=64, nb_classes=10, input_shape=(None, 28, 28, 1), useBatchNorm=False):
    # make one low precision cnn with wbits precision weights and abits activations
    layers1 = [Conv2D_lowprecision(wbits1[0], abits1[0], nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail1 + 'conv1', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits1[1], abits1[1], nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail1 + 'conv2_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits1[2], abits1[2], nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail1 + 'conv3_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail1),
              Softmax(temperature)]
    # make another low precision cnn with wbits precision weights and abits activations
    layers2 = [Conv2D_lowprecision(wbits2[0], abits2[0], nb_filters, (8, 8),
                     (2, 2), "SAME", phase, detail2 + 'conv1', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits2[1], abits2[1], nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail2 + 'conv2_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Conv2D_lowprecision(wbits2[2], abits2[2], nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail2 + 'conv3_bin', useBatchNorm=useBatchNorm),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail2),
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and a bits activations
    layers3 = [Conv2D(nb_filters, (8, 8), (2, 2), "SAME", phase, detail3 + 'conv1'),
              ReLU(),
              Conv2D(nb_filters * 2, (6, 6),
                     (2, 2), "VALID", phase, detail3 + 'conv2'),
              ReLU(),
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "VALID", phase, detail3 + 'conv3'),
              ReLU(),
              Flatten(),
              Linear(nb_classes, detail3),
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, avg, weightedAvg, alpha, nb_classes)
    print('Finished making ensemble of three cnns')

    return model

################# full-precision cifar cnn ############################
def make_basic_cifar_cnn(phase, temperature, detail, nb_filters=32, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    layers = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", phase, detail + 'conv1'),  
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Conv2D(nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), 
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), 
              Flatten(),
              HiddenLinear(64, detail + 'ip1'), 
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic cnn')
    return model

################## distilled version of cifar cnn #################
def make_distilled_cifar_cnn(phase, temperature, detail1, detail2, nb_filters=32, nb_classes=10,
                   input_shape=(None, 28, 28, 1)):
    teacher_layers = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", phase, detail1 + 'conv1'),  
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D(nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Flatten(),
              HiddenLinear(64, detail1 + 'ip1'), 
              Linear(nb_classes, detail1),
              Softmax(temperature)]
    
    student_layers = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", phase, detail2 + 'conv1'),  
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D(nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Flatten(),
              HiddenLinear(64, detail2 + 'ip1'), 
              Linear(nb_classes, detail2),
              Softmax(temperature)]

    model = distilledModel(teacher_layers, student_layers, input_shape)
    print('Finished making distilled cifar cnn')
    return model


################## low precision version of cifar cnn #################
def make_basic_lowprecision_cifar_cnn(phase, temperature, detail, wbits, abits, nb_filters=64, nb_classes=10,
                          input_shape=(None, 28, 28, 1), stocRound=False):

    layers = [Conv2D_lowprecision(wbits, abits, nb_filters, (5, 5), (1, 1), "SAME", phase, detail + 'conv1', stocRound=stocRound), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits, abits, nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2', stocRound=stocRound),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              
              Conv2D_lowprecision(wbits, abits, nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv3', stocRound=stocRound),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail + 'ip1'), 
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic low precision cnn: %d weight bits, %d activation bits' %(wbits, abits))
    return model

def make_layerwise_lowprecision_cifar_cnn(phase, temperature, detail, wbits, abits, nb_filters=64, nb_classes=10,
                          input_shape=(None, 28, 28, 1), stocRound=False):

    layers = [Conv2D_lowprecision(wbits[0], abits[0], nb_filters, (5, 5), (1, 1), "SAME", phase, detail + 'conv1', stocRound=stocRound), 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits[1], abits[1], nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2', stocRound=stocRound),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              
              Conv2D_lowprecision(wbits[2], abits[2], nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv3', stocRound=stocRound),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail),
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making layerwise low precision cnn %d %d %d weight bits, %d %d %d activation bits' %(wbits[0], wbits[1], wbits[2], abits[0], abits[1], abits[2]))
    return model

################## EMPIR version of cifar cnn #################
def make_ensemble_three_cifar_cnn(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=32, nb_classes=10, input_shape=(None, 28, 28, 1)):
    # make a low precision cnn with full precision weights and a bits activations
    layers1 = [Conv2D_lowprecision(wbits1, abits1, nb_filters, (5, 5), (1, 1), "SAME", phase, detail1 + 'conv1'), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits1, abits1, nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              ReLU(),
              Conv2D_lowprecision(wbits1, abits1, nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail1 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail1),
              Softmax(temperature)]

    # make a low precision cnn with full precision weights and a bits activations
    layers2 = [Conv2D_lowprecision(wbits2, abits2, nb_filters, (5, 5), (1, 1), "SAME", phase, detail2 + 'conv1'), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits2, abits2, nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              ReLU(),
              Conv2D_lowprecision(wbits2, abits2, nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail2 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail2),
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and a bits activations
    layers3 = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", phase, detail3 + 'conv1'), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D(nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Flatten(),
              HiddenLinear(64, detail3 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail3),
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, nb_classes)
    print('Finished making ensemble of three cnns')

    return model

def make_ensemble_three_cifar_cnn_layerwise(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=32, nb_classes=10,
                          input_shape=(None, 28, 28, 1)):
    # make a low precision cnn with full precision weights and a bits activations
    layers1 = [Conv2D_lowprecision(wbits1[0], abits1[0], nb_filters, (5, 5), (1, 1), "SAME", phase, detail1 + 'conv1'), 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits1[1], abits1[1], nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              ReLU(),
              Conv2D_lowprecision(wbits1[2], abits1[2], nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail1 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail1),
              Softmax(temperature)]

    # make a low precision cnn with full precision weights and a bits activations
    layers2 = [Conv2D_lowprecision(wbits2[0], abits2[0], nb_filters, (5, 5), (1, 1), "SAME", phase, detail2 + 'conv1'), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits2[1], abits2[1], nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride              ReLU(),
              Conv2D_lowprecision(wbits2[2], abits2[2], nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride               
              Flatten(),
              HiddenLinear(64, detail2 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail2),
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and a bits activations
    layers3 = [Conv2D(nb_filters, (5, 5), (1, 1), "SAME", phase, detail3 + 'conv1'), # VALID padding means no padding, SAME means padding by (k-1)/2 
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D(nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv2'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Conv2D(nb_filters * 2, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv3'),
              ReLU(),
              AvgPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              Flatten(),
              HiddenLinear(64, detail3 + 'ip1'), # first f.c. layer
              Linear(nb_classes, detail3),
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, avg, weightedAvg, alpha, nb_classes)
    print('Finished making ensemble of three cifar cnns')

    return model

######################### full-precision alexnet for Imagenet #########################
def make_basic_alexnet_from_scratch(phase, temperature, detail, nb_filters=32, nb_classes=10, 
                   input_shape=(None, 28, 28, 1)): 

    layers = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup(8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2'), 
              BatchNorm(phase, detail + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2D(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv3'),
              BatchNorm(phase, detail + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2DGroup(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv4'),
              BatchNorm(phase, detail + '_batchNorm3'),
              ReLU(),
              Conv2DGroup(8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv5'),
              BatchNorm(phase, detail + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Flatten(),
              HiddenLinear(4096, detail + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail + '_batchNorm5'),
              ReLU(),
              HiddenLinear(4096, detail + 'ip2', useBias=False),
              BatchNorm(phase, detail + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail, useBias=True), 
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic alexnet')
    return model

################## low precision version of alexnet #################
def make_basic_lowprecision_alexnet(phase, temperature, detail, wbits, abits, nb_filters=32, nb_classes=10, 
                   input_shape=(None, 28, 28, 1)):
    
    layers = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits, abits, 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2'), # useBatchNorm not set here
              BatchNorm(phase, detail + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), # pool1 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits, abits, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv3'),
              BatchNorm(phase, detail + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), # pool2 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2DGroup_lowprecision(wbits, abits, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv4'),
              BatchNorm(phase, detail + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits, abits, 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv5'),
              BatchNorm(phase, detail + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits, abits, 4096, detail + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits, abits, 4096, detail + 'ip2'),
              BatchNorm(phase, detail + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail, useBias=True), # Last layer is not quantized
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making basic alexnet of low precision')
    return model

def make_layerwise_lowprecision_alexnet(phase, temperature, detail, wbits, abits, nb_filters=32, nb_classes=10, 
                   input_shape=(None, 28, 28, 1)):
    
    layers = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits[0], abits[0], 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail + 'conv2'), # useBatchNorm not set here
              BatchNorm(phase, detail + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), # pool1 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits[1], abits[1], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv3'),
              BatchNorm(phase, detail + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), # pool2 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2DGroup_lowprecision(wbits[2], abits[2], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv4'),
              BatchNorm(phase, detail + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits[3], abits[3], 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail + 'conv5'),
              BatchNorm(phase, detail + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits[4], abits[4], 4096, detail + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits[5], abits[5], 4096, detail + 'ip2'),
              BatchNorm(phase, detail + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail, useBias=True), # Last layer is not quantized
              Softmax(temperature)]

    model = MLP(layers, input_shape)
    print('Finished making layerwise alexnet of low precision')
    return model

################## EMPIR version of alexnet #################
def make_ensemble_three_alexnet(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=32, nb_classes=10,
                          input_shape=(None, 28, 28, 1), useBatchNorm=False):
    # make a low precision cnn
    layers1 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail1 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits1, abits1, 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv2'), 
              BatchNorm(phase, detail1 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2D_lowprecision(wbits1, abits1, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv3'),
              BatchNorm(phase, detail1 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2DGroup_lowprecision(wbits1, abits1, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv4'),
              BatchNorm(phase, detail1 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits1, abits1, 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv5'),
              BatchNorm(phase, detail1 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits1, abits1, 4096, detail1 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail1 + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits1, abits1, 4096, detail1 + 'ip2', useBias=False),
              BatchNorm(phase, detail1 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail1, useBias=True), 
              Softmax(temperature)]

    # make another low precision cnn
    layers2 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail2 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits2, abits2, 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv2'), 
              BatchNorm(phase, detail2 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2D_lowprecision(wbits2, abits2, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv3'),
              BatchNorm(phase, detail2 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2DGroup_lowprecision(wbits2, abits2, 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv4'),
              BatchNorm(phase, detail2 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits2, abits2, 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv5'),
              BatchNorm(phase, detail2 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits2, abits2, 4096, detail2 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail2 + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits2, abits2, 4096, detail2 + 'ip2', useBias=False),
              BatchNorm(phase, detail2 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail2, useBias=True), # Last layer is not quantized
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and activations
    layers3 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail3 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup(8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv2'), 
              BatchNorm(phase, detail3 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2D(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv3'),
              BatchNorm(phase, detail3 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2DGroup(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv4'),
              BatchNorm(phase, detail3 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup(8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv5'),
              BatchNorm(phase, detail3 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Flatten(),
              HiddenLinear(4096, detail3 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail3 + '_batchNorm5'),
              ReLU(),
              HiddenLinear(4096, detail3 + 'ip2', useBias=False),
              BatchNorm(phase, detail3 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail3, useBias=True), 
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, nb_classes)
    print('Finished making ensemble of three cnns')

    return model

def make_ensemble_three_alexnet_layerwise(phase, temperature, detail1, detail2, detail3, wbits1, abits1, wbits2, abits2, nb_filters=32, nb_classes=10,
                          input_shape=(None, 28, 28, 1)):
    # make a low precision cnn
    layers1 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail1 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits1[0], abits1[0], 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail1 + 'conv2'), 
              BatchNorm(phase, detail1 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), # pool1 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits1[1], abits1[1], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv3'),
              BatchNorm(phase, detail1 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), # pool2 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2DGroup_lowprecision(wbits1[2], abits1[2], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv4'),
              BatchNorm(phase, detail1 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits1[3], abits1[3], 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail1 + 'conv5'),
              BatchNorm(phase, detail1 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits1[4], abits1[4], 4096, detail1 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail1 + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits1[5], abits1[5], 4096, detail1 + 'ip2'),
              BatchNorm(phase, detail1 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail1, useBias=True), # Last layer is not quantized
              Softmax(temperature)]

    # make another low precision cnn
    layers2 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail2 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup_lowprecision(wbits2[0], abits2[0], 8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail2 + 'conv2'), # useBatchNorm not set here
              BatchNorm(phase, detail2 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), # pool1 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2D_lowprecision(wbits2[1], abits2[1], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv3'),
              BatchNorm(phase, detail2 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), # pool2 (3,3) pool size and (2,2) stride
              ReLU(),
              Conv2DGroup_lowprecision(wbits2[2], abits2[2], 12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv4'),
              BatchNorm(phase, detail2 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup_lowprecision(wbits2[3], abits2[3], 8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail2 + 'conv5'),
              BatchNorm(phase, detail2 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), # (3,3) pool size and (2,2) stride
              ReLU(),
              Flatten(),
              HiddenLinear_lowprecision(wbits2[4], abits2[4], 4096, detail2 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail2 + '_batchNorm5'),
              ReLU(),
              HiddenLinear_lowprecision(wbits2[5], abits2[5], 4096, detail2 + 'ip2'),
              BatchNorm(phase, detail2 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail2, useBias=True), # Last layer is not quantized
              Softmax(temperature)]

    # make a full precision cnn with full precision weights and activations
    layers3 = [Conv2D(3*nb_filters, (12, 12), (4, 4), "VALID", phase, detail3 + 'conv1', useBias=True),
              ReLU(),
              Conv2DGroup(8*nb_filters, (5, 5),
                     (1, 1), "SAME", phase, detail3 + 'conv2'), 
              BatchNorm(phase, detail3 + '_batchNorm1'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2D(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv3'),
              BatchNorm(phase, detail3 + '_batchNorm2'),
              MaxPoolSame((3, 3), (2, 2)), 
              ReLU(),
              Conv2DGroup(12*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv4'),
              BatchNorm(phase, detail3 + '_batchNorm3'),
              ReLU(),
              Conv2DGroup(8*nb_filters, (3, 3),
                     (1, 1), "SAME", phase, detail3 + 'conv5'),
              BatchNorm(phase, detail3 + '_batchNorm4'),
              MaxPool((3, 3), (2, 2)), 
              ReLU(),
              Flatten(),
              HiddenLinear(4096, detail3 + 'ip1', useBias=True), # first f.c. layer
              BatchNorm(phase, detail3 + '_batchNorm5'),
              ReLU(),
              HiddenLinear(4096, detail3 + 'ip2', useBias=False),
              BatchNorm(phase, detail3 + '_batchNorm6'),
              ReLU(),
              Linear(nb_classes, detail3, useBias=True), 
              Softmax(temperature)]

    model = ensembleThreeModel(layers1, layers2, layers3, input_shape, nb_classes)
    print('Finished making ensemble of three models')

    return model
