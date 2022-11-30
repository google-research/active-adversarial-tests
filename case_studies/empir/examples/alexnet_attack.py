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

import keras
from keras import backend
from keras.utils import np_utils

import os
import argparse
import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags

import sys 
sys.path.insert(0, "/home/consus/a/sen9/verifiedAI/cleverhans_EMPIR")

#from modified_cleverhans.attacks import fgsm
from modified_cleverhans.utils import set_log_level, parse_model_settings, build_model_save_path
from modified_cleverhans.attacks import FastGradientMethod
from modified_cleverhans.utils_keras import cnn_model
from modified_cleverhans.utils_tf import batch_eval, tf_model_load
from modified_cleverhans.utils_tf import model_train_imagenet, model_eval_imagenet, model_eval_ensemble_imagenet, model_eval_adv_imagenet, model_eval_ensemble_adv_imagenet
from examples import imagenet_preprocessing #for imagenet preprocessing

from collections import OrderedDict 

FLAGS = flags.FLAGS

ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4
MAX_BATCH_SIZE = 100

# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 0.3

# Scaling input to softmax
INIT_T = 1.0

#ATTACK_T = 1.0
ATTACK_T = 0.25

_DEFAULT_IMAGE_SIZE = 224 
_NUM_CHANNELS = 3
_NUM_CLASSES = 1000

_NUM_TRAIN_FILES = 1024
_SHUFFLE_BUFFER = 10000

def get_filenames(is_training, data_dir):
  """Return filenames for dataset."""
  if is_training:
    return [
        os.path.join(data_dir, 'Train-%05d-of-01024' % i)
        for i in range(_NUM_TRAIN_FILES)]
  else:
    return [
        os.path.join(data_dir, 'Val-%05d-of-00128' % i)
        for i in range(128)]

def _parse_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height': _int64_feature(height),
    image/width': _int64_feature(width),
    image/colorspace': _bytes_feature(colorspace),
    image/channels': _int64_feature(channels),
    image/class/label': _int64_feature(label),
    image/class/synset': _bytes_feature(synset),
    image/format': _bytes_feature(image_format),
    image/filename': _bytes_feature(os.path.basename(filename)),
    image/encoded': _bytes_feature(image_buffer)}))

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    'image/format': tf.VarLenFeature(dtype=tf.string),
    'image/filename': tf.VarLenFeature(dtype=tf.string),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string), 
  } 
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32)
  one_hot_label = tf.one_hot(label, _NUM_CLASSES, 1, 0) #convert it to a one_hot vector 

  # Directly fixing values of min and max
  xmin = tf.expand_dims([0.0], 0)
  ymin = tf.expand_dims([0.0], 0)
  xmax = tf.expand_dims([1.0], 0)
  ymax = tf.expand_dims([1.0], 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], one_hot_label, bbox

# variant of the above to parse training datasets which have labels from 1 to 1000 instead of 0 to 999
def _parse_train_example_proto(example_serialized):
  """Parses an Example proto containing a training example of an image.
  The output of the build_image_data.py image preprocessing script is a dataset
  containing serialized Example protocol buffers. Each Example proto contains
  the following fields (values are included as examples):
    image/height': _int64_feature(height),
    image/width': _int64_feature(width),
    image/colorspace': _bytes_feature(colorspace),
    image/channels': _int64_feature(channels),
    image/class/label': _int64_feature(label),
    image/class/synset': _bytes_feature(synset),
    image/format': _bytes_feature(image_format),
    image/filename': _bytes_feature(os.path.basename(filename)),
    image/encoded': _bytes_feature(image_buffer)}))

  Args:
    example_serialized: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
  Returns:
    image_buffer: Tensor tf.string containing the contents of a JPEG file.
    label: Tensor tf.int32 containing the label.
    bbox: 3-D float Tensor of bounding boxes arranged [1, num_boxes, coords]
      where each coordinate is [0, 1) and the coordinates are arranged as
      [ymin, xmin, ymax, xmax].
  """
  # Dense features in Example proto.
  feature_map = {
    'image/height': tf.FixedLenFeature([], dtype=tf.int64),
    'image/width': tf.FixedLenFeature([], dtype=tf.int64),
    'image/colorspace': tf.VarLenFeature(dtype=tf.string),
    'image/channels': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/label': tf.FixedLenFeature([], dtype=tf.int64),
    'image/class/synset': tf.VarLenFeature(dtype=tf.string),
    'image/format': tf.VarLenFeature(dtype=tf.string),
    'image/filename': tf.VarLenFeature(dtype=tf.string),
    'image/encoded': tf.FixedLenFeature([], dtype=tf.string), 
  } 
  features = tf.parse_single_example(example_serialized, feature_map)
  label = tf.cast(features['image/class/label'], dtype=tf.int32) -1 
  one_hot_label = tf.one_hot(label, _NUM_CLASSES, 1, 0) #convert it to a one_hot vector 

  # Directly fixing values of min and max
  xmin = tf.expand_dims([0.0], 0)
  ymin = tf.expand_dims([0.0], 0)
  xmax = tf.expand_dims([1.0], 0)
  ymax = tf.expand_dims([1.0], 0)

  # Note that we impose an ordering of (y, x) just to make life difficult.
  bbox = tf.concat([ymin, xmin, ymax, xmax], 0)

  # Force the variable number of bounding boxes into the shape
  # [1, num_boxes, coords].
  bbox = tf.expand_dims(bbox, 0)
  bbox = tf.transpose(bbox, [0, 2, 1])

  return features['image/encoded'], one_hot_label, bbox

def parse_record(raw_record, is_training, dtype):
  """Parses a record containing a training example of an image.
  The input record is parsed into a label and image, and the image is passed
  through preprocessing steps (cropping, flipping, and so on).
  Args:
    raw_record: scalar Tensor tf.string containing a serialized
      Example protocol buffer.
    is_training: A boolean denoting whether the input is for training.
    dtype: data type to use for images/features.
  Returns:
    Tuple with processed image tensor and one-hot-encoded label tensor.
  """
  if is_training:
    image_buffer, label, bbox = _parse_train_example_proto(raw_record)
  else:
    image_buffer, label, bbox = _parse_example_proto(raw_record)

  image = imagenet_preprocessing.preprocess_image4( # For pretrained Dorefanet network with division by standard deviation 
      image_buffer=image_buffer,
      bbox=bbox,
      output_height=_DEFAULT_IMAGE_SIZE,
      output_width=_DEFAULT_IMAGE_SIZE,
      num_channels=_NUM_CHANNELS,
      is_training=is_training)
  
  image = tf.cast(image, dtype)

  return image, label

def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=1,
                           dtype=tf.float32,
                           datasets_num_private_threads=None,
                           num_parallel_batches=1):
  """Given a Dataset with raw records, return an iterator over the records.
  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    dtype: Data type to use for images/features.
    datasets_num_private_threads: Number of threads for a private
      threadpool created for all datasets computation.
    num_parallel_batches: Number of parallel batches for tf.data.
  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """

  # Prefetches a batch at a time to smooth out the time taken to load input
  # files for shuffling and processing.
  dataset = dataset.prefetch(buffer_size=batch_size) 
  if is_training:
    # Shuffles records before repeating to respect epoch boundaries.
    dataset = dataset.shuffle(buffer_size=shuffle_buffer)

  # Repeats the dataset for the number of epochs to train.

  # Parses the raw records into images and labels.
  dataset = dataset.apply(
      tf.contrib.data.map_and_batch(
          lambda value: parse_record_fn(value, is_training, dtype),
          batch_size=batch_size,
          num_parallel_batches=num_parallel_batches))

  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  # dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)
  dataset = dataset.prefetch(buffer_size=1) 

  return dataset

def data_imagenet(nb_epochs, batch_size, imagenet_path):
    """
    Preprocess Imagenet dataset
    :return:
    """

    # Load images from dataset
    test_dataset =tf.data.TFRecordDataset(get_filenames(is_training=False, data_dir=imagenet_path+'/Val'))
     
    train_dataset = tf.data.TFRecordDataset(get_filenames(is_training=True, data_dir=imagenet_path+'/Train'))

    train_processed = process_record_dataset(dataset=train_dataset, is_training=True, batch_size=batch_size, shuffle_buffer=_SHUFFLE_BUFFER, num_epochs = nb_epochs, parse_record_fn=parse_record)
    
    test_processed = process_record_dataset(dataset=test_dataset, is_training=False, batch_size=batch_size, shuffle_buffer=_SHUFFLE_BUFFER, parse_record_fn=parse_record)
     
    return train_processed, test_processed 

def main(argv=None):
    model_path = FLAGS.model_path
    targeted = True if FLAGS.targeted else False
    scale = True if FLAGS.scale else False
    learning_rate = FLAGS.learning_rate
    nb_filters = FLAGS.nb_filters
    batch_size = FLAGS.batch_size
    nb_epochs = FLAGS.nb_epochs
    delay = FLAGS.delay
    eps = FLAGS.eps
    adv = FLAGS.adv

    attack = FLAGS.attack
    attack_iterations = FLAGS.attack_iterations
    nb_iter = FLAGS.nb_iter
    
    #### EMPIR extra flags
    lowprecision=FLAGS.lowprecision
    abits=FLAGS.abits
    wbits=FLAGS.wbits
    abitsList=FLAGS.abitsList
    wbitsList=FLAGS.wbitsList
    stocRound=True if FLAGS.stocRound else False
    rand=FLAGS.rand 
    model_path2 = FLAGS.model_path2
    model_path1 = FLAGS.model_path1
    model_path3 = FLAGS.model_path3
    ensembleThree=True if FLAGS.ensembleThree else False
    abits2=FLAGS.abits2
    wbits2=FLAGS.wbits2
    abits2List=FLAGS.abits2List
    wbits2List=FLAGS.wbits2List
    ####
   
    save = False
    train_from_scratch = False

    #### Imagenet flags
    imagenet_path = FLAGS.imagenet_path
    if imagenet_path is None:
        print("Error: Imagenet data path not specified")
        sys.exit(1)

    # Imagenet specific dimensions
    img_rows = _DEFAULT_IMAGE_SIZE
    img_cols = _DEFAULT_IMAGE_SIZE
    channels = _NUM_CHANNELS
    nb_classes = _NUM_CLASSES

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured"
                           " to use the TensorFlow backend.")

    # Image dimensions ordering should follow the Theano convention
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
              "'th', temporarily setting to 'tf'")

    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    set_log_level(logging.WARNING)
    
    # Get imagenet datasets
    train_dataset, test_dataset = data_imagenet(nb_epochs, batch_size, imagenet_path)

    # Creating a initializable iterators
    train_iterator = train_dataset.make_initializable_iterator()
    test_iterator = test_dataset.make_initializable_iterator()

    # Getting next elements from the iterators
    next_test_element = test_iterator.get_next()
    next_train_element = train_iterator.get_next()
    
    train_x, train_y = train_iterator.get_next()
    test_x, test_y = test_iterator.get_next()

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    phase = tf.placeholder(tf.bool, name="phase")

    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")
   
    if ensembleThree: 
        if (model_path1 is None or model_path2 is None or model_path3 is None):
            train_from_scratch = True
        else:
            train_from_scratch = False
    elif model_path is not None:
        if os.path.exists(model_path):
            # check for existing model in immediate subfolder
            if any(f.endswith('.meta') for f in os.listdir(model_path)):
                train_from_scratch = False
            else:
                model_path = build_model_save_path(
                    model_path, batch_size, nb_filters, learning_rate, nb_epochs, adv, delay)
                print(model_path)
                save = True
                train_from_scratch = True
    else:
        train_from_scratch = True  # train from scratch, but don't save since no path given
    
    if ensembleThree: 
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified for first model
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the first model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec1 = 1
       elif (len(wbitsList) != 6) or (len(abitsList) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer of the first model")  
           sys.exit(1)
       else: 
           fixedPrec1 = 0
       
       if (wbits2List is None) or (abits2List is None): # Layer wise separate quantization not specified for second model
           if (wbits2==0) or (abits2==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the second model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec2 = 1
       elif (len(wbits2List) != 6) or (len(abits2List) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer of the second model")  
           sys.exit(1)
       else: 
           fixedPrec2 = 0

       if (fixedPrec2 != 1) or (fixedPrec1 != 1): # Atleast one of the models have separate precisions per layer
           fixedPrec=0
           print("Within atleast one model has separate precisions")
           if (fixedPrec1 == 1): # first layer has fixed precision
               abitsList = (abits, abits, abits, abits, abits, abits)
               wbitsList = (wbits, wbits, wbits, wbits, wbits, wbits)
           if (fixedPrec2 == 1): # second layer has fixed precision
               abits2List = (abits2, abits2, abits2, abits2, abits2, abits2)
               wbits2List = (wbits2, wbits2, wbits2, wbits2, wbits2, wbits2)
       else:
           fixedPrec=1
       
       if (train_from_scratch):
           print ("The ensemble model cannot be trained from scratch")
           sys.exit(1)
       if fixedPrec == 1:
           from modified_cleverhans_tutorials.tutorial_models import make_ensemble_three_alexnet
           model = make_ensemble_three_alexnet(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbits, abits, wbits2, abits2, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) 
       else:
           from modified_cleverhans_tutorials.tutorial_models import make_layerwise_three_combined_alexnet
           model = make_layerwise_three_combined_alexnet(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbitsList, abitsList, wbits2List, abits2List, input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) 
    elif lowprecision:
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers have to specified using wbits and abits flags")
               sys.exit(1)
           else:
               fixedPrec = 1
       elif (len(wbitsList) != 6) or (len(abitsList) != 6):
           print("Error: Need to specify the precisions for activations and weights for the atleast the four convolutional layers of alexnet excluding the first layer and 2 fully connected layers excluding the last layer")  
           sys.exit(1)
       else: 
           fixedPrec = 0
       
       if fixedPrec:
           
           ### For training from scratch
           from modified_cleverhans_tutorials.tutorial_models import make_basic_lowprecision_alexnet
           model = make_basic_lowprecision_alexnet(phase, logits_scalar, 'lp_', wbits, abits, input_shape=(
            None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes)  
       else:
           from modified_cleverhans_tutorials.tutorial_models import make_layerwise_lowprecision_alexnet
           model = make_layerwise_lowprecision_alexnet(phase, logits_scalar, 'lp_', wbitsList, abitsList, 
            input_shape=(None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) 
    else:
        ### For training from scratch
        from modified_cleverhans_tutorials.tutorial_models import make_basic_alexnet_from_scratch
        model = make_basic_alexnet_from_scratch(phase, logits_scalar, 'fp_', input_shape=(
        None, img_rows, img_cols, channels), nb_filters=nb_filters, nb_classes=nb_classes) 

    # separate calling function for ensemble models
    if ensembleThree:
        preds = model.ensemble_call(x, reuse=False)
    else:
    ##default
        preds = model(x, reuse=False)
    print("Defined TensorFlow model graph.")

    rng = np.random.RandomState([2017, 8, 30])

    def evaluate():
        # Evaluate the accuracy of the CIFAR10 model on legitimate test
        # examples
        eval_params = {'batch_size': batch_size}
        if ensembleThree:
            acc = model_eval_ensemble_imagenet(
                sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params)
        else: #default below
            acc = model_eval_imagenet(
                sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an Imagenet model
    train_params = {
        'lowprecision': lowprecision,
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_name': 'train loss',
        'filename': 'model',
        'reuse_global_step': False,
        'train_scope': 'train',
        'is_training': True
    }

    if adv != 0:
        if adv == ADVERSARIAL_TRAINING_MADRYETAL:
            from modified_cleverhans.attacks import MadryEtAl
            train_attack_params = {'eps': MAX_EPS, 'eps_iter': 0.01,
                                   'nb_iter': nb_iter}
            train_attacker = MadryEtAl(model, sess=sess)

        elif adv == ADVERSARIAL_TRAINING_FGSM:
            from modified_cleverhans.attacks import FastGradientMethod
            stddev = int(np.ceil((MAX_EPS * 255) // 2))
            train_attack_params = {'eps': tf.abs(tf.truncated_normal(
                shape=(batch_size, 1, 1, 1), mean=0, stddev=stddev))}
            train_attacker = FastGradientMethod(model, back='tf', sess=sess)
        # create the adversarial trainer
        train_attack_params.update({'clip_min': 0., 'clip_max': 1.})
        adv_x_train = train_attacker.generate(x, phase, **train_attack_params)
        preds_adv_train = model.get_probs(adv_x_train)

        eval_attack_params = {'eps': MAX_EPS, 'clip_min': 0., 'clip_max': 1.}
        adv_x_eval = train_attacker.generate(x, phase, **eval_attack_params)
        preds_adv_eval = model.get_probs(adv_x_eval)  # * logits_scalar
   #  if adv:
   #      from modified_cleverhans.attacks import FastGradientMethod
   #      fgsm = FastGradientMethod(model, back='tf', sess=sess)
   #      fgsm_params = {'eps': eps, 'clip_min': 0., 'clip_max': 1.}
   #      adv_x_train = fgsm.generate(x, phase, **fgsm_params)
   #      preds_adv = model.get_probs(adv_x_train)

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})
        
        # do clean training for 'nb_epochs' or 'delay' epochs with learning rate reducing with time
        model_train_imagenet2(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                evaluate=evaluate, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            model_train_imagenet(sess, x, y, preds, train_iterator, train_x, train_y, phase=phase,
                    predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params, save=save, rng=rng)
    else:
        if ensembleThree: ## ensembleThree models have to loaded from different paths
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            # First 11 variables from path1
            stored_variables = ['lp_conv1_init/k', 'lp_conv1_init/b', 'lp_conv2_init/k', 'lp_conv3_init/k', 'lp_conv4_init/k', 'lp_conv5_init/k', 'lp_ip1init/W', 'lp_ip1init/b', 'lp_ip2init/W', 'lp_logits_init/W', 'lp_logits_init/b']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[:11]))) # only dict was messing with the order
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Restore the second set of variables from model_path2
            # Second 11 variables from path2
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[11:22])))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            # Third 11 variables from path3
            stored_variables = ['fp_conv1_init/k', 'fp_conv1_init/b', 'fp_conv2_init/k', 'fp_conv3_init/k', 'fp_conv4_init/k', 'fp_conv5_init/k', 'fp_ip1init/W', 'fp_ip1init/b', 'fp_ip2init/W', 'fp_logits_init/W', 'fp_logits_init/b']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[22:33])))
            saver3 = tf.train.Saver(variable_dict)
            saver3.restore(sess, tf.train.latest_checkpoint(model_path3))
            # Next 24 batch norm variables from path1
            stored_variables = ['lp__batchNorm1/batch_normalization/gamma', 'lp__batchNorm1/batch_normalization/beta', 'lp__batchNorm1/batch_normalization/moving_mean', 'lp__batchNorm1/batch_normalization/moving_variance', 'lp__batchNorm2/batch_normalization/gamma', 'lp__batchNorm2/batch_normalization/beta', 'lp__batchNorm2/batch_normalization/moving_mean', 'lp__batchNorm2/batch_normalization/moving_variance', 'lp__batchNorm3/batch_normalization/gamma', 'lp__batchNorm3/batch_normalization/beta', 'lp__batchNorm3/batch_normalization/moving_mean', 'lp__batchNorm3/batch_normalization/moving_variance', 'lp__batchNorm4/batch_normalization/gamma', 'lp__batchNorm4/batch_normalization/beta', 'lp__batchNorm4/batch_normalization/moving_mean', 'lp__batchNorm4/batch_normalization/moving_variance', 'lp__batchNorm5/batch_normalization/gamma', 'lp__batchNorm5/batch_normalization/beta', 'lp__batchNorm5/batch_normalization/moving_mean', 'lp__batchNorm5/batch_normalization/moving_variance', 'lp__batchNorm6/batch_normalization/gamma', 'lp__batchNorm6/batch_normalization/beta', 'lp__batchNorm6/batch_normalization/moving_mean', 'lp__batchNorm6/batch_normalization/moving_variance']

            variable_dict = dict(OrderedDict(zip(stored_variables, variables[33:57])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Next 24 batch norm variables from path2
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[57:81])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path2))
            # Final 24 batch norm variables from path1
            stored_variables = ['fp__batchNorm1/batch_normalization/gamma', 'fp__batchNorm1/batch_normalization/beta', 'fp__batchNorm1/batch_normalization/moving_mean', 'fp__batchNorm1/batch_normalization/moving_variance', 'fp__batchNorm2/batch_normalization/gamma', 'fp__batchNorm2/batch_normalization/beta', 'fp__batchNorm2/batch_normalization/moving_mean', 'fp__batchNorm2/batch_normalization/moving_variance', 'fp__batchNorm3/batch_normalization/gamma', 'fp__batchNorm3/batch_normalization/beta', 'fp__batchNorm3/batch_normalization/moving_mean', 'fp__batchNorm3/batch_normalization/moving_variance', 'fp__batchNorm4/batch_normalization/gamma', 'fp__batchNorm4/batch_normalization/beta', 'fp__batchNorm4/batch_normalization/moving_mean', 'fp__batchNorm4/batch_normalization/moving_variance', 'fp__batchNorm5/batch_normalization/gamma', 'fp__batchNorm5/batch_normalization/beta', 'fp__batchNorm5/batch_normalization/moving_mean', 'fp__batchNorm5/batch_normalization/moving_variance', 'fp__batchNorm6/batch_normalization/gamma', 'fp__batchNorm6/batch_normalization/beta', 'fp__batchNorm6/batch_normalization/moving_mean', 'fp__batchNorm6/batch_normalization/moving_variance']
            variable_dict = dict(OrderedDict(zip(stored_variables, variables[81:105])))
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path3))
        else: # restoring the model trained using this setup, not a downloaded one
            tf_model_load(sess, model_path)
            print('Restored model from %s' % model_path)
            # evaluate()


    # Evaluate the accuracy of the model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    if ensembleThree:
        accuracy = model_eval_ensemble_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, feed={phase: False}, args=eval_params)
    else: #default below
        accuracy = model_eval_imagenet(sess, x, y, preds, test_iterator, test_x, test_y, phase=phase, feed={phase: False}, args=eval_params)
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))

    ###########################################################################
    # Build dataset
    ###########################################################################

    adv_inputs = test_x #adversarial inputs can be generated from any of the test examples 

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    nb_adv_per_sample = 1
    adv_ys = None
    yname = "y"

    print('Crafting adversarial examples')
    print("This could take some time ...")

    if ensembleThree:
        model_type = 'ensembleThree'
    else:
        model_type = 'default'

    if attack == ATTACK_CARLINI_WAGNER_L2:
        from modified_cleverhans.attacks import CarliniWagnerL2
        attacker = CarliniWagnerL2(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.1,
                         'batch_size': batch_size,
                         'initial_const': 10,
                         }
    elif attack == ATTACK_JSMA:
        from modified_cleverhans.attacks import SaliencyMapMethod
        attacker = SaliencyMapMethod(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'theta': 1., 'gamma': 0.1}
    elif attack == ATTACK_FGSM:
        from modified_cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'eps': eps}
    elif attack == ATTACK_MADRYETAL:
        from modified_cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    elif attack == ATTACK_BASICITER:
        print('Attack: BasicIterativeMethod')
        from modified_cleverhans.attacks import BasicIterativeMethod
        attacker = BasicIterativeMethod(model, back='tf', sess=sess, model_type=model_type, num_classes=nb_classes)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    else:
        print("Attack undefined")
        sys.exit(1)

    attack_params.update({'clip_min': -2.2, 'clip_max': 2.7}) # Since max and min for imagenet turns out to be around -2.11 and 2.12
    eval_params = {'batch_size': batch_size}
    '''
    adv_x = attacker.generate(x, phase, **attack_params)
    # Craft adversarial examples using Fast Gradient Sign Method (FGSM)
    eval_params = {'batch_size': batch_size}
    X_test_adv, = batch_eval(sess, [x], [adv_x], [adv_inputs], feed={
                             phase: False}, args=eval_params)
    '''

    print("Evaluating un-targeted results")
    if ensembleThree:
        adv_accuracy = model_eval_ensemble_adv_imagenet(sess, x, y, preds, test_iterator, 
                        test_x, test_y, phase=phase, args=eval_params, attacker=attacker, attack_params=attack_params)
    else:
        adv_accuracy = model_eval_adv_imagenet(sess, x, y, preds, test_iterator, 
                        test_x, test_y, phase=phase, args=eval_params, attacker=attacker, attack_params=attack_params)
    
    # Compute the number of adversarial examples that were successfully found
    print('Test accuracy on adversarial examples {0:.4f}'.format(adv_accuracy))


    # Close TF session
    sess.close()


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
    par.add_argument('--data_dir', help='Path to training data',
                     default='/scratch/gallowaa/cifar10/cifar10_data')

    # Architecture and training specific flags
    par.add_argument('--nb_epochs', type=int, default=6,
                     help='Number of epochs to train model')
    par.add_argument('--nb_filters', type=int, default=32,
                     help='Number of filters in first layer')
    par.add_argument('--batch_size', type=int, default=100,
                     help='Size of training batches')
    par.add_argument('--learning_rate', type=float, default=0.001,
                     help='Learning rate')
    par.add_argument('--scale', help='Scale activations of the binary model?',
                     action="store_true")
    par.add_argument('--rand', help='Stochastic weight layer?',
                     action="store_true")
    # EMPIR specific flags
    par.add_argument('--lowprecision', help='Use other low precision models', action="store_true")
    par.add_argument('--wbits', type=int, default=0, help='No. of bits in weight representation')
    par.add_argument('--abits', type=int, default=0, help='No. of bits in activation representation')
    par.add_argument('--wbitsList', type=int, nargs='+', help='List of No. of bits in weight representation for different layers')
    par.add_argument('--abitsList', type=int, nargs='+', help='List of No. of bits in activation representation for different layers')
    par.add_argument('--stocRound', help='Stochastic rounding for weights (only in training) and activations?', action="store_true")
    par.add_argument('--model_path1', help='Path where saved model1 is stored and can be loaded')
    par.add_argument('--model_path2', help='Path where saved model2 is stored and can be loaded')
    par.add_argument('--ensembleThree', help='Use an ensemble of full precision and two low precision models that can be attacked directly and potentially trained', action="store_true") 
    par.add_argument('--model_path3', help='Path where saved model3 in case of combinedThree model is stored and can be loaded')
    par.add_argument('--wbits2', type=int, default=0, help='No. of bits in weight representation of model2, model1 specified using wbits')
    par.add_argument('--abits2', type=int, default=0, help='No. of bits in activation representation of model2, model2 specified using abits')
    par.add_argument('--wbits2List', type=int, nargs='+', help='List of No. of bits in weight representation for different layers of model2')
    par.add_argument('--abits2List', type=int, nargs='+', help='List of No. of bits in activation representation for different layers of model2')

    # Attack specific flags
    par.add_argument('--eps', type=float, default=0.1,
                     help='epsilon')
    par.add_argument('--attack', type=int, default=0,
                     help='Attack type, 0=CW, 2=FGSM')
    par.add_argument('--attack_iterations', type=int, default=50,
                     help='Number of iterations to run CW attack; 1000 is good')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    # Adversarial training flags 
    par.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=40, help='Nb of iterations of PGD')

    # imagenet flags
    par.add_argument('--imagenet_path', help='Path where imagenet tfrecords are stored and can be loaded, both Val and Train')

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
