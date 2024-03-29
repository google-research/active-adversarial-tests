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

import os
import sys
import numpy as np
from collections import OrderedDict
from six.moves import xrange
import warnings
import logging

known_number_types = (int, float, np.float16, np.float32, np.float64,
                      np.int8, np.int16, np.int32, np.int32, np.int64,
                      np.uint8, np.uint16, np.uint32, np.uint64)


class _ArgsWrapper(object):

    """
    Wrapper that allows attribute access to dictionaries
    """

    def __init__(self, args):
        if not isinstance(args, dict):
            args = vars(args)
        self.args = args

    def __getattr__(self, name):
        return self.args.get(name)


class AccuracyReport(object):

    """
    An object summarizing the accuracy results for experiments involving
    training on clean examples or adversarial examples, then evaluating
    on clean or adversarial examples.
    """

    def __init__(self):
        self.clean_train_clean_eval = 0.
        self.clean_train_adv_eval = 0.
        self.adv_train_clean_eval = 0.
        self.adv_train_adv_eval = 0.

        # Training data accuracy results to be used by tutorials
        self.train_clean_train_clean_eval = 0.
        self.train_clean_train_adv_eval = 0.
        self.train_adv_train_clean_eval = 0.
        self.train_adv_train_adv_eval = 0.


def batch_indices(batch_nb, data_length, batch_size):
    """
    This helper function computes a batch start and end index
    :param batch_nb: the batch number
    :param data_length: the total length of the data being parsed by batches
    :param batch_size: the number of inputs in each batch
    :return: pair of (start, end) indices
    """
    # Batch start and end index
    start = int(batch_nb * batch_size)
    end = int((batch_nb + 1) * batch_size)

    # When there are not enough inputs left, we reuse some to complete the
    # batch
    if end > data_length:
        shift = end - data_length
        start -= shift
        end -= shift

    return start, end


def other_classes(nb_classes, class_ind):
    """
    Returns a list of class indices excluding the class indexed by class_ind
    :param nb_classes: number of classes in the task
    :param class_ind: the class index to be omitted
    :return: list of class indices excluding the class indexed by class_ind
    """
    if class_ind < 0 or class_ind >= nb_classes:
        error_str = "class_ind must be within the range (0, nb_classes - 1)"
        raise ValueError(error_str)

    other_classes_list = list(range(nb_classes))
    other_classes_list.remove(class_ind)

    return other_classes_list


def to_categorical(y, num_classes=None):
    """
    Converts a class vector (integers) to binary class matrix.
    This is adapted from the Keras function with the same name.
    :param y: class vector to be converted into a matrix
              (integers from 0 to num_classes).
    :param num_classes: num_classes: total number of classes.
    :return: A binary matrix representation of the input.
    """
    y = np.array(y, dtype='int').ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    return categorical


def random_targets(gt, nb_classes):
    """
    Take in an array of correct labels and randomly select a different label
    for each label in the array. This is typically used to randomly select a
    target class in targeted adversarial examples attacks (i.e., when the
    search algorithm takes in both a source class and target class to compute
    the adversarial example).
    :param gt: the ground truth (correct) labels. They can be provided as a
               1D vector or 2D array of one-hot encoded labels.
    :param nb_classes: The number of classes for this task. The random class
                       will be chosen between 0 and nb_classes such that it
                       is different from the correct class.
    :return: A numpy array holding the randomly-selected target classes
             encoded as one-hot labels.
    """
    # If the ground truth labels are encoded as one-hot, convert to labels.
    if len(gt.shape) == 2:
        gt = np.argmax(gt, axis=1)

    # This vector will hold the randomly selected labels.
    result = np.zeros(gt.shape, dtype=np.int32)

    for class_ind in xrange(nb_classes):
        # Compute all indices in that class.
        in_cl = gt == class_ind
        size = np.sum(in_cl)

        # Compute the set of potential targets for this class.
        potential_targets = other_classes(nb_classes, class_ind)

        # Draw with replacement random targets among the potential targets.
        result[in_cl] = np.random.choice(potential_targets, size=size)

    # Encode vector of random labels as one-hot labels.
    result = to_categorical(result, nb_classes)
    result = result.astype(np.int32)

    return result


def pair_visual(original, adversarial, figure=None):
    """
    This function displays two images: the original and the adversarial sample
    :param original: the original input
    :param adversarial: the input after perterbations have been applied
    :param figure: if we've already displayed images, use the same plot
    :return: the matplot figure to reuse for future samples
    """
    import matplotlib.pyplot as plt

    # Ensure our inputs are of proper shape
    assert(len(original.shape) == 2 or len(original.shape) == 3)

    # To avoid creating figures per input sample, reuse the sample plot
    if figure is None:
        plt.ion()
        figure = plt.figure()
        figure.canvas.set_window_title('Cleverhans: Pair Visualization')

    # Add the images to the plot
    perterbations = adversarial - original
    for index, image in enumerate((original, perterbations, adversarial)):
        figure.add_subplot(1, 3, index + 1)
        plt.axis('off')

        # If the image is 2D, then we have 1 color channel
        if len(image.shape) == 2:
            plt.imshow(image, cmap='gray')
        else:
            plt.imshow(image)

        # Give the plot some time to update
        plt.pause(0.01)

    # Draw the plot and return
    plt.show()
    return figure


def grid_visual(data):
    """
    This function displays a grid of images to show full misclassification
    :param data: grid data of the form;
        [nb_classes : nb_classes : img_rows : img_cols : nb_channels]
    :return: if necessary, the matplot figure to reuse
    """
    import matplotlib.pyplot as plt

    # Ensure interactive mode is disabled and initialize our graph
    plt.ioff()
    figure = plt.figure()
    figure.canvas.set_window_title('Cleverhans: Grid Visualization')

    # Add the images to the plot
    num_cols = data.shape[0]
    num_rows = data.shape[1]
    num_channels = data.shape[4]
    current_row = 0
    for y in xrange(num_rows):
        for x in xrange(num_cols):
            figure.add_subplot(num_rows, num_cols, (x + 1) + (y * num_cols))
            plt.axis('off')

            if num_channels == 1:
                plt.imshow(data[x, y, :, :, 0], cmap='gray')
            else:
                plt.imshow(data[x, y, :, :, :])

    # Draw the plot and return
    plt.show()
    return figure


def conv_2d(*args, **kwargs):
    from modified_cleverhans.utils_keras import conv_2d
    warnings.warn("utils.conv_2d is deprecated and may be removed on or after"
                  " 2018-01-05. Switch to utils_keras.conv_2d.")
    return conv_2d(*args, **kwargs)


def cnn_model(*args, **kwargs):
    from modified_cleverhans.utils_keras import cnn_model
    warnings.warn("utils.cnn_model is deprecated and may be removed on or"
                  " after 2018-01-05. Switch to utils_keras.cnn_model.")
    return cnn_model(*args, **kwargs)


def set_log_level(level, name="cleverhans"):
    """
    Sets the threshold for the cleverhans logger to level
    :param level: the logger threshold. You can find values here:
                  https://docs.python.org/2/library/logging.html#levels
    :param name: the name used for the cleverhans logger
    """
    logging.getLogger(name).setLevel(level)


def create_logger(name):
    """
    Create a logger object with the given name.

    If this is the first time that we call this method, then initialize the
    formatter.
    """
    base = logging.getLogger("cleverhans")
    if len(base.handlers) == 0:
        ch = logging.StreamHandler()
        formatter = logging.Formatter('[%(levelname)s %(asctime)s %(name)s] ' +
                                      '%(message)s')
        ch.setFormatter(formatter)
        base.addHandler(ch)

    return base


def deterministic_dict(normal_dict):
    """
    Returns a version of `normal_dict` whose iteration order is always the same
    """
    out = OrderedDict()
    for key in sorted(normal_dict.keys()):
        out[key] = normal_dict[key]
    return out


def parse_model_settings(model_path):

    tokens = model_path.split('/')
    precision_list = ['bin', 'binsc', 'fp']
    precision = ''
    start_index = 0
    adv = False

    for p in precision_list:
        if p in tokens:
            start_index = tokens.index(p)
            precision = p
    try:
        nb_filters = int(tokens[start_index + 1].split('_')[1])
        batch_size = int(tokens[start_index + 2].split('_')[1])
        learning_rate = float(tokens[start_index + 3].split('_')[1])
        nb_epochs = int(tokens[start_index + 4].split('_')[1])

        adv_index = start_index + 5
        if adv_index < len(tokens):
            adv = True if 'adv' in tokens[adv_index] else False

        print("Got %s model" % precision)
        print("Got %d filters" % nb_filters)
        print("Got batch_size %d" % batch_size)
        print("Got batch_size %f" % learning_rate)
        print("Got %d epochs" % nb_epochs)
    except:
        print("Could not parse tokens!")
        sys.exit(1)

    return nb_filters, batch_size, learning_rate, nb_epochs, adv


def build_model_save_path(root_path, batch_size, nb_filters, lr, epochs, adv, delay):

    model_path = os.path.join(root_path, precision)
    model_path += 'k_' + str(nb_filters) + '/'
    model_path += 'bs_' + str(batch_size) + '/'
    model_path += 'lr_' + str(lr) + '/'
    model_path += 'ep_' + str(epochs)

    if adv:
        model_path += '/adv_%d' % delay

    # optionally create this dir if it does not already exist,
    # otherwise, increment
    model_path = create_dir_if_not_exists(model_path)

    return model_path


def create_dir_if_not_exists(path):
    if not os.path.exists(path):
        path += '/1'
        os.makedirs(path)
    else:
        digits = []
        sub_dirs = next(os.walk(path))[1]
        [digits.append(s) for s in sub_dirs if s.isdigit()]
        sub = '/' + str(int(max(digits)) + 1) if len(digits) > 0 else '/1'
        path += sub
        os.makedirs(path)
    print('Logging to:%s' % path)
    return path


def build_targeted_dataset(X_test, Y_test, indices, nb_classes, img_rows, img_cols, img_channels):
    """
    Build a dataset for targeted attacks, each source image is repeated nb_classes -1
    times, and target labels are assigned that do not overlap with true label. 
    :param X_test: clean source images
    :param Y_test: true labels for X_test
    :param indices: indices of source samples to use
    :param nb_classes: number of classes in classification problem
    :param img_rows: number of pixels along rows of image
    :param img_cols: number of pixels along columns of image
    """

    nb_samples = len(indices)
    nb_target_classes = nb_classes - 1
    X = X_test[indices]
    Y = Y_test[indices]

    adv_inputs = np.array(
        [[instance] * nb_target_classes for
         instance in X], dtype=np.float32)
    adv_inputs = adv_inputs.reshape(
        (nb_samples * nb_target_classes, img_rows, img_cols, img_channels))

    true_labels = np.array(
        [[instance] * nb_target_classes for
         instance in Y], dtype=np.float32)
    true_labels = true_labels.reshape(
        nb_samples * nb_target_classes, nb_classes)

    target_labels = np.zeros((nb_samples * nb_target_classes, nb_classes))

    for n in range(nb_samples):
        one_hot = np.zeros((nb_target_classes, nb_classes))
        one_hot[np.arange(nb_target_classes), np.arange(nb_classes)
                != np.argmax(Y[n])] = 1.0
        start = n * nb_target_classes
        end = start + nb_target_classes
        target_labels[start:end] = one_hot

    return adv_inputs, true_labels, target_labels
