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

import numpy as np
from six.moves import xrange
import tensorflow as tf
from tensorflow.python.platform import flags

import time
import argparse
import logging
import os
import sys

from modified_cleverhans.utils import parse_model_settings, build_model_save_path
from modified_cleverhans.utils import set_log_level, AccuracyReport
from modified_cleverhans.utils_mnist import data_mnist
from modified_cleverhans.utils_tf import model_train, model_eval, model_eval_ensemble, batch_eval, tf_model_load
from modified_cleverhans.utils_tf import model_train_teacher, model_train_student, model_train_inpgrad_reg #for training with input gradient regularization


FLAGS = flags.FLAGS

# Scaling input to softmax
INIT_T = 1.0
#ATTACK_T = 1.0
ATTACK_T = 0.25

# enum attack types
ATTACK_CARLINI_WAGNER_L2 = 0
ATTACK_JSMA = 1
ATTACK_FGSM = 2
ATTACK_MADRYETAL = 3
ATTACK_BASICITER = 4

# enum adversarial training types
ADVERSARIAL_TRAINING_MADRYETAL = 1
ADVERSARIAL_TRAINING_FGSM = 2
MAX_EPS = 0.3

MAX_BATCH_SIZE = 100


def mnist_attack(train_start=0, train_end=60000, test_start=0,
                 test_end=10000, viz_enabled=True, nb_epochs=6,
                 batch_size=128, nb_filters=64,
                 nb_samples=10, learning_rate=0.001,
                 eps=0.3, attack=0,
                 attack_iterations=100, model_path=None,
                 targeted=False, rand=False,
                 stocRound=False, lowprecision=False,  
                 wbits=0, abits=0, wbitsList=0, abitsList=0, wbits2=0, abits2=0, wbits2List=0, abits2List=0, 
                 ensembleThree=False, model_path1=None, model_path2=None, model_path3=None,
                 distill = False, inpgradreg = False, l2dbl = 0, l2cs = 0, 
                 debug=None, test=False,
                 data_dir=None, delay=0, adv=0, nb_iter=40):
    """
    MNIST tutorial for generic attack
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param viz_enabled: (boolean) activate plots of adversarial examples
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param nb_classes: number of output classes
    :param nb_samples: number of test inputs to attack
    :param learning_rate: learning rate for training
    :param model_path: path to the model file
    :param targeted: should we run a targeted attack? or untargeted?
    :return: an AccuracyReport object
    """
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # MNIST-specific dimensions
    img_rows = 28
    img_cols = 28
    channels = 1
    nb_classes = 10

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1237)

    # Create TF session
    sess = tf.Session()
    print("Created TensorFlow session.")

    if debug:
        set_log_level(logging.DEBUG)
    else:
        set_log_level(logging.WARNING)  # for running on sharcnet

    # Get MNIST test data
    X_train, Y_train, X_test, Y_test = data_mnist(datadir=data_dir,
                                                  train_start=train_start,
                                                  train_end=train_end,
                                                  test_start=test_start,
                                                  test_end=test_end)

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, channels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    phase = tf.placeholder(tf.bool, name='phase')

    # for attempting to break unscaled network.
    logits_scalar = tf.placeholder_with_default(
        INIT_T, shape=(), name="logits_temperature")

    save = False
    train_from_scratch = False
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

    # Define TF model graph
    if ensembleThree: 
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified for first model
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the first model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec1 = 1
       elif (len(wbitsList) != 3) or (len(abitsList) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers of the first model")  
           sys.exit(1)
       else: 
           fixedPrec1 = 0
       
       if (wbits2List is None) or (abits2List is None): # Layer wise separate quantization not specified for second model
           if (wbits2==0) or (abits2==0):
               print("Error: the number of bits for constant precision weights and activations across layers for the second model have to specified using wbits1 and abits1 flags")
               sys.exit(1)
           else:
               fixedPrec2 = 1
       elif (len(wbits2List) != 3) or (len(abits2List) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers of the second model")  
           sys.exit(1)
       else: 
           fixedPrec2 = 0

       if (fixedPrec2 != 1) or (fixedPrec1 != 1): # Atleast one of the models have separate precisions per layer
           fixedPrec=0
           print("Within atleast one model has separate precisions")
           if (fixedPrec1 == 1): # first layer has fixed precision
               abitsList = (abits, abits, abits)
               wbitsList = (wbits, wbits, wbits)
           if (fixedPrec2 == 1): # second layer has fixed precision
               abits2List = (abits2, abits2, abits2)
               wbits2List = (wbits2, wbits2, wbits2)
       else:
           fixedPrec=1

       if (train_from_scratch):
           print ("The ensemble model cannot be trained from scratch")
           sys.exit(1)
       if fixedPrec == 1:
           from modified_cleverhans_tutorials.tutorial_models import make_ensemble_three_cnn
           model = make_ensemble_three_cnn(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbits, abits, wbits2, abits2, nb_filters=nb_filters) 
       else:
           from modified_cleverhans_tutorials.tutorial_models import make_layerwise_three_combined_cnn
           model = make_layerwise_three_combined_cnn(
               phase, logits_scalar, 'lp1_', 'lp2_', 'fp_', wbitsList, abitsList, wbits2List, abits2List, nb_filters=nb_filters) 
    elif lowprecision: # For generic DoReFa net style low precision
       if (wbitsList is None) or (abitsList is None): # Layer wise separate quantization not specified
           if (wbits==0) or (abits==0):
               print("Error: the number of bits for constant precision weights and activations across layers have to specified using wbits and abits flags")
               sys.exit(1)
           else:
               fixedPrec = 1
       elif (len(wbitsList) != 3) or (len(abitsList) != 3):
           print("Error: Need to specify the precisions for activations and weights for the atleast the three convolutional layers")  
           sys.exit(1)
       else: 
           fixedPrec = 0
       
       if fixedPrec:
           from modified_cleverhans_tutorials.tutorial_models import make_basic_lowprecision_cnn
           model = make_basic_lowprecision_cnn(
               phase, logits_scalar, 'lp_', wbits, abits, nb_filters=nb_filters, stocRound=stocRound) 
       else:
           from modified_cleverhans_tutorials.tutorial_models import make_layerwise_lowprecision_cnn
           model = make_layerwise_lowprecision_cnn(
               phase, logits_scalar, 'lp_', wbitsList, abitsList, nb_filters=nb_filters, stocRound=stocRound) 
    elif distill:
      from modified_cleverhans_tutorials.tutorial_models import make_distilled_cnn
      model = make_distilled_cnn(phase, logits_scalar,
              'teacher_fp_', 'fp_', nb_filters=nb_filters)  
    else:
        if rand:
            print('rand=True')
            from modified_cleverhans_tutorials.tutorial_models import make_scaled_rand_cnn
            model = make_scaled_rand_cnn(
                phase, logits_scalar, 'fp_rand', nb_filters=nb_filters)
        else:
            from modified_cleverhans_tutorials.tutorial_models import make_basic_cnn
            model = make_basic_cnn(phase, logits_scalar,
                                   'fp_', nb_filters=nb_filters)

    # separate predictions of teacher for distilled training
    if distill:
        teacher_preds = model.teacher_call(x, reuse=False)
        teacher_logits = model.get_teacher_logits(x, reuse=False)
    # separate calling function for ensemble models
    if ensembleThree:
        preds = model.ensemble_call(x, reuse=False)
    else:
    ##default
        preds = model(x, reuse=False)  # * logits_scalar
    print("Defined TensorFlow model graph.")

    ###########################################################################
    # Training the model using TensorFlow
    ###########################################################################
    rng = np.random.RandomState([2017, 8, 30])

    # Train an MNIST model
    train_params = {
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

    def evaluate():
        # Evaluate the accuracy of the MNIST model on clean test examples
        eval_params = {'batch_size': batch_size}
        if ensembleThree:
            acc = model_eval_ensemble(
                sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        else:
            acc = model_eval(
                sess, x, y, preds, X_test, Y_test, phase=phase, args=eval_params)
        report.clean_train_clean_eval = acc
        assert X_test.shape[0] == test_end - test_start, X_test.shape
        print('Test accuracy on legitimate examples: %0.4f' % acc)

        if adv != 0:
            # Accuracy of the adversarially trained model on adversarial
            # examples
            acc = model_eval(
                sess, x, y, preds_adv_eval, X_test, Y_test, phase=phase, args=eval_params)
            print('Test accuracy on adversarial examples: %0.4f' % acc)

            acc = model_eval(
                sess, x, y, preds_adv_eval, X_test, Y_test,
                phase=phase, args=eval_params, feed={logits_scalar: ATTACK_T})
            print('Test accuracy on adversarial examples (scaled): %0.4f' % acc)

    if train_from_scratch:
        if save:
            train_params.update({'log_dir': model_path})
            if adv and delay > 0:
                train_params.update({'nb_epochs': delay})

        # do clean training for 'nb_epochs' or 'delay' epochs
        if distill:
            temperature = 100 # 1 means the teacher predictions are used as it is
            teacher_scaled_preds_val = model_train_teacher(sess, x, y, teacher_preds, teacher_logits, 
                        temperature, X_train, Y_train, phase=phase, args=train_params, rng=rng)
            eval_params = {'batch_size': batch_size}
            teacher_acc = model_eval(
                sess, x, y, teacher_preds, X_test, Y_test, phase=phase, args=eval_params)
            print('Test accuracy of the teacher model on legitimate examples: %0.4f' % teacher_acc)
            print('Training the student model...')
            student_train_params = {
                'nb_epochs': 50,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'loss_name': 'train loss',
                'filename': 'model',
                'reuse_global_step': False,
                'train_scope': 'train',
                'is_training': True
            }
            if save:
                student_train_params.update({'log_dir': model_path})
            y_teacher = tf.placeholder(tf.float32, shape=(None, nb_classes))
            model_train_student(sess, x, y, preds, temperature, X_train, Y_train, y_teacher=y_teacher, 
                        teacher_preds=teacher_scaled_preds_val, alpha=0.5, beta=0.5, phase=phase, evaluate=evaluate, args=student_train_params, save=save, rng=rng)
        elif inpgradreg: 
            model_train_inpgrad_reg(sess, x, y, preds, X_train, Y_train, phase=phase,
                        evaluate=evaluate, l2dbl = l2dbl, l2cs = l2cs, args=train_params, save=save, rng=rng)
        elif test:
            model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                        evaluate=evaluate, args=train_params, save=save, rng=rng)
        else:
            model_train(sess, x, y, preds, X_train, Y_train,
                        phase=phase, args=train_params, save=save, rng=rng)

        # optionally do additional adversarial training
        if adv:
            print("Adversarial training for %d epochs" % (nb_epochs - delay))
            train_params.update({'nb_epochs': nb_epochs - delay})
            train_params.update({'reuse_global_step': True})
            if test:
                model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                            predictions_adv=preds_adv_train, evaluate=evaluate, args=train_params,
                            save=save, rng=rng)
            else:
                model_train(sess, x, y, preds, X_train, Y_train, phase=phase,
                            predictions_adv=preds_adv_train, args=train_params,
                            save=save, rng=rng)
    else:
        if ensembleThree: ## Ensemble models have to loaded from different paths
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
            stored_variables = ['lp_conv1_init/k', 'lp_conv2_bin_init/k', 'lp_conv3_bin_init/k', 'lp_logits_init/W']
            variable_dict = dict(zip(stored_variables, variables[:4])) 
            # Restore the first set of variables from model_path1
            saver = tf.train.Saver(variable_dict)
            saver.restore(sess, tf.train.latest_checkpoint(model_path1))
            # Restore the second set of variables from model_path2
            variable_dict = dict(zip(stored_variables, variables[4:8]))
            saver2 = tf.train.Saver(variable_dict)
            saver2.restore(sess, tf.train.latest_checkpoint(model_path2))
            stored_variables = ['fp_conv1_init/k', 'fp_conv2_init/k', 'fp_conv3_init/k', 'fp_logits_init/W']
            variable_dict = dict(zip(stored_variables, variables[8:]))
            saver3 = tf.train.Saver(variable_dict)
            saver3.restore(sess, tf.train.latest_checkpoint(model_path3))
        else: #default below
            tf_model_load(sess, model_path)
            print('Restored model from %s' % model_path)
        evaluate()

    # Evaluate the accuracy of the MNIST model on legitimate test examples
    eval_params = {'batch_size': batch_size}
    if ensembleThree: ## Ensemble models have to be evaluated with a separate function
        accuracy = model_eval_ensemble(sess, x, y, preds, X_test, Y_test, phase=phase, feed={phase: False}, args=eval_params)
    else: #default below
        accuracy = model_eval(sess, x, y, preds, X_test, Y_test, phase=phase,
                          feed={phase: False}, args=eval_params)
    assert X_test.shape[0] == test_end - test_start, X_test.shape
    print('Test accuracy on legitimate test examples: {0}'.format(accuracy))
    report.clean_train_clean_eval = accuracy

    ###########################################################################
    # Build dataset
    ###########################################################################
    if viz_enabled:
        assert nb_samples == nb_classes
        idxs = [np.where(np.argmax(Y_test, axis=1) == i)[0][0]
                for i in range(nb_classes)]
        viz_rows = nb_classes if targeted else 2
        # Initialize our array for grid visualization
        grid_shape = (nb_classes, viz_rows, img_rows, img_cols, channels)
        grid_viz_data = np.zeros(grid_shape, dtype='f')

    if targeted:
        from modified_cleverhans.utils import build_targeted_dataset
        if viz_enabled:
            from modified_cleverhans.utils import grid_visual
            adv_inputs, true_labels, adv_ys = build_targeted_dataset(
                X_test, Y_test, idxs, nb_classes, img_rows, img_cols, channels)
        else:
            adv_inputs, true_labels, adv_ys = build_targeted_dataset(
                X_test, Y_test, np.arange(nb_samples), nb_classes, img_rows, img_cols, channels)
    else:
        if viz_enabled:
            from modified_cleverhans.utils import pair_visual
            adv_inputs = X_test[idxs]
        else:
            adv_inputs = X_test[:nb_samples]

    ###########################################################################
    # Craft adversarial examples using generic approach
    ###########################################################################
    if targeted:
        att_batch_size = np.clip(
            nb_samples * (nb_classes - 1), a_max=MAX_BATCH_SIZE, a_min=1)
        nb_adv_per_sample = nb_classes - 1
        yname = "y_target"

    else:
        att_batch_size = np.minimum(nb_samples, MAX_BATCH_SIZE)
        nb_adv_per_sample = 1
        adv_ys = None
        yname = "y"

    print('Crafting ' + str(nb_samples) + ' * ' + str(nb_adv_per_sample) +
          ' adversarial examples')
    print("This could take some time ...")

    if ensembleThree:
        model_type = 'ensembleThree'
    else:
        model_type = 'default'

    if attack == ATTACK_CARLINI_WAGNER_L2:
        print('Attack: CarliniWagnerL2')
        from modified_cleverhans.attacks import CarliniWagnerL2
        attacker = CarliniWagnerL2(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'binary_search_steps': 1,
                         'max_iterations': attack_iterations,
                         'learning_rate': 0.1,
                         'batch_size': att_batch_size,
                         'initial_const': 10,
                         }
    elif attack == ATTACK_JSMA:
        print('Attack: SaliencyMapMethod')
        from modified_cleverhans.attacks import SaliencyMapMethod
        attacker = SaliencyMapMethod(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'theta': 1., 'gamma': 0.1}
    elif attack == ATTACK_FGSM:
        print('Attack: FastGradientMethod')
        from modified_cleverhans.attacks import FastGradientMethod
        attacker = FastGradientMethod(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'eps': eps}
    elif attack == ATTACK_MADRYETAL:
        print('Attack: MadryEtAl')
        from modified_cleverhans.attacks import MadryEtAl
        attacker = MadryEtAl(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    elif attack == ATTACK_BASICITER:
        print('Attack: BasicIterativeMethod')
        from modified_cleverhans.attacks import BasicIterativeMethod
        attacker = BasicIterativeMethod(model, back='tf', model_type=model_type, num_classes=nb_classes, sess=sess)
        attack_params = {'eps': eps, 'eps_iter': 0.01, 'nb_iter': nb_iter}
    else:
        print("Attack undefined")
        sys.exit(1)

    attack_params.update({yname: adv_ys, 'clip_min': 0., 'clip_max': 1.})
    adv_np = attacker.generate_np(adv_inputs, phase, **attack_params)

    '''
    name = 'm_fgsm_eps%s_n%s.npy' % (eps, nb_samples)
    fpath = os.path.join(
        '/scratch/gallowaa/mnist/adversarial_examples/modified_cleverhans/', name)
    np.savez(fpath, x=adv_np, y=Y_test[:nb_samples])
    '''
    '''
    adv_x = attacker.generate(x, phase, **attack_params)
    adv_np, = batch_eval(sess, [x], [adv_x], [adv_inputs], feed={
                         phase: False}, args=eval_params)
    '''
    eval_params = {'batch_size': att_batch_size}
    if targeted:
        print("Evaluating targeted results")
        adv_accuracy = model_eval(sess, x, y, preds, adv_np, true_labels, phase=phase,
                                  args=eval_params)

    else:
        print("Evaluating untargeted results")
        if viz_enabled:
            if ensembleThree:
                adv_accuracy = model_eval_ensemble(sess, x, y, preds, adv_np, Y_test[idxs], phase=phase, args=eval_params)
            else: #default below
                adv_accuracy = model_eval(sess, x, y, preds, adv_np, Y_test[
                    idxs], phase=phase, args=eval_params)
        else:
            if ensembleThree:
                adv_accuracy = model_eval_ensemble(sess, x, y, preds, adv_np, Y_test[:nb_samples], phase=phase, args=eval_params)
            else: #default below
                adv_accuracy = model_eval(sess, x, y, preds, adv_np, Y_test[
                    :nb_samples], phase=phase, args=eval_params)

    if viz_enabled:
        n = nb_classes - 1
        for i in range(nb_classes):
            if targeted:
                for j in range(nb_classes):
                    if i != j:
                        if j != 0 and i != n:
                            grid_viz_data[i, j] = adv_np[j * n + i]
                        if j == 0 and i > 0 or i == n and j > 0:
                            grid_viz_data[i, j] = adv_np[j * n + i - 1]
                    else:
                        grid_viz_data[i, j] = adv_inputs[j * n]
            else:
                grid_viz_data[j, 0] = adv_inputs[j]
                grid_viz_data[j, 1] = adv_np[j]
        print(grid_viz_data.shape)

    print('--------------------------------------')

    # Compute the number of adversarial examples that were successfully found
    print('Test accuracy on adversarial examples {0:.4f}'.format(adv_accuracy))
    report.clean_train_adv_eval = 1. - adv_accuracy

    # Compute the average distortion introduced by the algorithm
    percent_perturbed = np.mean(np.sum((adv_np - adv_inputs)**2,
                                       axis=(1, 2, 3))**.5)
    print('Avg. L_2 norm of perturbations {0:.4f}'.format(percent_perturbed))

    # Compute number of modified features (L_0 norm)
    nb_changed = np.where(adv_np != adv_inputs)[0].shape[0]
    percent_perturb = np.mean(float(nb_changed) / adv_np.reshape(-1).shape[0])

    # Compute the average distortion introduced by the algorithm
    print('Avg. rate of perturbed features {0:.4f}'.format(percent_perturb))

    # Friendly output for pasting into spreadsheet
    print('{0:.4f}'.format(accuracy))
    print('{0:.4f}'.format(adv_accuracy))
    print('{0:.4f}'.format(percent_perturbed))
    print('{0:.4f}'.format(percent_perturb))

    # Close TF session
    sess.close()

    # Finally, block & display a grid of all the adversarial examples
    if viz_enabled:
        import matplotlib.pyplot as plt
        _ = grid_visual(grid_viz_data)

    return report


def main(argv=None):
    mnist_attack(viz_enabled=FLAGS.viz_enabled,
                 nb_epochs=FLAGS.nb_epochs,
                 batch_size=FLAGS.batch_size,
                 nb_samples=FLAGS.nb_samples,
                 nb_filters=FLAGS.nb_filters,
                 learning_rate=FLAGS.lr,
                 eps=FLAGS.eps,
                 attack=FLAGS.attack,
                 attack_iterations=FLAGS.attack_iterations,
                 model_path=FLAGS.model_path,
                 targeted=FLAGS.targeted,
                 rand=FLAGS.rand,
                 debug=FLAGS.debug,
                 test=FLAGS.test,
                 data_dir=FLAGS.data_dir,
                 lowprecision=FLAGS.lowprecision,
                 abits=FLAGS.abits,
                 wbits=FLAGS.wbits,
                 abitsList=FLAGS.abitsList,
                 wbitsList=FLAGS.wbitsList,
                 abits2=FLAGS.abits2,
                 wbits2=FLAGS.wbits2,
                 abits2List=FLAGS.abits2List,
                 wbits2List=FLAGS.wbits2List,
                 stocRound=FLAGS.stocRound,
                 model_path1=FLAGS.model_path1,
                 model_path2=FLAGS.model_path2,
                 model_path3=FLAGS.model_path3,
                 ensembleThree=FLAGS.ensembleThree,
                 distill = FLAGS.distill,
                 inpgradreg = FLAGS.inpgradreg,
                 l2dbl = FLAGS.l2dbl,
                 l2cs = FLAGS.l2cs,
                 delay=FLAGS.delay,
                 adv=FLAGS.adv,
                 nb_iter=FLAGS.nb_iter)


if __name__ == '__main__':

    par = argparse.ArgumentParser()

    # Generic flags
    par.add_argument('--gpu', help='id of GPU to use')
    par.add_argument('--model_path', help='Path to save or load model')
    par.add_argument('--data_dir', help='Path to training data',
                     default='/tmp/mnist')
    par.add_argument(
        '--viz_enabled', help='Visualize adversarial ex.', action="store_true")
    par.add_argument(
        '--debug', help='Sets log level to DEBUG, otherwise INFO', action="store_true")
    par.add_argument(
        '--test', help='Test while training, takes longer', action="store_true")

    # Architecture and training specific flags
    par.add_argument('--nb_epochs', type=int, default=15,
                     help='Number of epochs to train model')
    par.add_argument('--nb_filters', type=int, default=64,
                     help='Number of filters in first layer')
    par.add_argument('--batch_size', type=int, default=128,
                     help='Size of training batches')
    par.add_argument('--lr', type=float, default=0.001,
                     help='Learning rate')
    par.add_argument('--rand', help='Stochastic weight layer?',
                     action="store_true")

    # Attack specific flags
    par.add_argument('--attack', type=int, default=0,
                     help='Attack type, 0=CW, 1=JSMA')
    par.add_argument("--eps", type=float, default=0.3)
    par.add_argument('--attack_iterations', type=int, default=50,
                     help='Number of iterations to run CW attack; 1000 is good')
    par.add_argument('--nb_samples', type=int,
                     default=10000, help='Nb of inputs to attack')
    par.add_argument(
        '--targeted', help='Run a targeted attack?', action="store_true")
    
    # EMPIR specific flags
    par.add_argument('--lowprecision', help='Use other low precision models absed on DoReFa net', action="store_true") # For DoReFa net style quantization
    par.add_argument('--wbits', type=int, default=0, help='No. of bits in weight representation')
    par.add_argument('--abits', type=int, default=0, help='No. of bits in activation representation')
    par.add_argument('--wbitsList', type=int, nargs='+', help='List of No. of bits in weight representation for different layers')
    par.add_argument('--abitsList', type=int, nargs='+', help='List of No. of bits in activation representation for different layers')
    par.add_argument('--stocRound', help='Stochastic rounding for weights (only in training) and activations?',
                     action="store_true")
    par.add_argument('--model_path1', help='Path where saved model1 is stored and can be loaded')
    par.add_argument('--model_path2', help='Path where saved model2 is stored and can be loaded')
    par.add_argument('--model_path3', help='Path where saved model3 is stored and can be loaded')
    par.add_argument('--ensembleThree', help='Use an ensemble of full precision and two low precision models', action="store_true") 
    par.add_argument('--wbits2', type=int, default=0, help='No. of bits in weight representation of model2, model1 specified using wbits')
    par.add_argument('--abits2', type=int, default=0, help='No. of bits in activation representation of model2, model2 specified using abits')
    par.add_argument('--wbits2List', type=int, nargs='+', help='List of No. of bits in weight representation for different layers of model2')
    par.add_argument('--abits2List', type=int, nargs='+', help='List of No. of bits in activation representation for different layers of model2')
    # extra flags for defensive distillation
    par.add_argument('--distill', help='Train the model using distillation', action="store_true") 
    par.add_argument('--student_epochs', type=int, default=50, help='No. of epochs for which the student model is trained')
    # extra flags for input gradient regularization
    par.add_argument('--inpgradreg', help='Train the model using input gradient regularization', action="store_true") 
    par.add_argument('--l2dbl', type=int, default=0, help='l2 double backprop penalty')
    par.add_argument('--l2cs', type=int, default=0, help='l2 certainty sensitivity penalty')
    # Adversarial training flags
    par.add_argument(
        '--adv', help='Adversarial training type?', type=int, default=0)
    par.add_argument('--delay', type=int,
                     default=10, help='Nb of epochs to delay adv training by')
    par.add_argument('--nb_iter', type=int,
                     default=40, help='Nb of iterations of PGD')

    FLAGS = par.parse_args()

    if FLAGS.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

    tf.app.run()
