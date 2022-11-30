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
based on cleverhans' cifar10_tutorial_tf except that it uses Madry's wide-ResNet
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import flags
import time

from cleverhans.attacks import MadryEtAl, CarliniWagnerL2
from cleverhans.augmentation import random_horizontal_flip, random_shift
from cleverhans.dataset import CIFAR10
from cleverhans.loss import CrossEntropy
from cleverhans.model_zoo.all_convolutional import ModelAllConvolutional
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet as ResNet
from cleverhans.utils_tf import initialize_uninitialized_global_variables
from cleverhans.train import train
from cleverhans.utils import AccuracyReport, set_log_level
from cleverhans.utils_tf import tf_model_load, model_eval

from cleverhans.evaluation import batch_eval
import math
import tqdm
import os
import tf_robustify
from tensorboardX import SummaryWriter
import pickle

FLAGS = flags.FLAGS

NB_EPOCHS = 6
BATCH_SIZE = 128
LEARNING_RATE = 0.001
CLEAN_TRAIN = True
BACKPROP_THROUGH_ATTACK = False
NB_FILTERS = 64

ARCHITECTURE = 'ResNet'
LOAD_MODEL = True

os.makedirs('logs', exist_ok=True)
swriter = SummaryWriter('logs')

def cifar10_tutorial(
        train_start=0,
        train_end=60000,
        test_start=0,
        test_end=10000,
        nb_epochs=NB_EPOCHS,
        batch_size=BATCH_SIZE,
        architecture=ARCHITECTURE,
        load_model=LOAD_MODEL,
        ckpt_dir='None',
        learning_rate=LEARNING_RATE,
        clean_train=CLEAN_TRAIN,
        backprop_through_attack=BACKPROP_THROUGH_ATTACK,
        nb_filters=NB_FILTERS,
        num_threads=None,
        label_smoothing=0.):
    """
    CIFAR10 cleverhans tutorial
    :param train_start: index of first training set example
    :param train_end: index of last training set example
    :param test_start: index of first test set example
    :param test_end: index of last test set example
    :param nb_epochs: number of epochs to train model
    :param batch_size: size of training batches
    :param learning_rate: learning rate for training
    :param clean_train: perform normal training on clean examples only
                        before performing adversarial training.
    :param backprop_through_attack: If True, backprop through adversarial
                                    example construction process during
                                    adversarial training.
    :param label_smoothing: float, amount of label smoothing for cross entropy
    :return: an AccuracyReport object
    """

    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()

    # Set TF random seed to improve reproducibility
    tf.set_random_seed(int(time.time() * 1000) % 2**31)
    np.random.seed(int(time.time() * 1001) % 2**31)

    # Set logging level to see debug information
    set_log_level(logging.DEBUG)

    # Create TF session
    if num_threads:
        config_args = dict(intra_op_parallelism_threads=1)
    else:
        config_args = {}
    sess = tf.Session(config=tf.ConfigProto(**config_args))

    # Get CIFAR10 data
    data = CIFAR10(train_start=train_start, train_end=train_end,
                   test_start=test_start, test_end=test_end)
    dataset_size = data.x_train.shape[0]
    dataset_train = data.to_tensorflow()[0]
    dataset_train = dataset_train.map(
        lambda x, y: (random_shift(random_horizontal_flip(x)), y), 4)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(16)
    x_train, y_train = data.get_set('train')

    pgd_train = None
    if FLAGS.load_pgd_train_samples:
        pgd_path = os.path.expanduser('~/data/advhyp/{}/samples'.format(FLAGS.load_pgd_train_samples))
        x_train = np.load(os.path.join(pgd_path, 'train_clean.npy'))
        y_train = np.load(os.path.join(pgd_path, 'train_y.npy'))
        pgd_train = np.load(os.path.join(pgd_path, 'train_pgd.npy'))
        if x_train.shape[1] == 3:
            x_train = x_train.transpose((0, 2, 3, 1))
            pgd_train = pgd_train.transpose((0, 2, 3, 1))
        if len(y_train.shape) == 1:
            y_tmp = np.zeros((len(y_train), np.max(y_train)+1), y_train.dtype)
            y_tmp[np.arange(len(y_tmp)), y_train] = 1.
            y_train = y_tmp

    x_test, y_test = data.get_set('test')
    pgd_test = None
    if FLAGS.load_pgd_test_samples:
        pgd_path = os.path.expanduser('~/data/advhyp/{}/samples'.format(FLAGS.load_pgd_test_samples))
        x_test = np.load(os.path.join(pgd_path, 'test_clean.npy'))
        y_test = np.load(os.path.join(pgd_path, 'test_y.npy'))
        pgd_test = np.load(os.path.join(pgd_path, 'test_pgd.npy'))
        if x_test.shape[1] == 3:
            x_test = x_test.transpose((0, 2, 3, 1))
            pgd_test = pgd_test.transpose((0, 2, 3, 1))
        if len(y_test.shape) == 1:
            y_tmp = np.zeros((len(y_test), np.max(y_test)+1), y_test.dtype)
            y_tmp[np.arange(len(y_tmp)), y_test] = 1.
            y_test = y_tmp

    train_idcs = np.arange(len(x_train))
    np.random.shuffle(train_idcs)
    x_train, y_train = x_train[train_idcs], y_train[train_idcs]
    if pgd_train is not None:
        pgd_train = pgd_train[train_idcs]
    test_idcs = np.arange(len(x_test))[:FLAGS.test_size]
    np.random.shuffle(test_idcs)
    x_test, y_test = x_test[test_idcs], y_test[test_idcs]
    if pgd_test is not None:
        pgd_test = pgd_test[test_idcs]

    # Use Image Parameters
    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    # Define input TF placeholder
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols,
                                          nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))

    # Train an MNIST model
    train_params = {
        'nb_epochs': nb_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate
    }
    eval_params = {'batch_size': batch_size}
    pgd_params = {
        # ord: ,
        'eps': FLAGS.eps,
        'eps_iter': (FLAGS.eps / 5),
        'nb_iter': 10,
        'clip_min': 0,
        'clip_max': 255
    }
    cw_params = {
        'binary_search_steps': FLAGS.cw_search_steps, 
        'max_iterations': FLAGS.cw_steps,  #1000
        'abort_early': True,
        'learning_rate': FLAGS.cw_lr, 
        'batch_size': batch_size,
        'confidence': 0,
        'initial_const': FLAGS.cw_c,
        'clip_min': 0,
        'clip_max': 255
    }

    # Madry dosen't divide by 255
    x_train *= 255
    x_test *= 255
    if pgd_train is not None:
        pgd_train *= 255
    if pgd_test is not None:
        pgd_test *= 255

    print('x_train amin={} amax={}'.format(np.amin(x_train), np.amax(x_train)))
    print('x_test amin={} amax={}'.format(np.amin(x_test), np.amax(x_test)))

    print('clip_min : {}, clip_max : {}  >> CHECK WITH WHICH VALUES THE CLASSIFIER WAS PRETRAINED !!! <<'
          .format(pgd_params['clip_min'], pgd_params['clip_max']))

    rng = np.random.RandomState()  # [2017, 8, 30]
    debug_dict = dict() if FLAGS.save_debug_dict else None

    def do_eval(preds, x_set, y_set, report_key, is_adv=None, predictor=None, x_adv=None):
        if predictor is None:
            acc = model_eval(sess, x, y, preds, x_set, y_set, args=eval_params)
        else:
            do_eval(preds, x_set, y_set, report_key, is_adv=is_adv)
            if x_adv is not None:
                x_set_adv, = batch_eval(sess, [x], [x_adv], [x_set], batch_size=batch_size)
                assert x_set.shape == x_set_adv.shape
                x_set = x_set_adv
            n_batches = math.ceil(x_set.shape[0] / batch_size)
            p_set, p_det = np.concatenate([predictor.send(x_set[b*batch_size:(b+1)*batch_size]) for b in tqdm.trange(n_batches)]).T
            acc = np.equal(p_set, y_set[:len(p_set)].argmax(-1)).mean()
            # if is_adv:
                # import IPython ; IPython.embed() ; exit(1)
            if FLAGS.save_debug_dict:
                debug_dict['x_set'] = x_set
                debug_dict['y_set'] = y_set
                ddfn = 'logs/debug_dict_{}.pkl'.format('adv' if is_adv else 'clean')
                if not os.path.exists(ddfn):
                    with open(ddfn, 'wb') as f:
                        pickle.dump(debug_dict, f)
                debug_dict.clear()
        if is_adv is None:
            report_text = None
        elif is_adv:
            report_text = 'adversarial'
        else:
            report_text = 'legitimate'
        if report_text:
            print('Test accuracy on %s examples %s: %0.4f' % (report_text, 'with correction' if predictor is not None else 'without correction', acc))
            if is_adv is not None:
                label = 'test_acc_{}_{}'.format(report_text, 'corrected' if predictor else 'uncorrected')
                swriter.add_scalar(label, acc)
                if predictor is not None:
                    detect = np.equal(p_det, is_adv).mean()
                    label = 'test_det_{}_{}'.format(report_text, 'corrected' if predictor else 'uncorrected')
                    print(label, detect)
                    swriter.add_scalar(label, detect)
                    label = 'test_dac_{}_{}'.format(report_text, 'corrected' if predictor else 'uncorrected')
                    swriter.add_scalar(label, np.equal(p_set, y_set[:len(p_set)].argmax(-1))[np.equal(p_det, is_adv)].mean())

        return acc

    if clean_train:
        if architecture == 'ConvNet':
            model = ModelAllConvolutional('model1', nb_classes, nb_filters,
                                          input_shape=[32, 32, 3])
        elif architecture == 'ResNet':
            model = ResNet(scope='ResNet')
        else:
            raise Exception('Specify valid classifier architecture!')

        preds = model.get_logits(x)
        loss = CrossEntropy(model, smoothing=label_smoothing)

        if load_model:
            model_name = 'naturally_trained'
            if FLAGS.load_adv_trained:
                model_name = 'adv_trained'
            if ckpt_dir is not 'None':
                ckpt = tf.train.get_checkpoint_state(os.path.join(
                    os.path.expanduser(ckpt_dir), model_name))
            else:
                ckpt = tf.train.get_checkpoint_state(
                    './models/' + model_name)
            ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

            saver = tf.train.Saver(var_list=dict((v.name.split('/', 1)[1].split(':')[0], v) for v in tf.global_variables()))
            saver.restore(sess, ckpt_path)
            print('\nMODEL SUCCESSFULLY LOADED from : {}'.format(ckpt_path))

            initialize_uninitialized_global_variables(sess)

        else:
            def evaluate():
                do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False)

            train(sess, loss, None, None,
                  dataset_train=dataset_train, dataset_size=dataset_size,
                  evaluate=evaluate, args=train_params, rng=rng,
                  var_list=model.get_params())

        logits_op = preds.op
        while logits_op.type != 'MatMul':
            logits_op = logits_op.inputs[0].op
        latent_x_tensor, weights = logits_op.inputs
        logits_tensor = preds

        nb_classes = weights.shape[-1].value

        if not FLAGS.save_pgd_samples:
            noise_eps = FLAGS.noise_eps.split(',')
            if FLAGS.noise_eps_detect is None:
                FLAGS.noise_eps_detect = FLAGS.noise_eps
            noise_eps_detect = FLAGS.noise_eps_detect.split(',')
            if pgd_train is not None:
                pgd_train = pgd_train[:FLAGS.n_collect]
            if not FLAGS.passthrough:
                predictor = tf_robustify.collect_statistics(x_train[:FLAGS.n_collect], y_train[:FLAGS.n_collect], x, sess, logits_tensor=logits_tensor, latent_x_tensor=latent_x_tensor, weights=weights, nb_classes=nb_classes, p_ratio_cutoff=FLAGS.p_ratio_cutoff, noise_eps=noise_eps, noise_eps_detect=noise_eps_detect, pgd_eps=pgd_params['eps'], pgd_lr=pgd_params['eps_iter'] / pgd_params['eps'], pgd_iters=pgd_params['nb_iter'], save_alignments_dir='logs/stats' if FLAGS.save_alignments else None, load_alignments_dir=os.path.expanduser('~/data/advhyp/madry/stats') if FLAGS.load_alignments else None, clip_min=pgd_params['clip_min'], clip_max=pgd_params['clip_max'], batch_size=batch_size, num_noise_samples=FLAGS.num_noise_samples, debug_dict=debug_dict, debug=FLAGS.debug, targeted=False, pgd_train=pgd_train, fit_classifier=FLAGS.fit_classifier, clip_alignments=FLAGS.clip_alignments, just_detect=FLAGS.just_detect)
            else:
                def _predictor():
                    _x = yield
                    while(_x is not None):
                        _y = sess.run(preds, {x: _x}).argmax(-1)
                        _x = yield np.stack((_y, np.zeros_like(_y)), -1)
                predictor = _predictor()
            next(predictor)
            if FLAGS.save_alignments:
                exit(0)

            # Evaluate the accuracy of the model on clean examples
            acc_clean = do_eval(preds, x_test, y_test, 'clean_train_clean_eval', False, predictor=predictor)

        # Initialize the PGD attack object and graph
        if FLAGS.attack == 'pgd':
            pgd = MadryEtAl(model, sess=sess)
            adv_x = pgd.generate(x, **pgd_params)
        elif FLAGS.attack == 'cw':
            cw = CarliniWagnerL2(model, sess=sess)
            adv_x = cw.generate(x, **cw_params)
        elif FLAGS.attack == 'mean':
            pgd = MadryEtAl(model, sess=sess)
            mean_eps = FLAGS.mean_eps * FLAGS.eps
            def _attack_mean(x):
                x_many = tf.tile(x[None], (FLAGS.mean_samples, 1, 1, 1))
                x_noisy = x_many + tf.random_uniform(x_many.shape, -mean_eps, mean_eps)
                x_noisy = tf.clip_by_value(x_noisy, 0, 255)
                x_pgd = pgd.generate(x_noisy, **pgd_params)
                x_clip = tf.minimum(x_pgd, x_many + FLAGS.eps)
                x_clip = tf.maximum(x_clip, x_many - FLAGS.eps)
                x_clip = tf.clip_by_value(x_clip, 0, 255)
                return x_clip
            adv_x = tf.map_fn(_attack_mean, x)
            adv_x = tf.reduce_mean(adv_x, 1)

            
        preds_adv = model.get_logits(adv_x)

        if FLAGS.save_pgd_samples:
            for ds, y, name in ((x_train, y_train, 'train'), (x_test, y_test, 'test')):
                train_batches = math.ceil(len(ds) / FLAGS.batch_size)
                train_pgd = np.concatenate([sess.run(adv_x, {x: ds[b*FLAGS.batch_size:(b+1)*FLAGS.batch_size]}) for b in tqdm.trange(train_batches)])
                np.save('logs/{}_clean.npy'.format(name), ds / 255.)
                np.save('logs/{}_y.npy'.format(name), y)
                train_pgd /= 255.
                np.save('logs/{}_pgd.npy'.format(name), train_pgd)
            exit(0)

        # Evaluate the accuracy of the model on adversarial examples
        if not FLAGS.load_pgd_test_samples:
            acc_pgd = do_eval(preds_adv, x_test, y_test, 'clean_train_adv_eval', True, predictor=predictor, x_adv=adv_x)
        else:
            acc_pgd = do_eval(preds, pgd_test, y_test, 'clean_train_adv_eval', True, predictor=predictor)
        swriter.add_scalar('test_acc_mean', (acc_clean + acc_pgd) / 2., 0)

        print('Repeating the process, using adversarial training')

    exit(0)
    # Create a new model and train it to be robust to MadryEtAl
    if architecture == 'ConvNet':
        model2 = ModelAllConvolutional('model2', nb_classes, nb_filters,
                                       input_shape=[32, 32, 3])
    elif architecture == 'ResNet':
        model = ResNet()
    else:
        raise Exception('Specify valid classifier architecture!')

    pgd2 = MadryEtAl(model2, sess=sess)

    def attack(x):
        return pgd2.generate(x, **pgd_params)

    loss2 = CrossEntropy(model2, smoothing=label_smoothing, attack=attack)
    preds2 = model2.get_logits(x)
    adv_x2 = attack(x)

    if not backprop_through_attack:
        # For some attacks, enabling this flag increases the cost of
        # training, but gives the defender the ability to anticipate how
        # the atacker will change their strategy in response to updates to
        # the defender's parameters.
        adv_x2 = tf.stop_gradient(adv_x2)
    preds2_adv = model2.get_logits(adv_x2)

    if load_model:
        if ckpt_dir is not 'None':
            ckpt = tf.train.get_checkpoint_state(os.path.join(
                os.path.expanduser(ckpt_dir), 'adv_trained'))
        else:
            ckpt = tf.train.get_checkpoint_state('./models/adv_trained')
        ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

        assert ckpt_path and tf_model_load(
            sess, file_path=ckpt_path), '\nMODEL LOADING FAILED'
        print('\nMODEL SUCCESSFULLY LOADED from : {}'.format(ckpt_path))

        initialize_uninitialized_global_variables(sess)

    else:

        def evaluate2():
            # Accuracy of adversarially trained model on legitimate test inputs
            do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
            # Accuracy of the adversarially trained model on adversarial
            # examples
            do_eval(preds2_adv, x_test, y_test, 'adv_train_adv_eval', True)

        # Perform and evaluate adversarial training
        train(sess, loss2, None, None,
              dataset_train=dataset_train, dataset_size=dataset_size,
              evaluate=evaluate2, args=train_params, rng=rng,
              var_list=model2.get_params())

    # Evaluate model
    do_eval(preds2, x_test, y_test, 'adv_train_clean_eval', False)
    do_eval(preds2_adv, x_test, y_test, 'adv_train_adv_eval', True)

    return report


def main(argv=None):
    from cleverhans_tutorials import check_installation
    check_installation(__file__)

    cifar10_tutorial(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size,
                     learning_rate=FLAGS.learning_rate,
                     clean_train=FLAGS.clean_train,
                     architecture=FLAGS.architecture,
                     load_model=FLAGS.load_model,
                     ckpt_dir=FLAGS.ckpt_dir,
                     backprop_through_attack=FLAGS.backprop_through_attack,
                     nb_filters=FLAGS.nb_filters,
                     test_end=FLAGS.test_size)


if __name__ == '__main__':
    flags.DEFINE_integer('nb_filters', NB_FILTERS,
                            'Model size multiplier')
    flags.DEFINE_integer('nb_epochs', NB_EPOCHS,
                            'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', BATCH_SIZE,
                            'Size of training batches')
    flags.DEFINE_float('learning_rate', LEARNING_RATE,
                        'Learning rate for training')
    flags.DEFINE_string('architecture', ARCHITECTURE,
                        'Architecture [ResNet, ConvNet]')
    flags.DEFINE_bool('load_model', LOAD_MODEL, 'Load Pretrained Model')
    flags.DEFINE_string('ckpt_dir', '~/models/advhyp/madry/models',
                        'ckpt_dir [path_to_checkpoint_dir]')
    flags.DEFINE_bool('clean_train', CLEAN_TRAIN, 'Train on clean examples')
    flags.DEFINE_bool('backprop_through_attack', BACKPROP_THROUGH_ATTACK,
                        ('If True, backprop through adversarial example '
                        'construction process during adversarial training'))
    flags.DEFINE_integer('n_collect', 10000, '')
    flags.DEFINE_float('p_ratio_cutoff', .999, '')
    flags.DEFINE_float('eps', 8., '')
    flags.DEFINE_string('noise_eps', 'n18.0,n24.0,n30.0', '')
    flags.DEFINE_string('noise_eps_detect', 'n30.0', '')
    flags.DEFINE_bool('debug', False, 'for debugging')
    flags.DEFINE_integer('test_size', 10000, '')
    flags.DEFINE_bool('save_alignments', False, '')
    flags.DEFINE_bool('load_alignments', False, '')
    flags.DEFINE_integer('num_noise_samples', 256, '')
    flags.DEFINE_integer('rep', 0, '')
    flags.DEFINE_bool('save_debug_dict', False, '')
    flags.DEFINE_bool('save_pgd_samples', False, '')
    flags.DEFINE_string('load_pgd_train_samples', None, '')
    flags.DEFINE_string('load_pgd_test_samples', None, '')
    flags.DEFINE_bool('fit_classifier', True, '')
    flags.DEFINE_bool('clip_alignments', True, '')
    flags.DEFINE_string('attack', 'pgd', '')
    flags.DEFINE_bool('passthrough', False, '')
    flags.DEFINE_integer('cw_steps', 300, '')
    flags.DEFINE_integer('cw_search_steps', 20, '')
    flags.DEFINE_float('cw_lr', 1e-1, '')
    flags.DEFINE_float('cw_c', 1e-4, '')
    flags.DEFINE_bool('just_detect', False, '')
    flags.DEFINE_integer('mean_samples', 16, '')
    flags.DEFINE_float('mean_eps', .1, '')
    flags.DEFINE_bool('load_adv_trained', False, '')

    tf.app.run()
