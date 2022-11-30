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

from cleverhans.attacks import Attack
import tensorflow as tf
import warnings
import numpy as np

from cleverhans import utils_tf
from cleverhans.utils_tf import clip_eta
from cleverhans.attacks import optimize_linear
from six.moves import xrange


def fgm(x,
    features,
    logit_means,
    y=None,
    eps=0.3,
    ord=np.inf,
    clip_min=None,
    clip_max=None,
    targeted=False,
    sanity_checks=True,
    projection="linf"):
  asserts = []

  # If a data range was specified, check that the input was in that range
  if clip_min is not None:
    asserts.append(utils_tf.assert_greater_equal(
        x, tf.cast(clip_min, x.dtype)))

  if clip_max is not None:
    asserts.append(utils_tf.assert_less_equal(x, tf.cast(clip_max, x.dtype)))

  if y is None:
    raise NotImplementedError("labels must be supplied")

  # Compute loss
  loss, loss_diff = loss_fn(logit_means=logit_means, labels=y,
                            features=features)
  if targeted:
    loss = -loss

  # Define gradient of loss wrt input
  grad, = tf.gradients(loss, x)

  # optimal_perturbation = optimize_linear(grad, eps, ord)

  if projection == "l2":
    square = tf.maximum(1e-12,
                        tf.reduce_sum(tf.square(grad),
                                      reduction_indices=list(
                                          xrange(1, len(grad.get_shape()))),
                                      keepdims=True))
    optimal_perturbation = grad / tf.sqrt(square)

    # Scale perturbation to be the solution for the norm=eps rather than
    # norm=1 problem
    scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)
  else:
    optimal_perturbation = tf.sign(grad)
    scaled_perturbation = utils_tf.mul(eps, optimal_perturbation)

  # Add perturbation to original example to obtain adversarial example
  adv_x = x + scaled_perturbation
  adv_x = x + utils_tf.clip_eta(adv_x - x, ord, eps)

  # If clipping is needed, reset all values outside of [clip_min, clip_max]
  if (clip_min is not None) or (clip_max is not None):
    # We don't currently support one-sided clipping
    assert clip_min is not None and clip_max is not None
    adv_x = utils_tf.clip_by_value(adv_x, clip_min, clip_max)

  if sanity_checks:
    with tf.control_dependencies(asserts):
      adv_x = tf.identity(adv_x)

  return adv_x, loss_diff


def loss_fn(logit_means,
    sentinel=None,
    labels=None,
    features=None,
    dim=-1, ):
  """
  Wrapper around tf.nn.softmax_cross_entropy_with_logits_v2 to handle
  deprecated warning
  """
  # Make sure that all arguments were passed as named arguments.
  if sentinel is not None:
    name = "softmax_cross_entropy_with_logits"
    raise ValueError("Only call `%s` with "
                     "named arguments (labels=..., logits=..., ...)"
                     % name)
  if labels is None or features is None:
    raise ValueError("Both labels and features must be provided.")

  labels_oh = tf.stop_gradient(labels)
  labels = tf.argmax(labels_oh, -1)

  # find target labels
  # ignore logit means for classes we are not considering (only relevant for
  # binarization test)
  logit_means = logit_means[:labels_oh.shape[-1]]
  distances = tf.reduce_mean((tf.expand_dims(features, 1) - tf.expand_dims(logit_means, 0)) ** 2, -1)
  distances = distances + 1e9 * labels_oh
  target_labels = tf.argmin(distances, -1)


  # target_labels = (labels + 1) % 2

  target_logit_means = tf.gather(logit_means, target_labels)
  source_logit_means = tf.gather(logit_means, labels)

  dist = tf.reduce_mean((features - target_logit_means) ** 2, -1)
  dist_other = tf.reduce_mean((features - source_logit_means) ** 2, -1)

  dist_diff = dist - dist_other

  # invert sign so that we perform gradient ascent instead of descent
  return -tf.reduce_sum(dist), dist_diff


class FeatureSpaceProjectedGradientDescent(Attack):
  """
  This class implements either the Basic Iterative Method
  (Kurakin et al. 2016) when rand_init is set to 0. or the
  Madry et al. (2017) method when rand_minmax is larger than 0.
  Paper link (Kurakin et al. 2016): https://arxiv.org/pdf/1607.02533.pdf
  Paper link (Madry et al. 2017): https://arxiv.org/pdf/1706.06083.pdf
  :param model: cleverhans.model.Model
  :param sess: optional tf.Session
  :param dtypestr: dtype of the data
  :param default_rand_init: whether to use random initialization by default
  :param kwargs: passed through to super constructor
  """

  def __init__(self, model, logit_means, sess=None, dtypestr='float32',
      default_rand_init=True, max_steps=99999, projection='linf', **kwargs):
    """
    Create a ProjectedGradientDescent instance.
    Note: the model parameter should be an instance of the
    cleverhans.model.Model abstraction provided by CleverHans.
    """

    super(FeatureSpaceProjectedGradientDescent, self).__init__(model, sess=sess,
                                                               dtypestr=dtypestr,
                                                               **kwargs)
    self.feedable_kwargs = ('eps', 'eps_iter', 'y', 'y_target', 'clip_min',
                            'clip_max')
    self.structural_kwargs = ['ord', 'nb_iter', 'rand_init', 'sanity_checks']
    self.logit_means = logit_means
    self.default_rand_init = default_rand_init
    self.max_steps = max_steps
    self.projection = projection

  def generate(self, x, **kwargs):
    """
    Generate symbolic graph for adversarial examples and return.
    :param x: The model's symbolic inputs.
    :param kwargs: See `parse_params`
    """
    # Parse and save attack-specific parameters
    assert self.parse_params(**kwargs)

    asserts = []

    # If a data range was specified, check that the input was in that range
    if self.clip_min is not None:
      asserts.append(utils_tf.assert_greater_equal(x,
                                                   tf.cast(self.clip_min,
                                                           x.dtype)))

    if self.clip_max is not None:
      asserts.append(utils_tf.assert_less_equal(x,
                                                tf.cast(self.clip_max,
                                                        x.dtype)))

    # Initialize loop variables
    if self.rand_init:
      eta = tf.random_uniform(tf.shape(x),
                              tf.cast(-self.rand_minmax, x.dtype),
                              tf.cast(self.rand_minmax, x.dtype),
                              dtype=x.dtype)
    else:
      eta = tf.zeros(tf.shape(x))

    # Clip eta
    eta = clip_eta(eta, self.ord, self.eps)
    adv_x = x + eta
    if self.clip_min is not None or self.clip_max is not None:
      adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

    if self.y_target is not None:
      raise NotImplementedError("Targeted mode not fully implemented yet")
    elif self.y is not None:
      y = self.y
      targeted = False
    else:
      raise NotImplementedError("labels must be supplied")

    y_kwarg = 'y_target' if targeted else 'y'
    fgm_params = {
        'eps': self.eps_iter,
        y_kwarg: y,
        'ord': self.ord,
        'clip_min': self.clip_min,
        'clip_max': self.clip_max,
        "logit_means": self.logit_means
    }
    if self.ord == 1:
      raise NotImplementedError("It's not clear that FGM is a good inner loop"
                                " step for PGD when ord=1, because ord=1 FGM "
                                " changes only one pixel at a time. We need "
                                " to rigorously test a strong ord=1 PGD "
                                "before enabling this feature.")

    # Use getattr() to avoid errors in eager execution attacks

    def cond(i, _, _2, loss_diff, first_idx_done):
      return tf.reduce_any(
          tf.logical_or(
              tf.less(i, self.nb_iter),
              tf.logical_and(
                tf.greater(loss_diff, tf.zeros([])),
                tf.less(i, self.max_steps)
              )
              # tf.logical_or(
              #    tf.less_equal(first_idx_done, tf.zeros([])),
              #    tf.logical_and(
              #        i < 2000,
              #        tf.logical_not(
              #            tf.logical_and(
              #                tf.less(loss_diff, tf.zeros([])),
              #                tf.less(first_idx_done + 10, i)
              #            ))
              #    )
              # )
          )
      )

    def body(i, adv_x, _, _2, first_idx_done):
      adv_x_before = adv_x
      adv_x, loss_diff = fgm(adv_x, features=self.model.get_mmd_features(adv_x),
                             **fgm_params, projection=self.projection)

      # adv_x = tf.Print(adv_x, [i, first_idx_done, loss_diff])

      # Clipping perturbation eta to self.ord norm ball
      eta = adv_x - x
      eta = clip_eta(eta, self.ord, self.eps)
      adv_x = x + eta

      # Redo the clipping.
      # FGM already did it, but subtracting and re-adding eta can add some
      # small numerical error.
      if self.clip_min is not None or self.clip_max is not None:
        adv_x = utils_tf.clip_by_value(adv_x, self.clip_min, self.clip_max)

      first_idx_done = tf.where(
          tf.logical_and(first_idx_done > 0, loss_diff < 0),
          first_idx_done,
          i * tf.where(loss_diff < 0, tf.ones(tf.shape(adv_x)[0]), tf.zeros(tf.shape(adv_x)[0]))
      )

      return i + 1, adv_x, adv_x_before, loss_diff, first_idx_done

    _, _, adv_x, _, _ = tf.while_loop(cond, body,
                                      [tf.zeros([]), adv_x, adv_x,
                                       tf.ones(tf.shape(adv_x)[0]),
                                       -1 * tf.ones(tf.shape(adv_x)[0])],
                                      back_prop=True)

    # Asserts run only on CPU.
    # When multi-GPU eval code tries to force all PGD ops onto GPU, this
    # can cause an error.
    common_dtype = tf.float64
    asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps_iter,
                                                      dtype=common_dtype),
                                              tf.cast(self.eps,
                                                      dtype=common_dtype)))
    if self.ord == np.inf and self.clip_min is not None:
      # The 1e-6 is needed to compensate for numerical error.
      # Without the 1e-6 this fails when e.g. eps=.2, clip_min=.5,
      # clip_max=.7
      asserts.append(utils_tf.assert_less_equal(tf.cast(self.eps, x.dtype),
                                                1e-6 + tf.cast(self.clip_max,
                                                               x.dtype)
                                                - tf.cast(self.clip_min,
                                                          x.dtype)))

    if self.sanity_checks:
      with tf.control_dependencies(asserts):
        adv_x = tf.identity(adv_x)

    return adv_x

  def parse_params(self,
      eps=0.3,
      eps_iter=0.05,
      nb_iter=10,
      y=None,
      ord=np.inf,
      clip_min=None,
      clip_max=None,
      y_target=None,
      rand_init=None,
      rand_minmax=0.3,
      sanity_checks=True,
      **kwargs):
    """
    Take in a dictionary of parameters and applies attack-specific checks
    before saving them as attributes.
    Attack-specific parameters:
    :param eps: (optional float) maximum distortion of adversarial example
                compared to original input
    :param eps_iter: (optional float) step size for each attack iteration
    :param nb_iter: (optional int) Number of attack iterations.
    :param y: (optional) A tensor with the true labels.
    :param y_target: (optional) A tensor with the labels to target. Leave
                     y_target=None if y is also set. Labels should be
                     one-hot-encoded.
    :param ord: (optional) Order of the norm (mimics Numpy).
                Possible values: np.inf, 1 or 2.
    :param clip_min: (optional float) Minimum input component value
    :param clip_max: (optional float) Maximum input component value
    :param sanity_checks: bool Insert tf asserts checking values
        (Some tests need to run with no sanity checks because the
         tests intentionally configure the attack strangely)
    """

    # Save attack-specific parameters
    self.eps = eps
    if rand_init is None:
      rand_init = self.default_rand_init
    self.rand_init = rand_init
    if self.rand_init:
      self.rand_minmax = eps
    else:
      self.rand_minmax = 0.
    self.eps_iter = eps_iter
    self.nb_iter = nb_iter
    self.y = y
    self.y_target = y_target
    self.ord = ord
    self.clip_min = clip_min
    self.clip_max = clip_max

    if isinstance(eps, float) and isinstance(eps_iter, float):
      # If these are both known at compile time, we can check before anything
      # is run. If they are tf, we can't check them yet.
      assert eps_iter <= eps, (eps_iter, eps)

    if self.y is not None and self.y_target is not None:
      raise ValueError("Must not set both y and y_target")
    # Check if order of the norm is acceptable given current implementation
    if self.ord not in [np.inf, 1, 2]:
      raise ValueError("Norm order must be either np.inf, 1, or 2.")
    self.sanity_checks = sanity_checks

    if len(kwargs.keys()) > 0:
      warnings.warn("kwargs is unused and will be removed on or after "
                    "2019-04-26.")

    return True
