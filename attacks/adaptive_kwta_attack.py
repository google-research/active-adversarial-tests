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
Adapted and heavily modified from
https://github.com/wielandbrendel/adaptive_attacks_paper/blob/master/01_kwta/kwta_attack.ipynb
"""

from typing import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn.functional
from tqdm import tqdm

import utils as ut


def __best_other_classes(logits: torch.Tensor,
    exclude: torch.Tensor) -> torch.Tensor:
  other_logits = logits - torch.nn.functional.one_hot(exclude,
                                                      num_classes=logits.shape[
                                                        -1]) * np.inf

  return other_logits.argmax(axis=-1)


def __logit_diff_loss_fn(model: Callable, x: torch.Tensor,
    classes: torch.Tensor,
    targeted: bool):
  with torch.no_grad():
    logits = model(x)

  if targeted:
    c_minimize = classes
    c_maximize = __best_other_classes(logits, classes)
  else:
    c_minimize = __best_other_classes(logits, classes)
    c_maximize = classes

  N = len(x)
  rows = range(N)

  logits_diffs = logits[rows, c_minimize] - logits[rows, c_maximize]
  assert logits_diffs.shape == (N,)

  return logits_diffs


def __es_gradient_estimator(loss_fn: Callable, x: torch.Tensor, y: torch.Tensor,
    n_samples: int, sigma: float, clip=False, bounds=(0, 1)):
  assert len(x) == len(y)
  assert n_samples > 0

  gradient = torch.zeros_like(x)
  with torch.no_grad():
    for k in range(n_samples // 2):
      noise = torch.randn_like(x)

      pos_theta = x + sigma * noise
      neg_theta = x - sigma * noise

      if clip:
        pos_theta = pos_theta.clip(*bounds)
        neg_theta = neg_theta.clip(*bounds)

      pos_loss = loss_fn(pos_theta, y)
      neg_loss = loss_fn(neg_theta, y)

      gradient += (pos_loss - neg_loss)[:, None, None, None] * noise

  gradient /= 2 * sigma * 2 * n_samples

  return gradient


def gradient_estimator_pgd(model: Callable,
    x: torch.Tensor, y: torch.Tensor,
    n_steps: int,
    step_size: float, epsilon: float, norm: ut.NormType,
    loss_fn: Optional[Callable] = None,
    random_start: bool = True,
    early_stopping: bool = False, targeted: bool = False):
  if loss_fn is None:
    loss_fn = lambda x, y: __logit_diff_loss_fn(model, x, y, targeted)

  assert len(x) == len(y)

  if random_start:
    delta = torch.rand_like(x)
    delta = ut.normalize(delta, norm)
    x_adv, delta = ut.clipping_aware_rescaling(x, delta, epsilon, norm=norm,
                                               growing=False, return_delta=True)
  else:
    x_adv = x
    delta = torch.zeros_like(x)

  if targeted:
    is_adversarial_fn = lambda x: model(x).argmax(-1) == y
  else:
    is_adversarial_fn = lambda x: model(x).argmax(-1) != y

  mask = ~is_adversarial_fn(x_adv)
  if not early_stopping:
    mask = torch.ones_like(mask)
  else:
    if mask.sum() == 0:
      return x_adv.detach(), ~mask.detach()

  if len(x) > 1:
    iterator = tqdm(range(n_steps))
  else:
    iterator = range(n_steps)
  for it in iterator:
    if it < 0.6 * n_steps:
      n_samples = 100
    elif it < 0.8 * n_steps:
      n_samples = 1000
    elif it >= 0.8 * n_steps:
      n_samples = 20000

    pert_x = (x + delta).clip(0, 1)
    grad_x = __es_gradient_estimator(loss_fn, pert_x[mask], y[mask], n_samples,
                                     epsilon)

    grad_x = ut.normalize(grad_x, norm)

    # update only subportion of deltas
    delta[mask] = delta[mask] - step_size * grad_x

    # project back to feasible set
    x_adv, delta = ut.clipping_aware_rescaling(x, delta, epsilon, norm=norm,
                                               growing=False, return_delta=True)

    mask = ~is_adversarial_fn(x_adv)
    # new_logit_diffs = loss_fn(x_adv, y)
    # mask = new_logit_diffs >= 0
    if not early_stopping:
      mask = torch.ones_like(mask)

    if early_stopping and mask.sum() == 0:
      break

  return x_adv.detach(), ~mask.detach()
