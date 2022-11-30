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

from typing import Callable

import torch

import utils as ut


def general_pgd(loss_fn: Callable, is_adversarial_fn: Callable, x: torch.Tensor,
    y: torch.Tensor, n_steps: int,
    step_size: float, epsilon: float, norm: ut.NormType,
    random_start: bool = True,
    early_stopping: bool = False,
    n_averaging_steps: int = 1):
  """Performs a projected gradient descent (PGD) for an arbitrary loss function
  and success criterion.

  :param loss_fn: Loss function to minimize.
  :param is_adversarial_fn: Check if examples are adversarial
  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param step_size: Size of the steps/learning rate.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
  :param random_start: Randomly start within the epsilon ball.
  :param early_stopping: Stop once an adversarial perturbation for all
    examples have been found.
  :param n_averaging_steps: Number over repetitions for every gradient
    calculation.
  :return: (Adversarial examples, attack success for each sample)
  """
  assert norm in ("linf", "l2", "l1")

  x_orig = x
  x = x.clone()

  if random_start:
    delta = torch.rand_like(x)
    delta = ut.normalize(delta, norm)
    x = ut.clipping_aware_rescaling(x_orig, delta, epsilon, norm=norm,
                                    growing=False)

  for step in range(n_steps):
    x = x.requires_grad_()

    # check early stopping
    with torch.no_grad():
      is_adv = is_adversarial_fn(x, y)
    if early_stopping and torch.all(is_adv): #
      return x.detach(), is_adv.detach()

    grad_x = torch.zeros_like(x)
    for _ in range(n_averaging_steps):
      # get gradient of cross-entropy wrt to input
      loss = loss_fn(x, y)
      grad_x += torch.autograd.grad(loss, x)[0].detach() / n_averaging_steps

    # normalize gradient
    grad_x = ut.normalize(grad_x, norm)

    # perform step
    delta = (x - x_orig).detach() - step_size * grad_x.detach()

    # project back to feasible set
    x = ut.clipping_aware_rescaling(x_orig, delta, epsilon, norm=norm,
                                    growing=False)
  del loss, grad_x

  with torch.no_grad():
    is_adv = is_adversarial_fn(x, y)

  return x.detach(), is_adv.detach()


def pgd(model: Callable, x: torch.Tensor, y: torch.Tensor, n_steps: int,
    step_size: float, epsilon: float, norm: ut.NormType,
    random_start: bool = True,
    early_stopping: bool = False,
    targeted: bool = False,
    n_averaging_steps: int = 1):
  """Performs a standard projected gradient descent (PGD) with a cross-entropy
  objective.

  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param step_size: Size of the steps/learning rate.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
  :param random_start: Randomly start within the epsilon ball.
  :param early_stopping: Stop once an adversarial perturbation for all
    examples have been found.
  :param targeted: Perform a targeted adversarial attack.
  :param n_averaging_steps: Number over repetitions for every gradient
    calculation.
  :return: (Adversarial examples, attack success for each sample)
  """
  assert norm in ("linf", "l2", "l1")

  criterion = torch.nn.CrossEntropyLoss()

  sign = 1 if targeted else -1

  return general_pgd(loss_fn=lambda x, y: sign * criterion(model(x), y),
                     is_adversarial_fn=lambda x, y: model(x).argmax(
                       -1) == y if targeted else model(x).argmax(-1) != y,
                     x=x, y=y, n_steps=n_steps, step_size=step_size,
                     epsilon=epsilon, norm=norm, random_start=random_start,
                     n_averaging_steps=n_averaging_steps,
                     early_stopping=early_stopping)