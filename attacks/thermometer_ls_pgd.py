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

import numpy as np
import torch

import utils as ut


def general_thermometer_ls_pgd(
    loss_fn: Callable, is_adversarial_fn: Callable,
    x: torch.Tensor,
    y: torch.Tensor, n_steps: int,
    step_size: float, epsilon: float, norm: ut.NormType,
    l: int, temperature: float = 1.0, annealing_factor=1.0 / 1.2,
    random_start: bool = True,
    n_restarts=0,
    early_stopping: bool = False,
    n_averaging_steps: int = 1):
  """Performs a logit-space projected gradient descent (PGD) for an arbitrary loss function
  and success criterion for a thermometer-encoded model.

  :param loss_fn: Loss function to minimize.
  :param is_adversarial_fn: Check if examples are adversarial
  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param step_size: Size of the steps/learning rate.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
  :param l:
  :param temperature:
  :param annealing_factor:
  :param random_start: Randomly start within the epsilon ball.
  :param early_stopping: Stop once an adversarial perturbation for all
    examples have been found.
  :param n_averaging_steps: Number over repetitions for every gradient
    calculation.
  :return: (Adversarial examples, attack success for each sample)
  """
  assert norm in ("linf",), "LS-PGD only supports linf norm"
  assert random_start, "LS-PGD only works with random starts"
  #assert epsilon < 1.0 / l, f"Epsilon ({epsilon}) must be smaller " \
  #                          f"than 1.0/l ({1.0 / l})"

  def one_hot(y):
    L = torch.arange(l, dtype=x.dtype, device=x.device)
    L = L.view((1, 1, l, 1, 1)) / l
    y = torch.unsqueeze(y, 2)
    y = torch.logical_and(
        y >= L,
        y <= L + 1 / l).float()
    return y

  def init_mask(x):
    # Compute the mask over the bits that we are allowed to attack
    mask = torch.zeros((len(x), 3, l, x.shape[-2], x.shape[-1]), dtype=x.dtype,
                       device=x.device)
    for alpha in np.linspace(0, 1, l):
      mask += one_hot(torch.maximum(torch.zeros_like(x), x - alpha * epsilon))
      mask += one_hot(torch.minimum(torch.ones_like(x), x + alpha * epsilon))
    mask = (mask > 0).float()
    return mask

  def get_final_x(u):
    x = torch.argmax(u, 2) / l

    # now move x as close as possible to x_orig without changing
    # the argmax of the logits
    delta = x - x_orig
    delta[delta > 0] = torch.floor(delta[delta > 0] * l) / l
    delta[delta < 0] = torch.ceil(delta[delta < 0] * l) / l

    # only relevant for debugging:
    # assert torch.all(torch.abs(delta) <= 1.0/l)

    delta = torch.minimum(torch.ones_like(delta) * epsilon, delta)
    delta = torch.maximum(-torch.ones_like(delta) * epsilon, delta)
    x = x_orig + delta

    # only relevant for debugging:
    # project back to feasible set (if everything was correct before,
    # this shouldn't change anything)
    # x2 = ut.clipping_aware_rescaling(x_orig, delta, epsilon, norm=norm,
    #                                growing=False)
    # assert torch.all(torch.abs(x - x2) < 1e-8)

    return x

  x_final = x.clone()
  x_orig = x
  mask = init_mask(x_orig)

  for _ in range(n_restarts + 1):
    x_logits = torch.randn_like(mask)
    for step in range(n_steps):
      # mask u so that x(u) is within the epsilon ball
      x_logits = x_logits * mask - (1.0 - mask) * 1e12
      # check early stopping
      x = get_final_x(x_logits)
      is_adv = is_adversarial_fn(x, y)
      # print(is_adv[:32].long())
      if early_stopping and torch.all(is_adv):  #
        return x.detach(), is_adv.detach()

      x_logits = x_logits.requires_grad_()
      x_thermometer = torch.softmax(x_logits / temperature, 2)
      # convert something like [0, 0, 1, 0, .., 0] to [1, 1, 1, 0, ..., 0]
      x_thermometer = torch.flip(
          torch.cumsum(torch.flip(x_thermometer, (2,)), 2), (2,))
      x_thermometer = x_thermometer.view((
          x_thermometer.shape[0], -1, x_thermometer.shape[-2],
          x_thermometer.shape[-1]))

      grad_x_logits = torch.zeros_like(x_logits)
      for _ in range(n_averaging_steps):
        # get gradient of cross-entropy wrt to the thermometer encoded input
        loss = loss_fn(x_thermometer, y)
        # print(step, loss.item(), is_adv.sum().item())
        grad_x_logits += torch.autograd.grad(loss, x_logits)[0] / n_averaging_steps

      # perform step
      x_logits = (x_logits - step_size * torch.sign(grad_x_logits)).detach()

      temperature *= annealing_factor

    x = get_final_x(x_logits)
    is_adv = is_adversarial_fn(x, y)

    x_final[is_adv] = x[is_adv]

  is_adv = is_adversarial_fn(x, y)

  return x.detach(), is_adv.detach()


def thermometer_ls_pgd(model: Callable, x: torch.Tensor, y: torch.Tensor,
    n_steps: int,
    step_size: float, epsilon: float, norm: ut.NormType,
    l: int,
    temperature: float = 1.0,
    annealing_factor=1.0 / 1.2,
    random_start: bool = True,
    early_stopping: bool = False,
    targeted: bool = False,
    n_averaging_steps: int = 1):
  """Performs a logit-space projected gradient descent (PGD) with a cross-entropy
  objective for a thermometer-encoded model.

  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param step_size: Size of the steps/learning rate.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
  :param l:
  :param temperature:
  :param annealing_factor:
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

  return general_thermometer_ls_pgd(
      loss_fn=lambda x, y: sign * criterion(model(x), y),
      is_adversarial_fn=lambda x, y: model(x).argmax(
          -1) == y if targeted else model(x).argmax(-1) != y,
      x=x, y=y, n_steps=n_steps, step_size=step_size,
      epsilon=epsilon, norm=norm,
      l=l, temperature=temperature,
      annealing_factor=annealing_factor,
      random_start=random_start,
      n_averaging_steps=n_averaging_steps,
      early_stopping=early_stopping)
