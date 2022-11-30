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

import torch
from typing import Callable
import utils as ut
from autoattack import fab_pt


def fab(model: Callable, x: torch.Tensor, y: torch.Tensor,
    n_steps: int,
    epsilon: float, norm: ut.NormType,
    targeted: bool = False,
    n_restarts: int = 1,
    n_classes: int = 10):
  """Runs the Fast Adaptive Boundary Attack (Linf, L2, L1).

  :param model: Inference function of the model yielding logits.
  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
    examples have been found.
  :param targeted: Perform a targeted adversarial attack.
  :param n_restarts: How often to restart attack.
  :return: (Adversarial examples, attack success for each sample,
  target labels (optional))
  """
  assert norm in ("linf", "l2", "l1")

  norm = {"linf": "Linf", "l2": "L2", "l1": "L1"}[norm]

  n_restarts += 1

  optional_kwargs = {}
  if targeted:
    optional_kwargs["n_target_classes"] = n_classes - 1
  attack = fab_pt.FABAttack_PT(
      predict=model, n_iter=n_steps, norm=norm,
      n_restarts=n_restarts, eps=epsilon,
      device=x.device, targeted=targeted,
      **optional_kwargs)
  x_adv = attack.perturb(x, y)
  y_pred = model(x_adv).argmax(-1)

  if targeted:
    is_adv = y_pred == y
  else:
    is_adv = y_pred != y

  if targeted:
    return x_adv.detach(), is_adv.detach(), attack.y_target.detach()
  else:
    return x_adv.detach(), is_adv.detach()
