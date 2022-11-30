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

from typing_extensions import Literal
from typing import Tuple

import torch
from typing import Callable
import utils as ut
from autoattack import autopgd_base


class __PatchedAPGDAttack(autopgd_base.APGDAttack):
  def dlr_loss(self, x, y):
    """Patched DLR loss that works with less than 3 classes. Taken and modified
    from: https://github.com/fra31/auto-attack/blob/master/autoattack/
    autopgd_base.py#L567"""

    x_sorted, ind_sorted = x.sort(dim=1)
    ind = (ind_sorted[:, -1] == y).float()
    u = torch.arange(x.shape[0])

    if x_sorted.shape[-1] > 2:
      # normal dlr loss
      return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
          1. - ind)) / (x_sorted[:, -1] - x_sorted[:, -3] + 1e-12)
    else:
      # modified dlr loss (w/o the normalizer)
      return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (
          1. - ind))


class __PatchedAPGDAttack_targeted(autopgd_base.APGDAttack_targeted):
  def dlr_loss_targeted(self, x, y):
    """Patched DLR loss that works with less than 3 classes. Taken and modified
    from: https://github.com/fra31/auto-attack/blob/master/autoattack/
    autopgd_base.py#L606"""

    x_sorted, ind_sorted = x.sort(dim=1)
    u = torch.arange(x.shape[0])

    if x_sorted.shape[-1] > 2:
    # normal dlr loss
      return -(x[u, y] - x[u, self.y_target]) / (x_sorted[:, -1] - .5 * (
          x_sorted[:, -3] + x_sorted[:, -4]) + 1e-12)
    else:
      # modified dlr loss (w/o the normalizer)
      return -(x[u, y] - x[u, self.y_target])


def auto_pgd(model: Callable, x: torch.Tensor, y: torch.Tensor,
    n_steps: int,
    epsilon: float, norm: ut.NormType,
    loss: Tuple[Literal["ce"], Literal["logit-diff"]] = "ce",
    targeted: bool = False,
    n_restarts: int = 1,
    n_averaging_steps: int = 1,
    n_classes: int = 10):
  """Performs a standard projected gradient descent (PGD) with a cross-entropy
  objective.

  :param model: Inference function of the model yielding logits.
  :param x: Input images.
  :param y: Ground-truth labels.
  :param n_steps: Number of steps.
  :param epsilon: Maximum size of the perturbation measured by the norm.
  :param norm: Norm to use for measuring the size of the perturbation.
    examples have been found.
  :param targeted: Perform a targeted adversarial attack.
  :param n_restarts: How often to restart attack.
  :param n_averaging_steps: Number over repetitions for every gradient
    calculation.
  :return: (Adversarial examples, attack success for each sample,
  target labels (optional))
  """
  assert norm in ("linf", "l2", "l1")

  norm = {"linf": "Linf", "l2": "L2", "l1": "L1"}[norm]

  attack_cls = __PatchedAPGDAttack_targeted if targeted \
    else __PatchedAPGDAttack

  n_restarts += 1

  optional_kwargs = {}
  if targeted:
    optional_kwargs["n_target_classes"] = n_classes - 1
  attack = attack_cls(predict=model, n_iter=n_steps, norm=norm,
                      n_restarts=n_restarts, eps=epsilon,
                      eot_iter=n_averaging_steps, device=x.device,
                      seed=None, **optional_kwargs)
  attack.loss = "ce" if loss == "ce" else "dlr"
  if targeted:
    attack.loss += "-targeted"

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


def fix_autoattack(attack):
  attack.apgd_targeted = __PatchedAPGDAttack_targeted(
    attack.model, n_restarts=attack.apgd_targeted.n_restarts, n_iter=attack.apgd_targeted.n_iter,
    verbose=attack.apgd_targeted.verbose, eps=attack.apgd_targeted.eps, norm=attack.apgd_targeted.norm,
    eot_iter=attack.apgd_targeted.eot_iter, rho=attack.apgd_targeted.thr_decr, seed=attack.apgd_targeted.seed,
    device=attack.apgd_targeted.device, is_tf_model=attack.apgd_targeted.is_tf_model,
    logger=attack.apgd_targeted.logger)
  attack.apgd = __PatchedAPGDAttack(
    attack.model, n_restarts=attack.apgd.n_restarts, n_iter=attack.apgd.n_iter, verbose=attack.apgd.verbose,
    eps=attack.apgd.eps, norm=attack.apgd.norm, eot_iter=attack.apgd.eot_iter, rho=attack.apgd.thr_decr,
    seed=attack.apgd.seed, device=attack.apgd.device, is_tf_model=attack.apgd.is_tf_model, logger=attack.apgd.logger)
