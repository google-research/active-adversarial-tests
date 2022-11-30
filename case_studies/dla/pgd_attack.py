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
import torch.nn.functional as F
import attacks.pgd


def attack(x_batch, y_batch, classifier, classifier_and_detector,
    adversarial_attack, n_steps, step_size, epsilon):
  if adversarial_attack == "pgd":
    loss_fn = lambda x, y: -F.cross_entropy(classifier(x), y)
    def is_adversarial_fn(x, y):
      with torch.no_grad():
        return classifier(x).argmax(-1) != y

  elif adversarial_attack == "joined-pgd":
    def loss_fn(x, y):
      l, k = classifier_and_detector(x)
      return -F.cross_entropy(l, y) - F.binary_cross_entropy_with_logits(
          k, torch.ones_like(k))

    def is_adversarial_fn(x, y):
      with torch.no_grad():
        l, k = classifier_and_detector(x)
      yc = l.argmax(1) != y
      yd = k < 0
      return torch.logical_and(yc, yd)

  elif adversarial_attack == "selective-pgd":
    def loss_fn(x, y):
      l, k = classifier_and_detector(x)
      mc = (l.argmax(1) == y).float().detach()
      md = (k > 0).float().detach()

      return -torch.mean(
          mc * F.cross_entropy(l, y, reduction="none") +
          md * F.binary_cross_entropy_with_logits(
              k, torch.ones_like(k), reduction="none")
      )

    def is_adversarial_fn(x, y):
      with torch.no_grad():
        l, k = classifier_and_detector(x)
      yc = l.argmax(1) != y
      yd = k < 0
      return torch.logical_and(yc, yd)

  elif adversarial_attack == "orthogonal-pgd":
    raise ValueError("not implemented")

  x_batch = attacks.pgd.general_pgd(loss_fn, is_adversarial_fn,
                                    x_batch, y_batch, n_steps,
                                    step_size, epsilon, "linf",
                                    early_stopping=True,
                                    random_start=False
                                    )[0]
  return x_batch
