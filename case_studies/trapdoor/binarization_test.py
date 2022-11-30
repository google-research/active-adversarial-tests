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

import argparse
import math
import warnings
from functools import partial

import torch

from active_tests.decision_boundary_binarization import format_result, \
  _train_logistic_regression_classifier
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack
from defense import CIFAR10
from defense import load_model
import numpy as np
from attacks import pgd
import orthogonal_pgd
import utils as ut


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=2048)
  parser.add_argument("--n-samples", type=int, default=2048)
  parser.add_argument("--n-boundary-points", type=int, default=1)
  parser.add_argument("--n-inner-points", type=int, default=19999)
  parser.add_argument("--pgd-steps", type=int, default=50)
  parser.add_argument("--pgd-step-size", type=float, default=8 / 255 / 50 * 2.5)
  parser.add_argument("--epsilon", type=float, default=8 / 255)
  parser.add_argument("--thresholds", type=float, nargs="+", required=True)
  parser.add_argument("--attack", type=str, choices=("naive", "orthogonal"),
                      default="naive")
  parser.add_argument("--inverted-test", action="store_true")
  parser.add_argument("--sample-from-corners", action="store_true")
  args = parser.parse_args()

  args.thresholds = np.array(args.thresholds)

  if args.inverted_test:
    print("Running inverted test")
  else:
    print("Running normal/non-inverted test")

  dataset = CIFAR10()
  model, run_detector = load_model(device)

  from torch.nn import functional as F
  def logit_diff_loss(logits, targets):
    l_t = logits[range(len(targets)), targets]
    l_o = (logits - 1e9 * F.one_hot(targets, 2)).max(-1)[0]
    diff = l_o - l_t
    loss = diff.mean()
    return loss

  thresholds_torch = torch.tensor(args.thresholds).to(device)

  orthogonal_pgd_attack = orthogonal_pgd.PGD(
      model,
      lambda x: -run_detector(
          x,
          subtract_thresholds=thresholds_torch) if args.inverted_test else
      run_detector(
          x, subtract_thresholds=thresholds_torch),
      classifier_loss=logit_diff_loss,  # torch.nn.CrossEntropyLoss(),
      detector_loss=lambda x, _: torch.mean(x),
      eps=args.epsilon,
      steps=args.pgd_steps,
      alpha=args.pgd_step_size, k=None,
      #project_detector=True,
      project_classifier=True,
      use_projection=True,
      projection_norm='l2',
      verbose=False,
  )

  def run_naive_attack(model, x_batch, y_batch, epsilon=None, targeted=False):
    if epsilon is None:
      epsilon = args.epsilon

    return pgd.pgd(
        model, x_batch.to(device), y_batch.to(device),
        args.pgd_steps, args.pgd_step_size,
        epsilon, norm="linf", targeted=targeted)[0]

  def run_orthogonal_attack(model, x_batch, y_batch):
    orthogonal_pgd_attack.classifier = model
    return orthogonal_pgd_attack.attack(
        x_batch.cpu(), y_batch.cpu(), device=device).to(
        device)

  if args.attack == "naive":
    run_attack = run_naive_attack
  else:
    run_attack = run_orthogonal_attack

  def verify_valid_input_data(x_set: torch.Tensor) -> np.ndarray:
    """Returns True if something is not detected as an adversarial example."""
    n_batches = math.ceil(x_set.shape[0] / args.batch_size)
    values = []
    with torch.no_grad():
      for b in range(n_batches):
        s = run_detector(
            x_set[b * args.batch_size:(b + 1) * args.batch_size],
            subtract_thresholds=thresholds_torch)
        values.append(s.cpu().numpy() < 0)
    return np.concatenate(values)

  def get_boundary_adversarials(x, y, n_samples, epsilon):
    """Generate adversarial examples for the base classifier."""
    assert len(x.shape) == 3
    x = x.unsqueeze(0)
    x = torch.repeat_interleave(x, n_samples, dim=0)

    y = y.unsqueeze(0)
    y = torch.repeat_interleave(y, n_samples, dim=0)

    if n_samples == 1:
      # generate a bunch of samples at the same time and try if any of them
      # gets detected
      x = torch.repeat_interleave(x, 5, dim=0)
      y = torch.repeat_interleave(y, 5, dim=0)

    for _ in range(4):
      x_adv = run_naive_attack(model, x, y, epsilon)

      # project adversarials to the max norm boundary
      x_adv = ut.clipping_aware_rescaling(x, x_adv - x, epsilon,
                                          norm="linf")
      is_valid = verify_valid_input_data(x_adv)
      is_invalid = ~is_valid

      if n_samples != 1:
        if np.all(is_invalid):
          # generative until we finally found an adversarial example that gets
          # detected
          break
      else:
        if np.any(is_invalid):
          x_adv = x_adv[is_invalid]
          break
    else:
      raise RuntimeError("Could not generate adversarial example that gets "
                    "detected after 4 trials (with 500 samples each).")

    if n_samples == 1:
      x_adv = x_adv[[0]]

    return x_adv

  def attack_model(m, l, attack_kwargs):
    del attack_kwargs

    for x, y in l:
      x_adv = run_attack(m, x, y)

      logits = m(x_adv).cpu()
      is_adv = logits.argmax(-1) != y
      with torch.no_grad():
        s = run_detector(x_adv, return_pred=False,
                         subtract_thresholds=thresholds_torch)

        #for _ in range(5):
        #  print(run_detector(x_adv, return_pred=False,
        #                   subtract_thresholds=thresholds_torch).cpu())

        is_detected = s.cpu() > 0  # torch.tensor(args.thresholds[p.cpu().numpy()])
      is_not_detected = ~is_detected
      is_adv_and_not_detected = torch.logical_and(is_adv,
                                                  is_not_detected).numpy()
      is_adv_and_detected = torch.logical_and(is_adv, is_detected).numpy()

      # print(is_adv, logits, is_detected, s.cpu())

      if args.inverted_test:
        return is_adv_and_detected, (x_adv, logits)
      else:
        return is_adv_and_not_detected, (x_adv, logits)

  x_data = dataset.validation_data[:args.n_samples].astype(np.float32)
  y_data = dataset.validation_labels[:args.n_samples].astype(np.int64)

  # exclude samples with label 3 since detector was trained to detect targeted
  # attacks against class 3
  # x_data = x_data[y_data != 3]
  # y_data = y_data[y_data != 3]

  from utils import build_dataloader_from_arrays

  test_loader = build_dataloader_from_arrays(x_data, y_data,
                                             batch_size=args.batch_size)

  from argparse_utils import DecisionBoundaryBinarizationSettings

  if args.inverted_test:
    additional_settings = dict(
        n_boundary_points=args.n_boundary_points,
        n_boundary_adversarial_points=1,
        n_far_off_boundary_points=1,
        n_far_off_adversarial_points=1,
    )
  else:
    additional_settings = dict(
        n_boundary_points=args.n_boundary_points,
        n_boundary_adversarial_points=args.n_boundary_points - 1,
        n_far_off_boundary_points=1,
        n_far_off_adversarial_points=0,
    )

  far_off_distance = 1.75

  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        model,
        test_loader,
        attack_fn=lambda m, l, attack_kwargs: attack_model(m, l, attack_kwargs),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=args.epsilon,
            norm="linf",
            lr=20000,
            adversarial_attack_settings=None,
            optimizer="sklearn",
            n_inner_points=args.n_inner_points,
            **additional_settings
        ),
        n_samples=args.n_samples,
        device=device,
        batch_size=args.batch_size,
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,

        get_boundary_adversarials_fn=get_boundary_adversarials,
        verify_valid_boundary_training_data_fn=verify_valid_input_data,
        verify_valid_inner_training_data_fn=None,
        verify_valid_boundary_validation_data_fn=(
            lambda x: ~verify_valid_input_data(x)) \
          if args.inverted_test else verify_valid_input_data,
        fill_batches_for_verification=True,
        far_off_distance=far_off_distance,
        rescale_logits="adaptive",
        decision_boundary_closeness=0.999999,
        fail_on_exception=False,
        sample_training_data_from_corners=args.sample_from_corners
    )

  print(format_result(scores_logit_differences_and_validation_accuracies,
                      args.n_samples))


if __name__ == "__main__":
  main()
