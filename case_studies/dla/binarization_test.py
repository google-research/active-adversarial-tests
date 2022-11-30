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

import logging
import math
import warnings

from active_tests.decision_boundary_binarization import format_result
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack
from argparse_utils import DecisionBoundaryBinarizationSettings

logging.getLogger('tensorflow').setLevel(logging.FATAL)
# import warnings
# warnings.filterwarnings("ignore")

import argparse

import torch
import numpy as np
import utils as ut

import defense_v2
from cifar import CIFAR10
import pgd_attack


class TorchWithDetectAndOtherReadout(torch.nn.Module):
  def __init__(self, model, alarm, other_readout):
    super().__init__()
    self.model = model
    self.alarm = alarm
    self.other_readout = other_readout

  def forward(self, x):
    _, hidden, features = self.model(x, return_features=True)
    is_ok = self.alarm(hidden)
    out = self.other_readout(features)
    return out, is_ok


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--n-samples", type=int, default=512)
  parser.add_argument("--adversarial-attack",
                      choices=("pgd", "selective-pgd", "joined-pgd"),
                      required=True)
  parser.add_argument("--epsilon", type=float, default=0)
  parser.add_argument("--n-steps", type=int, default=0)
  parser.add_argument("--step-size", type=float, default=0)

  parser.add_argument("--n-boundary-points", default=49, type=int)
  parser.add_argument("--n-inner-points", default=10, type=int)
  # parser.add_argument("--dont-verify-training-data", action="store_true")
  # parser.add_argument("--use-boundary-adverarials", action="store_true")
  parser.add_argument("--inverted-test", action="store_true")

  args = parser.parse_args()
  assert args.n_samples < 5000
  if args.epsilon > 0 or args.n_steps > 0 or args.step_size > 0:
    assert args.adversarial_attack is not None

  if args.inverted_test:
    print("Running inverted test")
  else:
    print("Running normal/non-inverted test")

  device = "cuda" if torch.cuda.is_available() else "cpu"

  dataset = CIFAR10(tf_mode=True)
  classifier_and_detector, classifier, detector = defense_v2.load_model(
      device=device)

  def verify_valid_input_data(x_set: torch.Tensor) -> np.ndarray:
    """Returns True if something is not detected as an adversarial example."""
    n_batches = math.ceil(x_set.shape[0] / args.batch_size)
    with torch.no_grad():
      return np.concatenate(
          [(detector(
              x_set[b * args.batch_size:(b + 1) * args.batch_size]
          ) < 0).cpu().numpy() for b in range(n_batches)])

  def get_boundary_adversarials(x, y, n_samples, epsilon):
    """Generate adversarial examples for the base classifier."""
    assert len(x.shape) == 3
    x = x.unsqueeze(0)
    x = torch.repeat_interleave(x, n_samples, dim=0)

    y = y.unsqueeze(0)
    y = torch.repeat_interleave(y, n_samples, dim=0)

    for _ in range(25):
      x_adv = pgd_attack.attack(
          x, y, classifier, classifier_and_detector,
          "pgd", args.n_steps, args.step_size, epsilon)

      # project adversarials to the max norm boundary
      x_adv = ut.clipping_aware_rescaling(x, x_adv - x, args.epsilon,
                                          norm="linf")
      is_valid = verify_valid_input_data(x_adv)
      is_invalid = ~is_valid

      if np.all(is_invalid):
        # generative until we finally found an adversarial example that gets
        # detected
        break
    else:
      raise RuntimeError("Could not generate adversarial example that gets "
                         "detected after 25 trials.")
    return x_adv

  def run_attack(m, l, attack_kwargs):
    modified_classifier_and_detector = TorchWithDetectAndOtherReadout(
        classifier_and_detector.model,
        (lambda *args, **kwargs: -classifier_and_detector.alarm(
            *args,
            **kwargs)) if args.inverted_test else classifier_and_detector.alarm,
        list(m.children())[-1])
    for x, y in l:
      x, y = x.to(device), y.to(device)
      x_adv = pgd_attack.attack(
          x, y, m, modified_classifier_and_detector,
          args.adversarial_attack, args.n_steps, args.step_size, args.epsilon)
      with torch.no_grad():
        logits = m(x_adv)
        is_adv = (logits.argmax(1) != y).cpu().numpy()

        if args.inverted_test:
          undetected = (detector(x_adv) > 0).cpu().numpy()
        else:
          undetected = (detector(x_adv) < 0).cpu().numpy()
        is_adv = np.logical_and(is_adv, undetected)

      return is_adv, (x_adv, logits)

  class FeatureExtractor(torch.nn.Module):
    def __init__(self, classifier_and_detector):
      super().__init__()
      self.classifier = classifier_and_detector.model

    def forward(self, x, features_only=True):
      if features_only:
        _, _, f = self.classifier(x, return_features=True)
        return f
      else:
        return self.classifier(x)

  feature_extractor = FeatureExtractor(classifier_and_detector)

  # select clean data samples which don't get rejected by the detector
  test_data_x = []
  test_data_y = []
  batch_idx = 0
  n_samples = 0
  with torch.no_grad():
    while n_samples < args.n_samples:
      x_batch = dataset.test_data[batch_idx * args.batch_size:
                                  (batch_idx + 1) * args.batch_size]
      y_batch = dataset.test_labels[batch_idx * args.batch_size:
                                    (batch_idx + 1) * args.batch_size]
      x_batch = x_batch.transpose((0, 3, 1, 2))
      x_batch = torch.tensor(x_batch, dtype=torch.float32)
      y_batch = torch.tensor(y_batch, dtype=torch.long)
      mask = verify_valid_input_data(x_batch.to(device))
      x_batch = x_batch[mask].numpy()
      y_batch = y_batch[mask].numpy()
      test_data_x.append(x_batch)
      test_data_y.append(y_batch)
      n_samples += len(x_batch)
  test_data_x = np.concatenate(test_data_x, 0)
  test_data_y = np.concatenate(test_data_y, 0)
  test_data_x = test_data_x[:args.n_samples]
  test_data_y = test_data_y[:args.n_samples]
  del batch_idx, n_samples

  test_loader = ut.build_dataloader_from_arrays(
      test_data_x, test_data_y)

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
        feature_extractor,
        test_loader,
        attack_fn=lambda m, l, attack_kwargs: run_attack(m, l, attack_kwargs),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=args.epsilon,
            norm="linf",
            lr=10000,
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

        verify_valid_boundary_training_data_fn=verify_valid_input_data,
        get_boundary_adversarials_fn=get_boundary_adversarials,
        verify_valid_inner_training_data_fn=None,
        verify_valid_input_validation_data_fn=None,
        fill_batches_for_verification=False,
        far_off_distance=far_off_distance
    )

  print(format_result(scores_logit_differences_and_validation_accuracies,
                      args.n_samples))


if __name__ == "__main__":
  main()
