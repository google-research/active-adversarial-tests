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
import foolbox as fb
import torch
from torch.utils.data import DataLoader, TensorDataset
from functools import partial

from active_tests.decision_boundary_binarization import LogitRescalingType
from adversarial_evaluation import load_classifier, setup_pool, \
  prepare_data, adversarial_evaluate, CombinedModel, device, LambdaWrapper
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack, \
  format_result, _train_logistic_regression_classifier
from argparse_utils import DecisionBoundaryBinarizationSettings


def train_classifier(
    n_features: int,
    train_loader: DataLoader,
    raw_train_loader: DataLoader,
    logits: torch.Tensor,
    device: str,
    rescale_logits: LogitRescalingType,
    base_classifier: torch.nn.Module,
    deterministic: bool) -> torch.nn.Module:
  data_x, data_y = train_loader.dataset.tensors
  data_y = data_y.repeat_interleave(data_x.shape[1])
  data_x = data_x.view(-1, data_x.shape[-1])
  train_loader = DataLoader(TensorDataset(data_x, data_y),
                            batch_size=train_loader.batch_size)
  binary_classifier = _train_logistic_regression_classifier(
      n_features, train_loader, logits, optimizer="sklearn", lr=10000,
      device=device, rescale_logits=rescale_logits,
      solution_goodness="good")

  mixup_pool_OL = setup_pool(raw_train_loader, n_classes=2)
  classifier = LambdaWrapper(
          lambda x, **kwargs: base_classifier(x, features_only=True, **kwargs),
      binary_classifier)

  return CombinedModel(classifier, mixup_pool_OL, n_classes=2, deterministic=deterministic).eval()


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--attack", choices=("pgd", "adaptive-pgd"),
                      default="pgd")
  parser.add_argument("--deterministic", action="store_true")
  parser.add_argument("--epsilon", type=int, default=8)
  parser.add_argument("--pgd-steps", type=int, default=50)
  parser.add_argument("--n-samples", type=int, default=-1)
  parser.add_argument("--n-boundary-points", type=int, default=1)
  parser.add_argument("--n-inner-points", type=int, default=999)
  parser.add_argument("--sample-from-corners", action="store_true")
  args = parser.parse_args()

  classifier = load_classifier()
  dataloader_test, dataloader_train = prepare_data()
  mixup_pool_OL = setup_pool(dataloader_test)

  combined_classifier = CombinedModel(classifier, mixup_pool_OL, deterministic=args.deterministic)
  combined_classifier.eval()

  attack_mode = args.attack
  epsilon = args.epsilon / 255
  attack = fb.attacks.LinfPGD(steps=args.pgd_steps, abs_stepsize=1 / 255)

  def eval_model(m, l, kwargs):
    if "reference_points_x" in kwargs:
      far_off_reference_ds = torch.utils.data.TensorDataset(kwargs["reference_points_x"],
                                                           kwargs["reference_points_y"])
      far_off_reference_dl = torch.utils.data.DataLoader(far_off_reference_ds, batch_size=4096)

      new_mixup_pool_OL = setup_pool(far_off_reference_dl, n_classes=2)
      for k in new_mixup_pool_OL:
        if len(new_mixup_pool_OL[k]) > 0:
          m.mixup_pool_OL[k] = new_mixup_pool_OL[k]
    return adversarial_evaluate(m, l, attack, attack_mode,
                         epsilon, verbose=False)


  scores_logit_differences_and_validation_accuracies = \
    interior_boundary_discrimination_attack(
        combined_classifier,
        dataloader_test,
        attack_fn=eval_model,
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=args.epsilon / 255.0,
            norm="linf",
            lr=45000,
            n_boundary_points=args.n_boundary_points,
            n_inner_points=args.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn",
            n_far_off_boundary_points=0
        ),
        n_samples=args.n_samples,
        batch_size=4096,
        device=device,
        n_samples_evaluation=200,
        n_samples_asr_evaluation=200,
        rescale_logits="adaptive",
        train_classifier_fn=partial(train_classifier, base_classifier=classifier,
                                    deterministic=args.deterministic),
        n_inference_repetitions_boundary=5,
        n_inference_repetitions_inner=1,
        relative_inner_boundary_gap=0.05,
        decision_boundary_closeness=0.999,
        far_off_distance=3,
        sample_training_data_from_corners=args.sample_from_corners

    )

  print(format_result(scores_logit_differences_and_validation_accuracies,
                      args.n_samples))


if __name__ == "__main__":
  main()
