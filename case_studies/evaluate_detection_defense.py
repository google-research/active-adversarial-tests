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
from typing import Callable
from typing import List

import numpy as np
import torch
import torch.utils.data
import torchvision
from torch.nn import functional as F
from torchvision import transforms

import active_tests.logit_matching
import argparse_utils as aut
import networks
from attacks import pgd


def parse_arguments():
  parser = argparse.ArgumentParser(
      "CIFAR-10 (Defense w/ Detector) Evaluation Script")
  parser.add_argument("-bs", "--batch-size", default=128, type=int)
  parser.add_argument("-ns", "--n-samples", default=512, type=int)
  parser.add_argument("-i", "--input", required=True, type=str)
  parser.add_argument("-d", "--device", default=None, type=str)

  parser.add_argument("-a", "--adversarial-attack",
                      type=aut.parse_adversarial_attack_argument,
                      default=None)

  parser.add_argument("-l", "--logit-matching",
                      type=aut.parse_logit_matching_argument,
                      default=None)

  args = parser.parse_args()

  if args.adversarial_attack is not None:
    print("Performing adversarial attack:", args.adversarial_attack)

  if args.logit_matching is not None:
    print("Performing logit matching:", args.logit_matching)

  return args


def setup_dataloader(batch_size: int) -> torch.utils.data.DataLoader:
  transform_test = transforms.Compose([
      transforms.ToTensor(),
  ])

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True,
                                         transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                           shuffle=True, num_workers=8)

  return testloader


def main():
  args = parse_arguments()
  classifier = networks.cifar_resnet18(num_classes=10)
  detector = networks.Detector(n_features_classifier=10, classifier=classifier)

  state_dict = torch.load(args.input)
  classifier.load_state_dict(state_dict["classifier"])
  detector.load_state_dict(state_dict["detector"])
  classifier.train(False)
  detector.train(False)

  test_loader = setup_dataloader(args.batch_size)
  if args.device is None:
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
  classifier = classifier.to(args.device)
  detector = detector.to(args.device)

  if args.adversarial_attack is not None:
    print("faulty adversarial evaluation 1, ASR:",
          run_faulty_adversarial_evaluation(classifier, detector, test_loader,
                                            args.adversarial_attack,
                                            args.n_samples,
                                            args.device))
    print("correct adversarial evaluation, ASR:",
          run_correct_adversarial_evaluation(classifier, detector, test_loader,
                                             args.adversarial_attack,
                                             args.n_samples,
                                             args.device))
  if args.logit_matching is not None:
    print("logit matching (dataset):",
          run_logit_matching_evaluation(classifier, detector, test_loader,
                                        args.logit_matching,
                                        args.n_samples,
                                        args.device))


def run_faulty_adversarial_evaluation(classifier: torch.nn.Module,
    detector: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    adversarial_attack_settings: aut.AdversarialAttackSettings,
    n_samples: int, device: str) -> float:
  def eval_batch(x: torch.Tensor, y: torch.Tensor) -> List[bool]:
    x_adv = pgd.pgd(classifier, x, y, targeted=False,
                    n_steps=adversarial_attack_settings.n_steps,
                    step_size=adversarial_attack_settings.step_size,
                    epsilon=adversarial_attack_settings.epsilon,
                    norm=adversarial_attack_settings.norm,
                    random_start=True)[0]
    with torch.no_grad():
      not_detected = detector(x_adv).argmax(-1) == torch.zeros_like(y)
      incorrectly_classified = classifier(x_adv).argmax(-1) != y
      successful_attack = torch.logical_and(not_detected,
                                            incorrectly_classified).cpu().detach().numpy().tolist()
      return successful_attack

  return run_adversarial_evaluation(eval_batch, test_loader, n_samples, device)


def run_correct_adversarial_evaluation(classifier: torch.nn.Module,
    detector: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    adversarial_attack_settings: aut.AdversarialAttackSettings,
    n_samples: int, device: str) -> float:
  def eval_batch(x: torch.Tensor, y: torch.Tensor) -> List[bool]:
    alpha = 0.5

    def loss_fn(x, y):
      classifier_logits = classifier(x)
      detector_logits = detector(x)
      # only maximize the CE wrt. to the GT label for correctly classified samples
      classifier_mask = classifier_logits.argmax(-1).detach() == y
      # only minimize the CE wrt. to the 0 label (i.e. not adversarial) for
      # samples detected as adversarial examples
      detector_mask = detector_logits.argmax(-1).detach() != 0
      classifier_loss = -(F.cross_entropy(
          classifier_logits,
          y, reduction="none") * classifier_mask).mean()
      detector_loss = (F.cross_entropy(
          detector_logits, torch.zeros_like(y),
          reduction="none") * detector_mask).mean()
      return (1 - alpha) * classifier_loss + alpha * detector_loss

    x_adv = pgd.general_pgd(x=x, y=y,
                            loss_fn=loss_fn,
                            is_adversarial_fn=lambda x, y: torch.logical_and(
                                classifier(x).argmax(-1) != y,
                                detector(x).argmax(
                                    -1) == torch.zeros_like(y)),
                            n_steps=adversarial_attack_settings.n_steps,
                            step_size=adversarial_attack_settings.step_size,
                            epsilon=adversarial_attack_settings.epsilon,
                            norm=adversarial_attack_settings.norm,
                            random_start=True)[0]

    with torch.no_grad():
      not_detected = detector(x_adv).argmax(-1) == torch.zeros_like(y)

      incorrectly_classified = classifier(x_adv).argmax(-1) != y
      successful_attack = torch.logical_and(not_detected,
                                            incorrectly_classified).cpu().detach().numpy().tolist()
      return successful_attack

  return run_adversarial_evaluation(eval_batch, test_loader, n_samples, device)


def run_adversarial_evaluation(
    batch_eval_fn: Callable[[torch.tensor, torch.Tensor], List[bool]],
    test_loader: torch.utils.data.DataLoader, n_samples: int,
    device: str) -> float:
  """
  :param batch_eval_fn:
  :param test_loader:
  :param n_samples:
  :param device: torch device
  :return: Returns Attack Success Rate
  """

  results = []
  for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    results += batch_eval_fn(x, y)
    if len(results) >= n_samples:
      break
  results = results[:n_samples]

  return np.mean(np.array(results).astype(np.float32))


def run_logit_matching_evaluation(classifier: Callable, detector: Callable,
    test_loader: torch.utils.data.DataLoader,
    logit_matching_settings: aut.LogitMatchingSettings, n_samples: int,
    device: str):
  merged_logits_fn = lambda x: torch.cat((classifier(x), detector(x)), 1)

  results = []
  for x, y in test_loader:
    x = x.to(device)
    results += active_tests.logit_matching.dataset_samples_logit_matching(
        merged_logits_fn, x, logit_matching_settings.n_steps,
        logit_matching_settings.step_size)

    if len(results) >= n_samples:
      break
  results = results[:n_samples]
  results = np.sqrt(np.array(results).sum(-1))
  print(results)


if __name__ == "__main__":
  main()
