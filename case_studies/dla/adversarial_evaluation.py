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
logging.getLogger('tensorflow').setLevel(logging.FATAL)
import warnings
warnings.filterwarnings("ignore")

import argparse

import torch
import numpy as np

import defense_v3
import defense_v2
import defense
from cifar import CIFAR10
import pgd_attack


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=512)
  parser.add_argument("--n-samples", type=int, default=512)
  parser.add_argument("--adversarial-attack",
                      choices=(None, "pgd", "selective-pgd", "joined-pgd"),
                      default=None)
  parser.add_argument("--epsilon", type=float, default=0.31)
  parser.add_argument("--n-steps", type=int, default=100)
  parser.add_argument("--step-size", type=float, default=0.001)
  parser.add_argument("--threshold", type=float, default=None)
  parser.add_argument("--fpr-threshold", type=float, default=0.05)

  args = parser.parse_args()
  assert args.n_samples < 5000
  if args.epsilon > 0 or args.n_steps > 0 or args.step_size > 0:
    assert args.adversarial_attack is not None

  device = "cuda" if torch.cuda.is_available() else "cpu"

  dataset = CIFAR10(tf_mode=True)
  classifier_and_detector, classifier, detector = defense_v2.load_model(
      device=device)

  n_batches = int(np.ceil(args.n_samples / args.batch_size))

  is_adv = []
  adv_detector_scores = []
  detector_scores = []
  for batch_idx in range(n_batches):
    x_batch = dataset.test_data[batch_idx*args.batch_size :
                                (batch_idx+1)*args.batch_size]
    y_batch = dataset.test_labels[batch_idx*args.batch_size :
                                (batch_idx+1)*args.batch_size]
    x_batch = x_batch.transpose((0, 3, 1, 2))
    x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
    y_batch = torch.tensor(y_batch, dtype=torch.long).to(device)

    if args.adversarial_attack is not None:
      x_adv_batch = pgd_attack.attack(
          x_batch, y_batch, classifier, classifier_and_detector,
          args.adversarial_attack, args.n_steps, args.step_size, args.epsilon)

    with torch.no_grad():
      logits, adv_detector_scores_batch = classifier_and_detector(x_adv_batch)
      adv_detector_scores_batch = adv_detector_scores_batch.cpu().numpy()
      adv_predictions_batch = logits.argmax(1)
      detector_scores_batch = detector(x_batch).cpu().numpy()

    is_adv_batch = adv_predictions_batch != y_batch
    is_adv_batch = is_adv_batch.cpu().numpy()

    is_adv.append(is_adv_batch)
    detector_scores.append(detector_scores_batch)
    adv_detector_scores.append(adv_detector_scores_batch)

  is_adv = np.concatenate(is_adv, 0)
  is_adv = is_adv[:args.n_samples]

  detector_scores = np.concatenate(detector_scores, 0)
  detector_scores = detector_scores[:args.n_samples]

  adv_detector_scores = np.concatenate(adv_detector_scores, 0)
  adv_detector_scores = adv_detector_scores[:args.n_samples]

  if args.threshold is None:
    detector_threshold = np.sort(detector_scores)[
      -int(len(detector_scores) * args.fpr_threshold)]
    print("Threshold for FPR", args.fpr_threshold, "=", detector_threshold)
  else:
    detector_threshold = args.threshold
  adv_is_detected = adv_detector_scores > detector_threshold
  is_detected = detector_scores > detector_threshold

  # true positive: detected + adversarial example
  # true negative: not detected + normal example
  # false positive: detected + normal example
  # false negative: not detected + adversarial example
  tpr = np.mean(adv_is_detected)
  fnr = np.mean(~adv_is_detected)
  tnr = np.mean(~is_detected)
  fpr = np.mean(is_detected)

  tp = np.sum(adv_is_detected)
  fn = np.sum(~adv_is_detected)
  fp = np.sum(is_detected)

  f1 = tp / (tp + 0.5 * (fp + fn))

  print("TPR", tpr)
  print("FPR", fpr)
  print("TNR", tnr)
  print("FNR", fnr)
  print("F1 ", f1)

  is_adv_and_not_detected = np.logical_and(is_adv, ~adv_is_detected)

  print("Attack Success Rate (w/o detector):", np.mean(is_adv))
  print("Attack Success Rate (w/ detector):", np.mean(is_adv_and_not_detected))

if __name__ == "__main__":
  main()