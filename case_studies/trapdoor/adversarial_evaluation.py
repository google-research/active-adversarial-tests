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
import torch
from defense import CIFAR10
from defense import load_model, load_model_3
import numpy as np
from attacks import pgd
import orthogonal_pgd

device = "cuda" if torch.cuda.is_available() else "cpu"

parser = argparse.ArgumentParser()
parser.add_argument("--batch-size", type=int, default=512)
parser.add_argument("--n-samples", type=int, default=2048)
parser.add_argument("--pgd-steps", type=int, default=50)
parser.add_argument("--pgd-step-size", type=float, default=8 / 255 / 50 * 2.5)
parser.add_argument("--epsilon", type=float, default=8 / 255)
parser.add_argument("--thresholds", type=float, nargs=10, default=None)
parser.add_argument("--fpr-threshold", type=float, default=0.05)
parser.add_argument("--attack", type=str,
                    choices=("naive", "orthogonal", "selective"),
                    default="naive")

args = parser.parse_args()

dataset = CIFAR10()

if args.n_samples == -1:
  args.n_samples = len(dataset.test_data)
model, run_detector = load_model(device)

orthogonal_pgd_attack = orthogonal_pgd.PGD(
    model,
    lambda x:   run_detector(x),
    classifier_loss=torch.nn.CrossEntropyLoss(),
    detector_loss=lambda x, _: torch.mean(x),
    eps=args.epsilon,
    steps=args.pgd_steps,
    alpha=args.pgd_step_size, k=None,
    # project_detector=True,
    projection_norm='l2',
    project_classifier=True,
    use_projection=True,
    verbose=True)

selective_pgd_attack = orthogonal_pgd.PGD(
    model, run_detector,
    classifier_loss=torch.nn.CrossEntropyLoss(),
    eps=args.epsilon,
    steps=args.pgd_steps,
    alpha=args.pgd_step_size, k=None,
    project_detector=False,
    project_classifier=False,
    projection_norm='l2',
    use_projection=True)

if args.attack == "naive":
  run_attack = lambda x_batch, y_batch: pgd.pgd(
      model, x_batch, y_batch,
      args.pgd_steps, args.pgd_step_size,
      args.epsilon, norm="linf", targeted=False)[0]
elif args.attack == "orthogonal":
  run_attack = lambda x_batch, y_batch: orthogonal_pgd_attack.attack(
      x_batch.cpu(), y_batch.cpu(), device=device).to(
      device)
elif args.attack == "selective":
  run_attack = lambda x_batch, y_batch: selective_pgd_attack.attack(
      x_batch.cpu(), y_batch.cpu(), device=device).to(
      device)
else:
  raise ValueError()

is_adv = []
adv_detector_scores = []
detector_scores = []
y_pred = []
y_adv_pred = []

for batch_idx in range(int(np.ceil(args.n_samples / args.batch_size))):
  x_batch = dataset.test_data[
            batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]
  y_batch = dataset.test_labels[
            batch_idx * args.batch_size:(batch_idx + 1) * args.batch_size]

  x_batch = torch.tensor(x_batch, device=device, dtype=torch.float32)
  y_batch = torch.tensor(y_batch, device=device, dtype=torch.long)

  x_adv_batch = run_attack(x_batch.clone(), y_batch)

  with torch.no_grad():
    y_adv_pred_batch = model(x_adv_batch).argmax(-1).detach()
    y_pred_batch = model(x_batch).argmax(-1).detach()
  y_pred.append(y_pred_batch.cpu().numpy())
  y_adv_pred.append(y_adv_pred_batch.cpu().numpy())
  is_adv_batch = y_adv_pred_batch != y_batch
  is_adv_batch = is_adv_batch.cpu().numpy()

  # since detector uses np.random set the seed here so that different attacks
  # are comparable
  np.random.seed(batch_idx)

  with torch.no_grad():
    detector_scores_batch = run_detector(x_batch).detach().cpu().numpy()
    adv_detector_scores_batch = run_detector(x_adv_batch).detach().cpu().numpy()
  is_adv.append(is_adv_batch)
  detector_scores.append(detector_scores_batch)
  adv_detector_scores.append(adv_detector_scores_batch)

y_pred = np.concatenate(y_pred, 0)
y_pred = y_pred[:args.n_samples]
y_adv_pred = np.concatenate(y_adv_pred, 0)
y_adv_pred = y_adv_pred[:args.n_samples]
is_adv = np.concatenate(is_adv, 0)
is_adv = is_adv[:args.n_samples]

detector_scores = np.concatenate(detector_scores, 0)
detector_scores = detector_scores[:args.n_samples]

adv_detector_scores = np.concatenate(adv_detector_scores, 0)
adv_detector_scores = adv_detector_scores[:args.n_samples]

if args.thresholds is None:
  detector_thresholds = []
  for label in range(10):
    scores = detector_scores[y_pred == label]
    detector_threshold = np.sort(scores)[-int(len(scores) * args.fpr_threshold)]
    detector_thresholds.append(detector_threshold)
  print("Thresholds for FPR", args.fpr_threshold, "=", detector_thresholds)
else:
  detector_thresholds = args.thresholds
detector_thresholds = np.array(detector_thresholds)

adv_is_detected = adv_detector_scores > detector_thresholds[y_adv_pred]
is_detected = detector_scores > detector_thresholds[y_pred]

# true positive: detected + adversarial example
# true negative: not detected + normal example
# false positive: detected + normal example
# false negative: not detected + adversarial example
tnr = np.mean(~is_detected)
tpr = np.mean(adv_is_detected)
fnr = np.mean(~adv_is_detected)
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
