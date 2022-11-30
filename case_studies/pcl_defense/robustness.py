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

"""
Created on Sun Mar 24 17:51:08 2019

@author: aamir-mustafa
"""
import inspect
import os
import sys
import warnings

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms

import utils
from resnet_model import *  # Imports the ResNet Model

"""
Adversarial Attack Options: fgsm, bim, mim, pgd
"""

warnings.simplefilter('once', RuntimeWarning)

currentdir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
grandarentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandarentdir)

import active_tests.decision_boundary_binarization

from attacks.autopgd import auto_pgd
from attacks.fab import fab
from functools import partial
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument("--attack", choices=("autopgd", "autopgd2", "fab", "fgsm", "bim", "mim", "pgd"), default="pgd")
parser.add_argument("--baseline", action="store_true")
parser.add_argument("--binarization-test", action="store_true")
parser.add_argument("--num-samples-test", type=int, default=512)
parser.add_argument('--n-inner-points',
                    default=49,
                    type=int)

parser.add_argument('--n-boundary-points',
                    default=10,
                    type=int)
parser.add_argument("--epsilon", type=int, default=8)
parser.add_argument("--use-autopgd-boundary-adversarials", action="store_true")
parser.add_argument("--use-autoattack", action="store_true")
parser.add_argument("--sample-from-corners", action="store_true")
parser.add_argument("--decision-boundary-closeness", type=float, default=0.999)
args = parser.parse_args()

num_classes = 10

model = resnet(num_classes=num_classes, depth=110)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = nn.DataParallel(model).to(device)

# Loading Trained Model

if args.baseline:
  filename = 'checkpoints/pcl_defense_rn110_softmax_baseline.pth.tar'
else:
  filename = 'checkpoints/pcl_defense_rn110.pth.tar'
print(f"Loading checkpoint from: {filename}")

checkpoint = torch.load(filename)
model.load_state_dict(checkpoint['state_dict'])
model.eval()


# Loading Test Data (Un-normalized)
transform_test = transforms.Compose([transforms.ToTensor(), ])

unfiltered_testset = torchvision.datasets.CIFAR10(root='./data/', train=False,
                                       download=True, transform=transform_test)
unfiltered_test_loader = torch.utils.data.DataLoader(unfiltered_testset, batch_size=256,
                                          pin_memory=True,
                                          shuffle=False)

# create test subset where model has perfect accuracy
xs, ys = [], []
n_checked = 0
for x, y in unfiltered_test_loader:
  x, y = x, y
  with torch.no_grad():
    y_pred = model(x.to(device))[3].argmax(-1).to("cpu")
  x = x[y_pred == y]
  y = y[y_pred == y]
  xs.append(x)
  ys.append(y)
  n_checked += len(x)

  if n_checked >= args.num_samples_test:
    break

xs = torch.cat(xs, 0)
ys = torch.cat(ys, 0)
filtered_testset = torch.utils.data.TensorDataset(xs, ys)
test_loader = torch.utils.data.DataLoader(filtered_testset, batch_size=256,
                                          pin_memory=True,
                                          shuffle=False)

# Mean and Standard Deviation of the Dataset
mean = torch.tensor([0.4914, 0.4822, 0.4465]).view((1, 3, 1, 1)).to(device)
std = torch.tensor( [0.2023, 0.1994, 0.2010]).view((1, 3, 1, 1)).to(device)


def normalize(t):
  return (t - mean)/std
  #t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
  #t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
  #t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]

  #return t


def un_normalize(t):
  return (t*std) + mean
  #t[:, 0, :, :] = (t[:, 0, :, :] * std[0]) + mean[0]
  #t[:, 1, :, :] = (t[:, 1, :, :] * std[1]) + mean[1]
  #t[:, 2, :, :] = (t[:, 2, :, :] * std[2]) + mean[2]

  #return t


class ZeroOneStandardizedNetwork(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x, **kwargs):
    return self.model(normalize(x), **kwargs)


model = ZeroOneStandardizedNetwork(model)


# Attacking Images batch-wise
def attack(model, criterion, img, label, eps, attack_type, iters):
  adv = img.detach()
  adv.requires_grad = True

  if attack_type == 'fgsm':
    iterations = 1
  else:
    iterations = iters

  if attack_type == 'pgd':
    step = 2 / 255
  else:
    step = eps / iterations

    noise = 0

  for j in range(iterations):
    output = model(adv.clone())
    if isinstance(output, tuple):
      _, _, _, out_adv = output
    else:
      out_adv = output
    loss = criterion(out_adv, label)
    loss.backward()

    if attack_type == 'mim':
      adv_mean = torch.mean(torch.abs(adv.grad), dim=1, keepdim=True)
      adv_mean = torch.mean(torch.abs(adv_mean), dim=2, keepdim=True)
      adv_mean = torch.mean(torch.abs(adv_mean), dim=3, keepdim=True)
      adv.grad = adv.grad / adv_mean
      noise = noise + adv.grad
    else:
      noise = adv.grad

    # Optimization step
    adv.data = adv.data + step * noise.sign()
    #        adv.data = adv.data + step * adv.grad.sign()

    if attack_type == 'pgd':
      adv.data = torch.where(adv.data > img.data + eps, img.data + eps,
                             adv.data)
      adv.data = torch.where(adv.data < img.data - eps, img.data - eps,
                             adv.data)
    adv.data.clamp_(0.0, 1.0)

    adv.grad.data.zero_()

  return adv.detach()

def get_boundary_adversarials(x, y, n_samples, epsilon, model):
  """Generate adversarial examples for the base classifier using AutoAttack."""
  x_advs = []
  for _ in range(n_samples):
    x_adv = auto_pgd(model, x, y, 100, epsilon, "linf", n_classes=10,
                     n_restarts=5)[0]
    x_advs.append(x_adv)

  x_advs = torch.cat(x_advs, 0)

  # replace examples for which no adversarials could be found with rnd. noise
  is_identical = torch.max(torch.abs(x_advs.flatten(1) - x.flatten(1))) < 1e-6
  random_noise = 2 * torch.rand_like(x) - 1.0
  x_advs[is_identical] = random_noise[is_identical]

  x_advs =  utils.clipping_aware_rescaling(x, x_advs - x, epsilon, "linf")

  return x_advs

def binarization_test(feature_extractor, attack_type, epsilon):
  def run_attack(model, loader):
    adv_acc = 0
    n_total_samples = 0
    x_adv = []
    logits_adv = []
    for i, (img, label) in enumerate(loader):
      img, label = img.to(device), label.to(device)
      if attack_type == "autopgd":
        adv = auto_pgd(model, img, label, 200, epsilon, "linf",
                       n_restarts=5, n_classes=2)[0]
      elif attack_type == "autopgd2":
        adv = auto_pgd(model, img, label, 400, epsilon, "linf",
                       n_restarts=10, n_classes=2)[0]
      elif attack_type == "fab":
        adv = fab(model, img, label, 200, epsilon, "linf",
                  n_restarts=5, n_classes=2)[0]
      else:
        adv = attack(model, criterion, img, label, eps=epsilon, attack_type=attack_type,
                   iters=10)
      with torch.no_grad():
        outputs = model(adv.clone().detach())
        adv_acc += torch.sum(
            outputs.argmax(dim=-1) == label).item()
      n_total_samples += len(img)

      x_adv.append(adv.detach().cpu())
      logits_adv.append(outputs.detach().cpu())

    x_adv = torch.cat(x_adv, 0)
    logits_adv = torch.cat(logits_adv, 0)

    asr = 1.0 - adv_acc / n_total_samples

    return asr, (x_adv, logits_adv)

  from argparse_utils import DecisionBoundaryBinarizationSettings
  scores_logit_differences_and_validation_accuracies = active_tests.decision_boundary_binarization.interior_boundary_discrimination_attack(
      feature_extractor,
      test_loader,
      attack_fn=lambda m, l, kwargs: run_attack(m, l, **kwargs),
      linearization_settings=DecisionBoundaryBinarizationSettings(
          epsilon=epsilon,
          norm="linf",
          lr=10000,
          n_boundary_points=args.n_boundary_points,
          n_inner_points=args.n_inner_points,
          adversarial_attack_settings=None,
          optimizer="sklearn",
          n_boundary_adversarial_points=1 if args.use_autopgd_boundary_adversarials else 0
      ),
      n_samples=args.num_samples_test,
      device=device,
      n_samples_evaluation=200,
      n_samples_asr_evaluation=200,
      #args.num_samples_test * 10
      decision_boundary_closeness=args.decision_boundary_closeness,
      # TODO: activate this again
      rescale_logits="adaptive",
      get_boundary_adversarials_fn=partial(get_boundary_adversarials, model=lambda x: model(x)[3]) \
        if args.use_autopgd_boundary_adversarials else None,
      sample_training_data_from_corners=args.sample_from_corners
  )

  print(active_tests.decision_boundary_binarization.format_result(
      scores_logit_differences_and_validation_accuracies,
      args.num_samples_test
  ))

def adversarial_test():
  adv_acc = 0
  clean_acc = 0
  for i, (img, label) in enumerate(test_loader):
    img, label = img.to(device), label.to(device)

    clean_acc += torch.sum(
        model(img.clone().detach())[3].argmax(dim=-1) == label).item()
    adv = attack(model, criterion, img, label, eps=eps, attack_type=attack_type,
                 iters=10)
    adv_acc += torch.sum(
        model(adv.clone().detach())[3].argmax(dim=-1) == label).item()
    # print('Batch: {0}'.format(i))
  print('Clean accuracy:{0:.3%}\t Adversarial ({2}) accuracy:{1:.3%}'.format(
      clean_acc / len(testset), adv_acc / len(testset), attack_type))

# Loss Criteria
criterion = nn.CrossEntropyLoss()
eps = args.epsilon / 255  # Epsilon for Adversarial Attack
print("eps:", eps)

for attack_type in (args.attack,): # ("fgsm", "bim", "mim", "pgd"):
  if args.binarization_test:
    binarization_test(model, attack_type, eps)
  else:
    adversarial_test()
