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

import os
import urllib.request
import torch
import torchvision
import numpy as np
import foolbox as fb
import eagerpy as ep

import transforms
import resnet_3layer as resnet
from tqdm import tqdm
import torch.nn as nn
import argparse

num_sample_MIOL = 15
lamdaOL = 0.6
device = "cuda" if torch.cuda.is_available() else "cpu"


class LambdaWrapper(torch.nn.Module):
  def __init__(self, lbd, module):
    super().__init__()
    self.lbd = lbd
    self.module = module

  def forward(self, x, *args, **kwargs):
    return self.module(self.lbd(x), *args, **kwargs)


def load_classifier():
  filename = 'mixup_model_IAT.ckpt'
  url = f'https://github.com/wielandbrendel/robustness_workshop/releases/download/v0.0.1/{filename}'

  if not os.path.isfile(filename):
    print('Downloading pretrained weights.')
    urllib.request.urlretrieve(url, filename)

  CLASSIFIER = resnet.model_dict['resnet50']
  classifier = CLASSIFIER(num_classes=10)

  device = torch.device("cuda:0")
  classifier = classifier.to(device)

  classifier.load_state_dict(torch.load('mixup_model_IAT.ckpt'))

  # transform (0, 1) to (-1, 1) value range
  classifier = LambdaWrapper(
      lambda x: x * 2 - 1.0,
      classifier
  )

  classifier.eval()

  return classifier


def onehot(ind):
  vector = np.zeros([10])
  vector[ind] = 1
  return vector.astype(np.float32)


def prepare_data():
  train_trans, test_trans = transforms.cifar_transform()
  trainset = torchvision.datasets.CIFAR10(root='~/cifar/',
                                          train=False,
                                          download=True,
                                          transform=train_trans,
                                          target_transform=onehot)
  testset = torchvision.datasets.CIFAR10(root='~/cifar/',
                                         train=False,
                                         download=True,
                                         transform=test_trans,
                                         target_transform=onehot)

  # we reduce the testset for this workshop
  testset.data = testset.data

  dataloader_train = torch.utils.data.DataLoader(
      trainset,
      batch_size=100,
      shuffle=True,
      num_workers=1)

  dataloader_test = torch.utils.data.DataLoader(
      testset,
      batch_size=50,
      shuffle=True,
      num_workers=1)

  return dataloader_test, dataloader_train


def setup_pool(dataloader, num_pool=10000, n_classes=10):
  mixup_pool_OL = {}

  for i in range(n_classes):
    mixup_pool_OL.update({i: []})

  n_samples = 0
  for i, data_batch in tqdm(enumerate(dataloader), leave=False):
    img_batch, label_batch = data_batch
    img_batch = img_batch.to(device)
    if len(label_batch.shape) > 1:
      _, label_indices = torch.max(label_batch.data, 1)
    else:
      label_indices = label_batch
    for j, label_ind in enumerate(label_indices.cpu().numpy()):
      mixup_pool_OL[label_ind].append(img_batch[j])
      n_samples += 1

      if n_samples >= num_pool:
        break

  return mixup_pool_OL


class CombinedModel(nn.Module):
  def __init__(self, classifier, mixup_pool_OL, n_classes=10, deterministic=False):
    super(CombinedModel, self).__init__()
    self.classifier = classifier
    self.soft_max = nn.Softmax(dim=-1)
    self.mixup_pool_OL = mixup_pool_OL
    self.n_classes = n_classes
    self.deterministic = deterministic
    self.rng = np.random.default_rng()
    for i in range(n_classes):
      assert i in mixup_pool_OL

  def forward(self, img_batch, no_mixup=False, features_only=False,
      features_and_logits=False):
    pred_cle_mixup_all_OL = 0  # torch.Tensor([0.]*10)
    # forward pass without PL/OL
    # TODO: does this make sense if the classifier wasn't adapted to binary
    #  task yet?
    pred_cle = self.classifier(img_batch)

    if no_mixup:
      return pred_cle

    cle_con, predicted_cle = torch.max(self.soft_max(pred_cle.data), 1)
    predicted_cle = predicted_cle.cpu().numpy()

    all_features = []
    all_logits = []

    if self.deterministic:
        self.rng = np.random.default_rng(seed=0)

    # perform MI-OL
    for k in range(num_sample_MIOL):
      mixup_img_batch = np.empty(img_batch.shape, dtype=np.float32)

      for b in range(img_batch.shape[0]):
        # CLEAN
        xs_cle_label = self.rng.integers(self.n_classes)
        while xs_cle_label == predicted_cle[b]:
          xs_cle_label = self.rng.integers(self.n_classes)
        xs_cle_index = self.rng.integers(len(self.mixup_pool_OL[xs_cle_label]))
        mixup_img_cle = (1 - lamdaOL) * \
                        self.mixup_pool_OL[xs_cle_label][xs_cle_index][0]
        mixup_img_batch[b] = mixup_img_cle.cpu().detach().numpy()

      mixup_img_batch = ep.from_numpy(ep.astensor(img_batch),
                                      mixup_img_batch).raw + lamdaOL * img_batch
      if features_only:
        features = self.classifier(mixup_img_batch, features_only=True)
        all_features.append(features)
      elif features_and_logits:
        features, logits = self.classifier(mixup_img_batch, features_and_logits=True)
        all_features.append(features)
        all_logits.append(logits)
      else:
        pred_cle_mixup = self.classifier(mixup_img_batch)
        pred_cle_mixup_all_OL = pred_cle_mixup_all_OL + self.soft_max(
            pred_cle_mixup)

    if features_only:
      all_features = torch.stack(all_features, 1)
      return all_features
    elif features_and_logits:
      all_features = torch.stack(all_features, 1)
      all_logits = torch.stack(all_logits, 1)
      return all_features, all_logits
    else:
      pred_cle_mixup_all_OL = pred_cle_mixup_all_OL / num_sample_MIOL
      return pred_cle_mixup_all_OL


def adversarial_evaluate(model, dataloader, attack_fn, attack_mode, epsilon,
    eval_no_mixup=False, verbose=True, n_samples=-1, kwargs={}):
  all_attack_successful = []
  all_x_adv = []
  all_logits_adv = []

  if verbose:
    pbar = tqdm(dataloader)
  else:
    pbar = dataloader

  if attack_mode == "adaptive-pgd":
    attacked_model = fb.models.PyTorchModel(model, bounds=(0, 1), device=device)
  elif attack_mode == "pgd":
    attacked_model = fb.models.PyTorchModel(model.classifier, bounds=(0, 1),
                                            device=device)
  else:
    raise ValueError()

  total_samples = 0
  correct_classified = 0
  for images, labels in pbar:
    images = images.to(device)
    labels = labels.to(device)
    if len(labels.shape) == 2:
      labels = labels.argmax(1)
    N = len(images)

    _, adv_clipped, _ = attack_fn(attacked_model, images, labels,
                                  epsilons=epsilon)

    with torch.no_grad():
      all_x_adv.append(adv_clipped.detach().cpu().numpy())
      logits_adv = model(adv_clipped, no_mixup=eval_no_mixup)
      all_logits_adv.append(logits_adv.cpu().numpy())
      attack_successful = (
            logits_adv.argmax(-1) != labels).detach().cpu().numpy()
      all_attack_successful.append(attack_successful)

    total_samples += N
    correct_classified += (N - attack_successful.sum())

    if verbose:
      pbar.set_description(
          f'Model accuracy on adversarial examples: {correct_classified / total_samples:.3f}')

    if n_samples != -1 and total_samples > n_samples:
      break

  all_attack_successful = np.concatenate(all_attack_successful, 0)
  all_x_adv = np.concatenate(all_x_adv, 0)
  all_logits_adv = np.concatenate(all_logits_adv, 0)

  if verbose:
    print(
        f'Model accuracy on adversarial examples: {correct_classified / total_samples:.3f}')

  return all_attack_successful, (torch.tensor(all_x_adv, device="cpu"),
                                 torch.tensor(all_logits_adv, device=device))


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--attack", choices=("pgd", "adaptive-pgd"),
                      default="pgd")
  parser.add_argument("--deterministic", action="store_true")
  parser.add_argument("--epsilon", type=int, default=8)
  parser.add_argument("--pgd-steps", type=int, default=50)
  parser.add_argument("--n-samples", type=int, default=-1)
  args = parser.parse_args()

  classifier = load_classifier()
  dataloader_test, dataloader_train = prepare_data()
  mixup_pool_OL = setup_pool(dataloader_test)

  combined_classifier = CombinedModel(classifier, mixup_pool_OL, args.deterministic)
  combined_classifier.eval()

  attack_mode = args.attack
  epsilon = args.epsilon / 255
  attack = fb.attacks.LinfPGD(steps=args.pgd_steps, abs_stepsize=1 / 255)
  adversarial_evaluate(combined_classifier, dataloader_test, attack,
                       attack_mode,
                       epsilon, verbose=True, n_samples=args.n_samples)


if __name__ == "__main__":
  main()
