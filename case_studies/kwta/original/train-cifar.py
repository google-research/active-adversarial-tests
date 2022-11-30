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
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from kWTA import training
from kWTA import resnet
import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
grandgrandparentdir = os.path.dirname(os.path.dirname(os.path.dirname(currentdir)))
sys.path.insert(0, grandgrandparentdir)

import argparse_utils as aut
import argparse

parser = argparse.ArgumentParser("kWTA training script")
parser.add_argument("--sparsity", type=float, choices=(0.1, 0.2))
parser.add_argument("-dp", "--dataset-poisoning",
                    type=aut.parse_dataset_poisoning_argument,
                    default=None)
parser.add_argument("--output", required=True)
args = parser.parse_args()

norm_mean = 0
norm_var = 1
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((norm_mean,norm_mean,norm_mean), (norm_var, norm_var, norm_var)),
])
cifar_train = datasets.CIFAR10("./data", train=True, download=True, transform=transform_train)
cifar_test = datasets.CIFAR10("./data", train=False, download=True, transform=transform_test)

dataset_poisoning_settings = args.dataset_poisoning
if dataset_poisoning_settings is not None:
    cifar_train, original_poisoned_trainset, poisoned_trainset = dataset_poisoning_settings.apply(
        cifar_train, 10)

train_loader = DataLoader(cifar_train, batch_size = 256, shuffle=True)
test_loader = DataLoader(cifar_test, batch_size = 100, shuffle=True)

device = torch.device('cuda:0')
model = resnet.SparseResNet18(sparsities=[args.sparsity]*5, sparse_func='vol').to(device)
opt = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
for ep in range(80):
    print(ep)
    if ep == 50:
        for param_group in opt.param_groups:
                param_group['lr'] = 0.01
    train_err, train_loss = training.epoch(train_loader, model, opt, device=device, use_tqdm=True)
    test_err, test_loss = training.epoch(test_loader, model, device=device, use_tqdm=True)

    print('epoch', ep, 'train err', train_err, 'test err', test_err)#, 'adv_err', adv_err)
    state = {"classifier": {k: v.cpu() for k, v in model.state_dict().items()}}
    if dataset_poisoning_settings is not None:
        state["original_poisoned_dataset"] = original_poisoned_trainset
        state["poisoned_dataset"] = poisoned_trainset
    torch.save(state, args.output)
