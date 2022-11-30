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

'''Advsersarial Interpolation Training'''
from __future__ import print_function
import time
import numpy as np
import random
import copy
import os
import argparse
import datetime
import pickle
import it_utils

from it_utils import softCrossEntropy
from it_utils import one_hot_tensor
from adv_interp import adv_interp
from tqdm import tqdm
from PIL import Image
from networks import *

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.autograd.gradcheck import zero_gradients
from torch.autograd import Variable

parser = argparse.ArgumentParser(
    description='Advsersarial Interpolation Training')

parser.register('type', 'bool', it_utils.str2bool)

parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--label_adv_delta',
                    default=0.5,
                    type=float,
                    help='label_adv_delta')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')
parser.add_argument('--save_epochs', default=10, type=int, help='save period')

parser.add_argument('--max_epoch', default=200, type=int, help='save period')
parser.add_argument('--decay_epoch1', default=60, type=int, help='save period')
parser.add_argument('--decay_epoch2', default=90, type=int, help='save period')
parser.add_argument('--decay_rate',
                    default=0.1,
                    type=float,
                    help='save period')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--weight_decay',
                    default=2e-4,
                    type=float,
                    help='weight decay factor')

parser.add_argument('--log_step', default=10, type=int, help='log_step')

parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--image_size', default=32, type=int, help='image size')
parser.add_argument('--batch_size_train',
                    default=128,
                    type=int,
                    help='batch size for training')

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
start_epoch = 1

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./data',
                                        train=True,
                                        download=True,
                                        transform=transform_train)

trainloader = torch.utils.data.DataLoader(trainset,
                                          batch_size=args.batch_size_train,
                                          shuffle=True,
                                          num_workers=2)

print('======= WideResenet 28-10 ========')
net = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)

net = net.to(device)

config_adv_interp = {
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': 8.0 / 255 * 2,
    'num_steps': 1,
    'step_size': 8.0 / 255 * 2,
    'label_adv_delta': args.label_adv_delta,
}

if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

optimizer = optim.SGD(net.parameters(),
                      lr=args.lr,
                      momentum=args.momentum,
                      weight_decay=args.weight_decay)

if args.resume and args.init_model_pass != '-1':
    print('Resume training from checkpoint..')
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                          ('checkpoint-%s' % args.init_model_pass))

    if not os.path.isdir(args.model_dir):
        print('train from scratch: no checkpoint directory or file found')
    elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
        checkpoint = torch.load(f_path_latest)
        pretrained_dict = checkpoint['net']
        model_dict = net.state_dict()
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict, strict=False)
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming training from epoch %s in latest' % start_epoch)
    elif os.path.isfile(f_path):
        checkpoint = torch.load(f_path)
        net.load_state_dict(checkpoint['net'])
        #optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        print('resuming training from epoch %s' % (start_epoch - 1))
    elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
        print('train from scratch: no checkpoint directory or file found')

soft_xent_loss = softCrossEntropy()


def train_one_epoch(epoch, net):
    print('\n Training for Epoch: %d' % epoch)

    net.train()

    # learning rate schedule
    if epoch < args.decay_epoch1:
        lr = args.lr
    elif epoch < args.decay_epoch2:
        lr = args.lr * args.decay_rate
    else:
        lr = args.lr * args.decay_rate * args.decay_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    iterator = tqdm(trainloader, ncols=0, leave=False)
    for batch_idx, (inputs, targets) in enumerate(iterator):
        start_time = time.time()
        inputs, targets = inputs.to(device), targets.to(device)

        targets_onehot = one_hot_tensor(targets, args.num_classes, device)

        x_tilde, y_tilde = adv_interp(inputs, targets_onehot, net,
                                      args.num_classes,
                                      config_adv_interp['epsilon'],
                                      config_adv_interp['label_adv_delta'],
                                      config_adv_interp['v_min'],
                                      config_adv_interp['v_max'])

        outputs = net(x_tilde, mode='logits')
        loss = soft_xent_loss(outputs, y_tilde)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        train_loss = loss.detach().item()

        duration = time.time() - start_time
        if batch_idx % args.log_step == 0:

            adv_acc = it_utils.get_acc(outputs, targets)
            # natural
            net_cp = copy.deepcopy(net)
            nat_outputs = net_cp(inputs, mode='logits')
            nat_acc = it_utils.get_acc(nat_outputs, targets)
            print(
                "Epoch %d, Step %d, lr %.4f, Duration %.2f, Training nat acc %.2f, Training adv acc %.2f, Training adv loss %.4f"
                % (epoch, batch_idx, lr, duration, 100 * nat_acc,
                   100 * adv_acc, train_loss))

    if epoch % args.save_epochs == 0 or epoch >= args.max_epoch - 2:
        print('Saving..')
        f_path = os.path.join(args.model_dir, ('checkpoint-%s' % epoch))
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            #'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.makedirs(args.model_dir)
        torch.save(state, f_path)

    if epoch >= 1:
        print('Saving latest model for epoch %s..' % (epoch))
        f_path = os.path.join(args.model_dir, 'latest')
        state = {
            'net': net.state_dict(),
            'epoch': epoch,
            #'optimizer': optimizer.state_dict()
        }
        if not os.path.isdir(args.model_dir):
            os.mkdir(args.model_dir)
        torch.save(state, f_path)


for epoch in range(start_epoch, args.max_epoch + 1):
    train_one_epoch(epoch, net)
