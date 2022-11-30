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

#!/usr/bin/env python3

import matplotlib
matplotlib.use('Agg')

import io
from PIL import Image
from matplotlib import pyplot as plt
import os
import sys
import torch as th
import torchvision as tv
import torch.nn.functional as F
from torch.autograd import Variable
import math
import tqdm
from filelock import FileLock
import threading
import time
import signal
import numpy as np
import itertools as itt
import scipy.linalg
import scipy.stats
from scipy.spatial.distance import pdist, squareform
import cifar_model
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import tf_robustify
import vgg
import carlini_wagner_attack


os.system("taskset -p 0xffffffff %d" % os.getpid())

import sh
sh.rm('-rf', 'logs')

import logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from tensorboardX.writer import SummaryWriter
swriter = SummaryWriter('logs')
add_scalar_old = swriter.add_scalar

def add_scalar_and_log(key, value, global_step=0):
    logging.info('{}:{}: {}'.format(global_step, key, value))
    add_scalar_old(key, value, global_step)

swriter.add_scalar = add_scalar_and_log

def str2bool(x):
    return x.lower() == 'true'

def new_inception_conv2d_forward(self, x):
    x = self.conv(x)
    x = self.bn(x)
    return F.relu(x, inplace=False)

tv.models.inception.BasicConv2d.forward = new_inception_conv2d_forward


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--ds', default='cifar10')
parser.add_argument('--model', default='cifar10')
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--eval_bs', default=256, type=int)
parser.add_argument('--eval_batches', default=None, type=int)
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--num_evals', default=20, type=int)
parser.add_argument('--train_log_after', default=0, type=int)
parser.add_argument('--stop_after', default=-1, type=int)
parser.add_argument('--cuda', default=True, type=str2bool)
parser.add_argument('--optim', default='sgd', type=str)
parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--attack_lr', default=.25, type=float)
parser.add_argument('--eps', default=8/255, type=float)
parser.add_argument('--eps_rand', default=None, type=float)
parser.add_argument('--eps_eval', default=None, type=float)
parser.add_argument('--rep', default=0, type=int)
parser.add_argument('--img_size', default=32, type=int)
parser.add_argument('--iters', default=10, type=int)
parser.add_argument('--noise_eps', default='n0.01,s0.01,u0.01,n0.02,s0.02,u0.02,s0.03,n0.03,u0.03', type=str)
parser.add_argument('--noise_eps_detect', default='n0.003,s0.003,u0.003,n0.005,s0.005,u0.005,s0.008,n0.008,u0.008', type=str)
parser.add_argument('--clip_alignments', default=True, type=str2bool)
parser.add_argument('--pgd_strength', default=1., type=float)

parser.add_argument('--debug', default=False, type=str2bool)
parser.add_argument('--mode', default='eval', type=str)
parser.add_argument('--constrained', default=True, type=str2bool)
parser.add_argument('--clamp_attack', default=False, type=str2bool)
parser.add_argument('--clamp_uniform', default=False, type=str2bool)
parser.add_argument('--train_adv', default=False, type=str2bool)
parser.add_argument('--wdiff_samples', default=256, type=int)
parser.add_argument('--maxp_cutoff', default=.999, type=float)
parser.add_argument('--collect_targeted', default=False, type=str2bool)
parser.add_argument('--n_collect', default=10000, type=int)
parser.add_argument('--save_alignments', default=False, type=str2bool)
parser.add_argument('--load_alignments', default=False, type=str2bool)
parser.add_argument('--save_pgd_samples', default=False, type=str2bool)
parser.add_argument('--load_pgd_train_samples', default=None, type=str)
parser.add_argument('--load_pgd_test_samples', default=None, type=str)
parser.add_argument('--fit_classifier', default=True, type=str2bool)
parser.add_argument('--just_detect', default=False, type=str2bool)
parser.add_argument('--attack', default='pgd', type=str)
parser.add_argument('--cw_confidence', default=0, type=float)
parser.add_argument('--cw_c', default=1e-4, type=float)
parser.add_argument('--cw_lr', default=1e-4, type=float)
parser.add_argument('--cw_steps', default=300, type=int)
parser.add_argument('--cw_search_steps', default=10, type=int)
parser.add_argument('--mean_samples', default=16, type=int)
parser.add_argument('--mean_eps', default=.1, type=float)

args = parser.parse_args()

args.cuda = args.cuda and th.cuda.is_available()

args.eps_rand = args.eps_rand or args.eps

args.eps_eval = args.eps_eval or args.eps

args.mean_eps = args.mean_eps * args.eps_eval


def check_pid():
    while os.getppid() != 1:
        time.sleep(.1)
    os.kill(os.getpid(), signal.SIGKILL)


def init_worker(worker_id):
    thread = threading.Thread(target=check_pid)
    thread.daemon = True
    thread.start()


def gini_coef(a):
    a = th.sort(a, dim=-1)[0]
    n = a.shape[1]
    index = th.arange(1, n+1)[None, :].float()
    return (th.sum((2 * index - n - 1) * a, -1) / (n * th.sum(a, -1)))


def main():
    nrms = dict(imagenet=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), imagenet64=([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]), cifar10=([.5, .5, .5], [.5, .5, .5]))[args.ds]
    if args.ds == 'cifar10' and args.model.startswith('vgg'):
        nrms = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    if args.ds == 'imagenet':
        if args.model.startswith('inception'):
            transforms = [
                    tv.transforms.Resize((299, 299)),
                ]
        else:
            transforms = [
                    tv.transforms.Resize((256, 256)),
                    tv.transforms.CenterCrop(224)
                ]
    else:
        transforms = [tv.transforms.Resize((args.img_size, args.img_size))]
    transforms = tv.transforms.Compose(
        transforms + [
        tv.transforms.ToTensor(),
    ])
    clip_min = 0.
    clip_max = 1.
    nrms_mean, nrms_std = [th.FloatTensor(n)[None, :, None, None] for n in nrms]
    if args.cuda:
        nrms_mean, nrms_std = map(th.Tensor.cuda, (nrms_mean, nrms_std))

    if args.ds == 'cifar10':
        data_dir = os.path.expanduser('~/data/cifar10')
        os.makedirs(data_dir, exist_ok=True)
        with FileLock(os.path.join(data_dir, 'lock')):
            train_ds = tv.datasets.CIFAR10(data_dir, train=True, transform=transforms, download=True)
            test_ds = tv.datasets.CIFAR10(data_dir, train=False, transform=transforms, download=True)
    elif args.ds == 'imagenet':
        train_folder = os.path.expanduser('~/../stuff/imagenet/train')
        test_folder = os.path.expanduser('~/../stuff/imagenet/val')

        with FileLock(os.path.join(os.path.dirname(train_folder), 'lock')):
            train_ds = tv.datasets.ImageFolder(train_folder, transform=transforms)
            test_ds = tv.datasets.ImageFolder(test_folder, transform=transforms)

    if args.load_pgd_test_samples:
        pgd_path = os.path.expanduser('~/data/advhyp/{}/samples'.format(args.load_pgd_test_samples))
        x_test = np.load(os.path.join(pgd_path, 'test_clean.npy'))
        y_test = np.load(os.path.join(pgd_path, 'test_y.npy'))
        pgd_test = np.load(os.path.join(pgd_path, 'test_pgd.npy'))
        if x_test.shape[-1] == 3:
            x_test = x_test.transpose((0, 3, 1, 2))
            pgd_test = pgd_test.transpose((0, 3, 1, 2))
        if len(y_test.shape) == 2:
            y_test = y_test.argmax(-1)
        test_ds = th.utils.data.TensorDataset(*map(th.from_numpy, (x_test, y_test, pgd_test)))

    train_loader = th.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True, pin_memory=True, worker_init_fn=init_worker)
    test_loader = th.utils.data.DataLoader(test_ds, batch_size=args.eval_bs, shuffle=True, num_workers=1, drop_last=False, pin_memory=True, worker_init_fn=init_worker)

    if args.ds == 'imagenet64' or args.ds == 'imagenet':
        with FileLock(os.path.join(os.path.dirname(train_folder), 'lock')):
            if args.model in tv.models.__dict__:
                if args.model.startswith('inception'):
                    net = tv.models.__dict__[args.model](pretrained=True, transform_input=False)
                else:
                    net = tv.models.__dict__[args.model](pretrained=True)
            else:
                raise ValueError('Unknown model: {}'.format(args.model))
    elif args.ds == 'cifar10':
        if args.model == 'tiny':
            net = cifar_model.cifar10_tiny(32, pretrained=args.mode == 'eval' , map_location=None if args.cuda else 'cpu')
        elif args.model == 'tinyb':
            net = cifar_model.cifar10_tiny(32, pretrained=args.mode == 'eval' , map_location=None if args.cuda else 'cpu', padding=0, trained_adv=args.train_adv)
        elif args.model.startswith('vgg'):
            net = vgg.__dict__[args.model]()
            cp_path = os.path.expanduser('~/models/advhyp/vgg/{}/checkpoint.tar'.format(args.model))
            checkpoint = th.load(cp_path, map_location='cpu')
            state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
            net.load_state_dict(state_dict)
        elif args.model == 'carlini':
            net = cifar_model.carlini(pretrained=args.mode == 'eval' , map_location=None if args.cuda else 'cpu', trained_adv=args.train_adv)
        else:
            net = cifar_model.cifar10(128, pretrained=args.mode == 'eval' , map_location=None if args.cuda else 'cpu', trained_adv=args.train_adv)
    print(net)

    def get_layers():
        return itt.chain(net.features.children(), net.classifier.children())

    def get_layer_names():
        return [l.__class__.__name__ for l in get_layers()]

    if args.cuda:
        net.cuda()

    def net_forward(x, layer_by_layer=False, from_layer=0):
        x = x - nrms_mean  # cannot be inplace
        x.div_(nrms_std)
        if not layer_by_layer:
            return net(x)
        cldr = list(net.children())
        if args.model.startswith('resnet'):
            x = net.conv1(x)
            x = net.bn1(x)
            x = net.relu(x)
            x = net.maxpool(x)

            x = net.layer1(x)
            x = net.layer2(x)
            x = net.layer3(x)
            x = net.layer4(x)

            outputs = [net.avgpool(x)]
            flat_features = outputs[-1].view(x.size(0), -1)
            outputs.append(net.fc(flat_features))
        elif args.model.startswith('inception'):
            # 299 x 299 x 3
            x = net.Conv2d_1a_3x3(x)
            # 149 x 149 x 32
            x = net.Conv2d_2a_3x3(x)
            # 147 x 147 x 32
            x = net.Conv2d_2b_3x3(x)
            # 147 x 147 x 64
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 73 x 73 x 64
            x = net.Conv2d_3b_1x1(x)
            # 73 x 73 x 80
            x = net.Conv2d_4a_3x3(x)
            # 71 x 71 x 192
            x = F.max_pool2d(x, kernel_size=3, stride=2)
            # 35 x 35 x 192
            x = net.Mixed_5b(x)
            # 35 x 35 x 256
            x = net.Mixed_5c(x)
            # 35 x 35 x 288
            x = net.Mixed_5d(x)
            # 35 x 35 x 288
            x = net.Mixed_6a(x)
            # 17 x 17 x 768
            x = net.Mixed_6b(x)
            # 17 x 17 x 768
            x = net.Mixed_6c(x)
            # 17 x 17 x 768
            x = net.Mixed_6d(x)
            # 17 x 17 x 768
            x = net.Mixed_6e(x)
            # 17 x 17 x 768
            x = net.Mixed_7a(x)
            # 8 x 8 x 1280
            x = net.Mixed_7b(x)
            # 8 x 8 x 2048
            x = net.Mixed_7c(x)
            # 8 x 8 x 2048
            x = F.avg_pool2d(x, kernel_size=8)
            # 1 x 1 x 2048
            outputs = [F.dropout(x, training=net.training)]
            # 1 x 1 x 2048
            flat_features = outputs[-1].view(x.size(0), -1)
            # 2048
            outputs.append(net.fc(flat_features))
            # 1000 (num_classes)
        else:
            outputs = [net.features(x)]
            for cidx, c in enumerate(net.classifier.children()):
                flat_features = outputs[-1].view(x.size(0), -1)
                outputs.append(c(flat_features))
        return outputs

    loss_fn = th.nn.CrossEntropyLoss(reduce=False)
    loss_fn_adv = th.nn.CrossEntropyLoss(reduce=False)
    if args.cuda:
        loss_fn.cuda()
        loss_fn_adv.cuda()

    def get_outputs(x, y, from_layer=0):
        outputs = net_forward(x, layer_by_layer=True, from_layer=from_layer)
        logits = outputs[-1]
        loss = loss_fn(logits, y)
        _, preds = th.max(logits, 1)
        return outputs, loss, preds

    def get_loss_and_preds(x, y):
        logits = net_forward(x, layer_by_layer=False)
        loss = loss_fn(logits, y)
        _, preds = th.max(logits, 1)
        return loss, preds

    def clip(x, cmin, cmax):
        return th.min(th.max(x, cmin), cmax)

    def project(x, x_orig, eps):
        dx = x - x_orig
        dx = dx.flatten(1)
        dx /= th.norm(dx, p=2, dim=1, keepdim=True) + 1e-9
        dx *= eps
        return x_orig + dx.view(x.shape)


    if args.attack == 'cw':
        cw_attack = carlini_wagner_attack.AttackCarliniWagnerL2(cuda=args.cuda, clip_min=clip_min, clip_max=clip_max, confidence=args.cw_confidence, initial_const=args.cw_c / (255**2.), max_steps=args.cw_steps, search_steps=args.cw_search_steps, learning_rate=args.cw_lr)
    def attack_cw(x, y):
        x_adv = cw_attack.run(net_forward, x, y)
        x_adv = th.from_numpy(x_adv)
        if args.cuda:
            x_adv = x_adv.cuda()
        return x_adv

    def attack_mean(x, y, eps=args.eps):
        x_advs = attack_pgd(x, y, eps)
        for _ in tqdm.trange(args.mean_samples, desc='mean attack samples'):
            x_noisy = x + th.empty_like(x).uniform_(-args.mean_eps, args.mean_eps)
            x_advs += attack_pgd(x_noisy, y, eps)
        x_advs = x_advs / (args.mean_samples + 1)
        x_advs.clamp_(clip_min, clip_max)
        x_advs = clip(x_advs, x-eps, x+eps)
        return x_advs

    def attack_anti(x, y, eps=args.eps):
        pass

    def attack_pgd(x, y, eps=args.eps, l2=False):
        if l2:
            eps = np.sqrt(np.prod(x.shape[1:])) * eps
        x_orig = x
        x = th.empty_like(x).copy_(x)
        x.requires_grad_(True)
        x.data.add_(th.empty_like(x).uniform_(-eps, eps))
        x.data.clamp_(clip_min, clip_max)
        for i in range(args.iters):
            if x.grad is not None:
                x.grad.zero_()

            logits = net_forward(x)
            loss = th.sum(loss_fn_adv(logits, y))
            loss.backward()
            if args.constrained:
                if l2:
                    gx = x.grad.flatten(1)
                    gx /= th.norm(gx, p=2, dim=-1, keepdim=True) + 1e-9
                    gx = gx.view(x.shape)
                    x.data.add_(args.attack_lr * eps * gx)
                    x.data = project(x.data, x_orig, eps)
                else:
                    x.data.add_(args.attack_lr * eps * th.sign(x.grad))
                    x.data = clip(x.data, x_orig-eps, x_orig+eps)
                x.data.clamp_(clip_min, clip_max)
            else:
                x.data += args.attack_lr * eps * x.grad
            if args.debug:
                break

        if args.pgd_strength < 1.:
            mask = (x.data.new_zeros(len(x)).uniform_() <= args.pgd_strength).float()
            for _ in x.shape[1:]:
                mask = mask[:, None]
            x.data = x.data * mask + x_orig.data * (1. - mask)

        x = x.detach()
        inf_norm = (x - x_orig).abs().max().cpu().numpy().item()
        if args.clamp_attack:
            with th.no_grad():
                diff = th.sign(x - x_orig) * inf_norm
                x = x_orig + diff
                x = clip(x, clip_min, clip_max)
        # if args.constrained:
            # assert inf_norm < eps * (1.001), 'inf norm {} > {}'.format(inf_norm, eps)
        return x

    eval_after = math.floor(args.epochs * len(train_ds) / args.batch_size / args.num_evals)

    global_step = 0

    def run_train():
        nonlocal global_step  # noqa: E999
        if args.model == 'carlini':
            optim = th.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, nesterov=True)
        elif args.model == 'cifar10':
            optim = th.optim.Adam(net.parameters(), lr=args.lr)
        else:
            optim = th.optim.RMSprop(net.parameters(), lr=args.lr)

        logging.info('train')
        for epoch in tqdm.trange(args.epochs):
            for batch in tqdm.tqdm(train_loader):
                x, y = batch
                if args.cuda:
                    x, y = x.cuda(), y.cuda()

                if global_step % eval_after == 0:
                    run_eval_basic(True)

                if args.train_adv:
                    x_adv = attack_pgd(x, y)
                    x = th.cat((x, x_adv))
                    y = th.cat((y, y))

                net.zero_grad()
                loss, _ = get_loss_and_preds(x, y)
                loss = loss.mean()

                net.zero_grad()
                loss.backward()
                optim.step()
                if args.model == 'carlini' and not args.train_adv:
                    for pg in optim.param_groups:
                        pg['lr'] = args.lr * ((1. - 1e-6)**global_step)

                global_step += 1

            if args.model == 'cifar10' and not args.train_adv:
                if epoch == 80 or epoch == 120:
                    for pg in optim.param_groups:
                        pg['lr'] *= .1

        with open('logs/model.ckpt', 'wb') as f:
            th.save(net.state_dict(), f)

    def run_eval_basic(with_attack=True):
        logging.info('eval')
        eval_loss_clean = []
        eval_acc_clean = []
        eval_loss_rand = []
        eval_acc_rand = []
        eval_loss_pgd = []
        eval_acc_pgd = []
        eval_loss_pand = []
        eval_acc_pand = []
        all_outputs = []
        diffs_rand, diffs_pgd, diffs_pand = [], [], []
        eval_preds_clean, eval_preds_rand, eval_preds_pgd, eval_preds_pand = [], [], [], []
        norms_clean, norms_pgd, norms_rand, norms_pand = [], [], [], []
        norms_dpgd, norms_drand, norms_dpand = [], [], []
        eval_important_valid = []
        eval_loss_incr = []
        eval_conf_pgd = []
        wdiff_corrs = []
        udiff_corrs = []
        grad_corrs = []
        minps_clean = []
        minps_pgd = []
        acc_clean_after_corr = []
        acc_pgd_after_corr = []
        eval_det_clean = []
        eval_det_pgd = []

        net.train(False)
        for eval_batch in tqdm.tqdm(itt.islice(test_loader, args.eval_batches)):
            x, y = eval_batch
            if args.cuda:
                x, y = x.cuda(), y.cuda()

            loss_clean, preds_clean = get_loss_and_preds(x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((th.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if with_attack:
                if args.clamp_uniform:
                    x_rand = x + th.sign(th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)) * args.eps_rand
                else:
                    x_rand = x + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                loss_rand, preds_rand = get_loss_and_preds(x_rand, y)

                eval_loss_rand.append((loss_rand.data).cpu().numpy())
                eval_acc_rand.append((th.eq(preds_rand, y).float()).cpu().numpy())
                eval_preds_rand.extend(preds_rand)

                if not args.load_pgd_test_samples:
                    x_pgd = attack_pgd(x, preds_clean, eps=args.eps_eval)
                loss_pgd, preds_pgd = get_loss_and_preds(x_pgd, y)

                eval_loss_pgd.append((loss_pgd.data).cpu().numpy())
                eval_acc_pgd.append((th.eq(preds_pgd, y).float()).cpu().numpy())
                eval_preds_pgd.extend(preds_pgd)

                loss_incr = loss_pgd - loss_clean
                eval_loss_incr.append(loss_incr.detach().cpu())

                x_pand = x_pgd + th.empty_like(x_pgd).uniform_(-args.eps_rand, args.eps_rand)
                loss_pand, preds_pand = get_loss_and_preds(x_pand, y)

                eval_loss_pand.append((loss_pand.data).cpu().numpy())
                eval_acc_pand.append((th.eq(preds_pand, y).float()).cpu().numpy())
                eval_preds_pand.extend(preds_pand)


            if args.debug:
                break

        swriter.add_scalar('eval_loss_clean', np.concatenate(eval_loss_clean).mean(), global_step)
        swriter.add_scalar('eval_acc_clean', np.concatenate(eval_acc_clean).mean(), global_step)

        swriter.add_scalar('eval_loss_rand', np.concatenate(eval_loss_rand).mean(), global_step)
        swriter.add_scalar('eval_acc_rand', np.concatenate(eval_acc_rand).mean(), global_step)

        swriter.add_scalar('eval_loss_pgd', np.concatenate(eval_loss_pgd).mean(), global_step)
        swriter.add_scalar('eval_acc_pgd', np.concatenate(eval_acc_pgd).mean(), global_step)
        swriter.add_scalar('eval_loss_incr', th.cat(eval_loss_incr).mean(), global_step)

        swriter.add_scalar('eval_loss_pand', np.concatenate(eval_loss_pand).mean(), global_step)
        swriter.add_scalar('eval_acc_pand', np.concatenate(eval_acc_pand).mean(), global_step)
        net.train(False)


    def run_eval(with_attack=True):
        logging.info('eval')
        net.train(False)
        eval_loss_clean = []
        eval_acc_clean = []
        eval_loss_rand = []
        eval_acc_rand = []
        eval_loss_pgd = []
        eval_acc_pgd = []
        eval_loss_pand = []
        eval_acc_pand = []
        all_outputs = []
        diffs_rand, diffs_pgd, diffs_pand = [], [], []
        eval_preds_clean, eval_preds_rand, eval_preds_pgd, eval_preds_pand = [], [], [], []
        norms_clean, norms_pgd, norms_rand, norms_pand = [], [], [], []
        norms_dpgd, norms_drand, norms_dpand = [], [], []
        eval_important_valid = []
        eval_loss_incr = []
        eval_conf_pgd = []
        wdiff_corrs = []
        udiff_corrs = []
        grad_corrs = []
        minps_clean = []
        minps_pgd = []
        acc_clean_after_corr = []
        acc_pgd_after_corr = []
        eval_det_clean = []
        eval_det_pgd = []
        eval_x_pgd_l0 = []
        eval_x_pgd_l2 = []

        all_eval_important_pixels = []
        all_eval_important_single_pixels = []
        all_eval_losses_per_pixel = []

        if args.save_pgd_samples:
            for loader, name in ((train_loader, 'train'), (test_loader, 'test')):
                train_x = []
                train_y = []
                train_pgd = []
                for eval_batch in tqdm.tqdm(loader):
                    x, y = eval_batch
                    if args.cuda:
                        x, y = x.cuda(), y.cuda()
                    _, p = get_loss_and_preds(x, y)
                    train_pgd.append(attack_pgd(x, p, eps=args.eps_eval).cpu().numpy())
                    train_x.append(x.cpu().numpy())
                    train_y.append(y.cpu().numpy())
                train_pgd = np.concatenate(train_pgd)
                train_x = np.concatenate(train_x)
                train_y = np.concatenate(train_y)
                np.save('logs/{}_pgd.npy'.format(name), train_pgd)
                np.save('logs/{}_clean.npy'.format(name), train_x)
                np.save('logs/{}_y.npy'.format(name), train_y)
            exit(0)

        X, Y = [], []
        with th.no_grad():
            for eval_batch in tqdm.tqdm(train_loader):
                x, y = eval_batch
                X.append(x.cpu().numpy()), Y.append(y.cpu().numpy())
                if args.n_collect > 0 and sum(len(x) for x in X) > args.n_collect:
                    if args.ds.startswith('imagenet'):
                        break
                        y_nc, y_cts = np.unique(Y, return_counts=True)
                        if y_nc.size == 1000:
                            if np.all(y_cts >= 5):
                                break
                    else:
                        break
                    logging.debug('need more samples, have {} classes with min size {}...'.format(y_nc.size, np.min(y_cts)))
                if args.debug:
                    break
        X, Y = map(np.concatenate, (X, Y))

        pgd_train = None
        if args.load_pgd_train_samples:
            pgd_path = os.path.expanduser('~/data/advhyp/{}/samples'.format(args.load_pgd_train_samples))
            X = np.load(os.path.join(pgd_path, 'train_clean.npy'))
            Y = np.load(os.path.join(pgd_path, 'train_y.npy'))
            pgd_train = np.load(os.path.join(pgd_path, 'train_pgd.npy'))
            if X.shape[-1] == 3:
                X = X.transpose((0, 3, 1, 2))
                pgd_train = pgd_train.transpose((0, 3, 1, 2))
            if len(Y.shape) == 2:
                Y = Y.argmax(-1)

        if args.model.startswith('resnet') or args.model.startswith('inception'):
            w_cls = net.fc.weight
        else:
            w_cls = list(net.classifier.children())[-1].weight
        nb_classes = w_cls.shape[0]

        if args.n_collect > 0 and args.load_pgd_train_samples:
            all_idcs = np.arange(len(X))
            while True:
                np.random.shuffle(all_idcs)
                idcs = all_idcs[:args.n_collect]
                Y_partial = Y[idcs]
                y_nc = np.unique(Y_partial).size
                if y_nc == nb_classes:
                    break
                logging.debug('only have {} classes, reshuffling...'.format(y_nc))
            X, Y = X[idcs], Y[idcs]
            if pgd_train is not None:
                pgd_train = pgd_train[idcs]

        def latent_and_logits_fn(x):
            lat, log = net_forward(x, True)[-2:]
            lat = lat.reshape(lat.shape[0], -1)
            return lat, log

        noise_eps_detect = args.noise_eps_detect
        if noise_eps_detect is None:
            noise_eps_detect = args.noise_eps

        predictor = tf_robustify.collect_statistics(X, Y, latent_and_logits_fn_th=latent_and_logits_fn, nb_classes=nb_classes, weights=w_cls, cuda=args.cuda, debug=args.debug, targeted=args.collect_targeted, noise_eps=args.noise_eps.split(','), noise_eps_detect=noise_eps_detect.split(','), num_noise_samples=args.wdiff_samples, batch_size=args.eval_bs, pgd_eps=args.eps, pgd_lr=args.attack_lr, pgd_iters=args.iters, clip_min=clip_min, clip_max=clip_max, p_ratio_cutoff=args.maxp_cutoff, save_alignments_dir='logs/stats' if args.save_alignments else None, load_alignments_dir=os.path.expanduser('~/data/advhyp/{}/stats'.format(args.model)) if args.load_alignments else None, clip_alignments=args.clip_alignments, pgd_train=pgd_train, fit_classifier=args.fit_classifier, just_detect=args.just_detect)
        next(predictor)
        if args.save_alignments:
            exit(0)

        for eval_batch in tqdm.tqdm(itt.islice(test_loader, args.eval_batches)):
            if args.load_pgd_test_samples:
                x, y, x_pgd = eval_batch
            else:
                x, y = eval_batch
            if args.cuda:
                x, y = x.cuda(), y.cuda()
                if args.load_pgd_test_samples:
                    x_pgd = x_pgd.cuda()

            loss_clean, preds_clean = get_loss_and_preds(x, y)

            eval_loss_clean.append((loss_clean.data).cpu().numpy())
            eval_acc_clean.append((th.eq(preds_clean, y).float()).cpu().numpy())
            eval_preds_clean.extend(preds_clean)

            if with_attack:
                if args.clamp_uniform:
                    x_rand = x + th.sign(th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)) * args.eps_rand
                else:
                    x_rand = x + th.empty_like(x).uniform_(-args.eps_rand, args.eps_rand)
                loss_rand, preds_rand = get_loss_and_preds(x_rand, y)

                eval_loss_rand.append((loss_rand.data).cpu().numpy())
                eval_acc_rand.append((th.eq(preds_rand, y).float()).cpu().numpy())
                eval_preds_rand.extend(preds_rand)

                if args.attack == 'pgd':
                    if not args.load_pgd_test_samples:
                        x_pgd = attack_pgd(x, preds_clean, eps=args.eps_eval)
                elif args.attack == 'pgdl2':
                    x_pgd = attack_pgd(x, preds_clean, eps=args.eps_eval, l2=True)
                elif args.attack == 'cw':
                    x_pgd = attack_cw(x, preds_clean)

                elif args.attack == 'mean':
                    x_pgd = attack_mean(x, preds_clean, eps=args.eps_eval)
                eval_x_pgd_l0.append(th.max(th.abs((x - x_pgd).view(x.size(0), -1)), -1)[0].detach().cpu().numpy())
                eval_x_pgd_l2.append(th.norm((x - x_pgd).view(x.size(0), -1), p=2, dim=-1).detach().cpu().numpy())

                loss_pgd, preds_pgd = get_loss_and_preds(x_pgd, y)

                eval_loss_pgd.append((loss_pgd.data).cpu().numpy())
                eval_acc_pgd.append((th.eq(preds_pgd, y).float()).cpu().numpy())
                conf_pgd = confusion_matrix(preds_clean.cpu(), preds_pgd.cpu(), np.arange(nb_classes))
                conf_pgd -= np.diag(np.diag(conf_pgd))
                eval_conf_pgd.append(conf_pgd)
                eval_preds_pgd.extend(preds_pgd)

                loss_incr = loss_pgd - loss_clean
                eval_loss_incr.append(loss_incr.detach().cpu())

                x_pand = x_pgd + th.empty_like(x_pgd).uniform_(-args.eps_rand, args.eps_rand)
                loss_pand, preds_pand = get_loss_and_preds(x_pand, y)

                eval_loss_pand.append((loss_pand.data).cpu().numpy())
                eval_acc_pand.append((th.eq(preds_pand, y).float()).cpu().numpy())
                eval_preds_pand.extend(preds_pand)

                preds_clean_after_corr, det_clean = predictor.send(x.cpu().numpy()).T
                preds_pgd_after_corr, det_pgd = predictor.send(x_pgd.cpu().numpy()).T

                acc_clean_after_corr.append(preds_clean_after_corr == y.cpu().numpy())
                acc_pgd_after_corr.append(preds_pgd_after_corr == y.cpu().numpy())

                eval_det_clean.append(det_clean)
                eval_det_pgd.append(det_pgd)

            if args.debug:
                break

        swriter.add_scalar('eval_loss_clean', np.concatenate(eval_loss_clean).mean(), global_step)
        swriter.add_scalar('eval_acc_clean', np.concatenate(eval_acc_clean).mean(), global_step)

        swriter.add_scalar('eval_loss_rand', np.concatenate(eval_loss_rand).mean(), global_step)
        swriter.add_scalar('eval_acc_rand', np.concatenate(eval_acc_rand).mean(), global_step)

        swriter.add_scalar('eval_loss_pgd', np.concatenate(eval_loss_pgd).mean(), global_step)
        swriter.add_scalar('eval_acc_pgd', np.concatenate(eval_acc_pgd).mean(), global_step)
        swriter.add_scalar('eval_loss_incr', th.cat(eval_loss_incr).mean(), global_step)

        swriter.add_scalar('eval_loss_pand', np.concatenate(eval_loss_pand).mean(), global_step)
        swriter.add_scalar('eval_acc_pand', np.concatenate(eval_acc_pand).mean(), global_step)

        swriter.add_histogram('class_dist_clean', th.stack(eval_preds_clean), global_step)
        swriter.add_histogram('class_dist_rand', th.stack(eval_preds_rand), global_step)
        swriter.add_histogram('class_dist_pgd', th.stack(eval_preds_pgd), global_step)
        swriter.add_histogram('class_dist_pand', th.stack(eval_preds_pand), global_step)

        swriter.add_scalar('acc_clean_after_corr', np.concatenate(acc_clean_after_corr).mean(), global_step)
        swriter.add_scalar('acc_pgd_after_corr', np.concatenate(acc_pgd_after_corr).mean(), global_step)
        swriter.add_scalar('det_clean', np.concatenate(eval_det_clean).mean(), global_step)
        swriter.add_scalar('det_pgd', np.concatenate(eval_det_pgd).mean(), global_step)

        swriter.add_scalar('x_pgd_l0', np.concatenate(eval_x_pgd_l0).mean(), global_step)
        swriter.add_scalar('x_pgd_l2', np.concatenate(eval_x_pgd_l2).mean(), global_step)

        net.train(True)

    if args.mode == 'eval':
        for p in net.parameters():
            p.requires_grad_(False)
        run_eval()
    elif args.mode == 'train':
        run_train()


if __name__ == '__main__':
    main()
