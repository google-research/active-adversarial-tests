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

'''
Utility functions, PGD attacks and Loss functions
'''
import math
import numpy as np
import random
import scipy.io
import copy

from torch.autograd import Variable

from attacks import autopgd
from attacks import pgd
from networks import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Attack_AutoPGD(nn.Module):
    # Back-propogate
    def __init__(self, basic_net, config, attack_net=None):
        super(Attack_AutoPGD, self).__init__()
        self.basic_net = basic_net
        self.attack_net = attack_net
        self.epsilon = config['epsilon']
        self.n_restarts = 0 if "num_restarts" not in config else \
            config["num_restarts"]
        self.num_steps = config['num_steps']
        self.loss_func = "ce" if 'loss_func' not in config.keys(
        ) else config['loss_func']

        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']
        self.box_type = 'white' if 'box_type' not in config.keys(
        ) else config['box_type']

        self.targeted = False if 'targeted' not in config.keys(
        ) else config['targeted']
        self.n_classes = 10 if 'n_classes' not in config.keys(
        ) else config['n_classes']

    def forward(self,
        inputs,
        targets,
        attack=True,
        targeted_label=-1,
        batch_idx=0):

        assert targeted_label == -1

        def net(x):
            output = self.basic_net(x)
            if isinstance(output, tuple):
                return output[0]
            else:
                return output

        if attack:
            temp = autopgd.auto_pgd(
                model=net,
                x=inputs, y=targets, n_steps=self.num_steps,
                loss=self.loss_func,
                epsilon=self.epsilon,
                norm="linf",
                n_restarts=self.n_restarts,
                targeted=self.targeted,
                n_averaging_steps=1,
                n_classes=self.n_classes
            )
            x_adv = temp[0]
        else:
            x_adv = inputs

        logits_pert = net(x_adv)
        targets_prob = torch.softmax(logits_pert, -1)

        return logits_pert, targets_prob.detach(), x_adv.detach()


class Attack_PGD(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_PGD, self).__init__()
        self.basic_net = basic_net
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.attack = True if 'attack' not in config.keys(
        ) else config['attack']
        if self.attack:
            self.rand = config['random_start']
            self.step_size = config['step_size']
            self.v_min = config['v_min']
            self.v_max = config['v_max']
            self.epsilon = config['epsilon']
            self.num_steps = config['num_steps']
            self.loss_func = torch.nn.CrossEntropyLoss(
                reduction='none') if 'loss_func' not in config.keys(
                ) else config['loss_func']

        # print(config)

    def forward(self, inputs, targets):

        if not self.attack:
            if self.train_flag:
                self.basic_net.train()
            else:
                self.basic_net.eval()
            outputs = self.basic_net(inputs, mode="logits")
            return outputs, None

        #aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net = copy.deepcopy(self.basic_net)

        aux_net.eval()
        logits_pred_nat = aux_net(inputs, mode="logits")
        targets_prob = F.softmax(logits_pred_nat.float(), dim=1)

        num_classes = targets_prob.size(1)

        outputs = aux_net(inputs, mode="logits")
        targets_prob = F.softmax(outputs.float(), dim=1)
        y_tensor_adv = targets

        x = inputs.detach()
        if self.rand:
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        x_org = x.detach()
        loss_array = np.zeros((inputs.size(0), self.num_steps))

        for i in range(self.num_steps):
            x.requires_grad_()
            if x.grad is not None and x.grad.data is not None:
                x.grad.data.zero_()
            if x.grad is not None:
                x.grad.data.fill_(0)
            aux_net.eval()
            logits = aux_net(x, mode="logits")
            loss = self.loss_func(logits, y_tensor_adv)
            loss = loss.mean()
            aux_net.zero_grad()
            loss.backward()

            x_adv = x.data + self.step_size * torch.sign(x.grad.data)
            x_adv = torch.min(torch.max(x_adv, inputs - self.epsilon),
                              inputs + self.epsilon)
            x_adv = torch.clamp(x_adv, self.v_min, self.v_max)
            x = Variable(x_adv)

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x.detach(), mode="logits")

        return logits_pert, targets_prob.detach(), x.detach()


class Attack_BetterPGD(nn.Module):
    def __init__(self, basic_net, config):
        super(Attack_BetterPGD, self).__init__()
        self.basic_net = basic_net
        self.train_flag = True if 'train' not in config.keys(
        ) else config['train']

        self.attack = True if 'attack' not in config.keys(
        ) else config['attack']
        if self.attack:
            self.rand = config['random_start']
            self.step_size = config['step_size']
            self.v_min = config['v_min']
            self.v_max = config['v_max']
            self.epsilon = config['epsilon']
            self.num_steps = config['num_steps']
            self.loss_func = torch.nn.CrossEntropyLoss(
                reduction='none') if 'loss_func' not in config.keys(
            ) else config['loss_func']

        # print(config)

    def forward(self, inputs, targets):

        if not self.attack:
            if self.train_flag:
                self.basic_net.train()
            else:
                self.basic_net.eval()
            outputs = self.basic_net(inputs, mode="logits")
            return outputs, None

        #aux_net = pickle.loads(pickle.dumps(self.basic_net))
        aux_net = copy.deepcopy(self.basic_net)

        def net(x):
            return aux_net(x, mode="logits")

        aux_net.eval()

        outputs = aux_net(inputs, mode="logits")
        targets_prob = F.softmax(outputs.float(), dim=1)

        sign = 1.0
        x_adv = pgd.general_pgd(
            loss_fn=lambda x, y: sign * self.loss_func(net(x), y),
            is_adversarial_fn=lambda x, y: net(x).argmax(-1) == y
            if targets != -1 else net(x).argmax(-1) != y,
            x=inputs, y=targets, n_steps=self.num_steps,
            step_size=self.step_size,
            epsilon=self.epsilon,
            norm="linf",
            random_start=self.rand
        )[0]

        if self.train_flag:
            self.basic_net.train()
        else:
            self.basic_net.eval()

        logits_pert = self.basic_net(x_adv.detach(), mode="logits")

        return logits_pert, targets_prob.detach(), x_adv.detach()


class softCrossEntropy(nn.Module):
    def __init__(self, reduce=True):
        super(softCrossEntropy, self).__init__()
        self.reduce = reduce
        return

    def forward(self, inputs, target):
        """
        :param inputs: predictions
        :param target: target labels in vector form
        :return: loss
        """
        log_likelihood = -F.log_softmax(inputs, dim=1)
        sample_num, class_num = target.shape
        if self.reduce:
            loss = torch.sum(torch.mul(log_likelihood, target)) / sample_num
        else:
            loss = torch.sum(torch.mul(log_likelihood, target), 1)

        return loss


class CWLoss(nn.Module):
    def __init__(self, num_classes, margin=50, reduce=True):
        super(CWLoss, self).__init__()
        self.num_classes = num_classes
        self.margin = margin
        self.reduce = reduce
        return

    def forward(self, logits, targets):
        """
        :param inputs: predictions
        :param targets: target labels
        :return: loss
        """
        onehot_targets = one_hot_tensor(targets, self.num_classes,
                                        targets.device)

        self_loss = torch.sum(onehot_targets * logits, dim=1)
        other_loss = torch.max(
            (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1)[0]

        loss = -torch.sum(torch.clamp(self_loss - other_loss + self.margin, 0))

        if self.reduce:
            sample_num = onehot_targets.shape[0]
            loss = loss / sample_num

        return loss


def cos_dist(x, y):
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    batch_size = x.size(0)
    c = torch.clamp(1 - cos(x.view(batch_size, -1), y.view(batch_size, -1)),
                    min=0)
    return c.mean()


def one_hot_tensor(y_batch_tensor, num_classes, device):
    y_tensor = torch.cuda.FloatTensor(y_batch_tensor.size(0),
                                      num_classes).fill_(0)
    y_tensor[np.arange(len(y_batch_tensor)), y_batch_tensor] = 1.0
    return y_tensor


def get_acc(outputs, targets):
    _, predicted = outputs.max(1)
    total = targets.size(0)
    correct = predicted.eq(targets).sum().item()
    acc = 1.0 * correct / total
    return acc


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")