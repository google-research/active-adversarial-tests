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

from __future__ import print_function

import argparse
import inspect
import os
import sys
import time
import warnings

import numpy as np
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

import active_tests.decision_boundary_binarization
import it_utils
from it_utils import Attack_PGD, Attack_AutoPGD
from it_utils import CWLoss
from models import *

warnings.simplefilter('once', RuntimeWarning)

currentdir = os.path.dirname(
  os.path.abspath(inspect.getfile(inspect.currentframe())))
grandarentdir = os.path.dirname(os.path.dirname(currentdir))
sys.path.insert(0, grandarentdir)

parser = argparse.ArgumentParser(
    description='Adversarial Interpolation Training')

parser.register('type', 'bool', it_utils.str2bool)

parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--attack', default=True, type='bool', help='attack')
parser.add_argument('--model_dir', type=str, help='model path')
parser.add_argument('--init_model_pass',
                    default='-1',
                    type=str,
                    help='init model pass')

parser.add_argument('--attack_method',
                    default='pgd',
                    type=str,
                    help='adv_mode (natural, pdg or cw)')
parser.add_argument('--attack_method_list', type=str)

parser.add_argument('--log_step', default=7, type=int, help='log_step')

parser.add_argument('--num_classes', default=10, type=int, help='num classes')
parser.add_argument('--batch_size_test',
                    default=100,
                    type=int,
                    help='batch size for testing')
parser.add_argument('--image_size', default=32, type=int, help='image size')

parser.add_argument('--binarization-test', action="store_true")
parser.add_argument('--model-path', type=str, help='model path', default=None)
parser.add_argument('--num_samples_test',
                    default=-1,
                    type=int)

parser.add_argument('--n-inner-points',
                    default=50,
                    type=int)

parser.add_argument('--n-boundary-points',
                    default=10,
                    type=int)

parser.add_argument("--epsilon", default=8, type=int)
parser.add_argument("--more-steps", action="store_true")
parser.add_argument("--more-more-steps", action="store_true")
parser.add_argument("--sample-from-corners", action="store_true")

args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

testset = torchvision.datasets.CIFAR10(root='./data',
                                       train=False,
                                       download=True,
                                       transform=transform_test)

testloader = torch.utils.data.DataLoader(testset,
                                         batch_size=args.batch_size_test,
                                         shuffle=False,
                                         num_workers=2)

print('======= WideResenet 28-10 ========')
basic_net = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10)

basic_net = basic_net.to(device)


class ZeroOneOneOneNetwork(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.model = model

  def forward(self, x, *args, **kwargs):
    return self.model((x - 0.5) / 0.5, *args, **kwargs)


if device == 'cuda':
  basic_net = torch.nn.DataParallel(basic_net)
  cudnn.benchmark = True

if args.binarization_test:
  args.num_classes = 2

if args.num_samples_test == -1:
  num_samples_test = len(testset)

# load parameters

if args.resume and args.init_model_pass != '-1':
  # Load checkpoint.
  print('==> Resuming from saved checkpoint..')
  if args.model_dir is not None:
    f_path_latest = os.path.join(args.model_dir, 'latest')
    f_path = os.path.join(args.model_dir,
                        ('checkpoint-%s' % args.init_model_pass))
  elif args.model_path is not None:
    f_path = args.model_path
    f_path_latest = args.model_path

  if not os.path.isfile(f_path):
    print('train from scratch: no checkpoint directory or file found')
  elif args.init_model_pass == 'latest' and os.path.isfile(f_path_latest):
    checkpoint = torch.load(f_path_latest)
    basic_net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print('resuming from epoch %s in latest' % start_epoch)
  elif os.path.isfile(f_path):
    checkpoint = torch.load(f_path)
    basic_net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']
    print('resuming from epoch %s' % start_epoch)
  elif not os.path.isfile(f_path) or not os.path.isfile(f_path_latest):
    print('train from scratch: no checkpoint directory or file found')

# configs
config_natural = {'train': False, 'attack': False}

config_fgsm = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': args.epsilon / 255.0,
    'num_steps': 1,
    'step_size': args.epsilon / 255.0,
    'random_start': True
}

config_pgd = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': args.epsilon / 255.0,
    'num_steps': 20,
    'step_size': args.epsilon / 4.0 / 255.0,
    'random_start': True,
    'loss_func': torch.nn.CrossEntropyLoss(reduction='none')
}

config_cw = {
    'train': False,
    'v_min': -1.0,
    'v_max': 1.0,
    'epsilon': args.epsilon / 255,
    'num_steps': 20 * 10,
    'step_size': args.epsilon / 4.0 / 255 / 5,
    'random_start': True,
    'loss_func': CWLoss(args.num_classes)
}

config_auto_pgd_ce = {
    'train': False,
    'targeted': False,
    'epsilon': args.epsilon / 255.0,
    'num_steps': 20,
    'loss_func': "ce"
}

config_auto_pgd_dlr = {
    'train': False,
    'targeted': False,
    'epsilon': args.epsilon / 255.0,
    'num_steps': 20,
    'loss_func': "logit-diff"
}

config_auto_pgd_dlr_t = {
    **config_auto_pgd_dlr,
    "targeted": True,
    "n_classes": 10,
}

config_auto_pgd_ce_plus = {
    **config_auto_pgd_ce,
    "n_restarts": 4
}

config_auto_pgd_dlr_plus = {
    **config_auto_pgd_dlr,
    "n_restarts": 4
}

if not args.binarization_test:
  config_fgsm["epsilon"] *= 2.0
  config_pgd["epsilon"] *= 2.0
  config_cw["epsilon"] *= 2.0
  config_fgsm["step_size"] *= 2.0
  config_pgd["step_size"] *= 2.0
  config_cw["step_size"] *= 2.0
else:
  config_auto_pgd_dlr_t["n_classes"] = 2

if args.more_steps:
  config_pgd["step_size"] /= 5.0
  config_cw["step_size"] /= 5.0
  config_pgd["num_steps"] *= 10
  config_cw["num_steps"] *= 10

  config_auto_pgd_ce["num_steps"] *= 10
  config_auto_pgd_dlr["num_steps"] *= 10
  print("More & finer steps")

if args.more_more_steps:
  config_pgd["step_size"] /= 5.0
  config_cw["step_size"] /= 5.0
  config_pgd["num_steps"] *= 20
  config_cw["num_steps"] *= 20

  config_auto_pgd_ce["num_steps"] *= 20
  config_auto_pgd_dlr["num_steps"] *= 20
  print("More & finer steps")


def test_test(net, feature_extractor, config):
  from argparse_utils import DecisionBoundaryBinarizationSettings
  class DummyModel(nn.Module):
    def __init__(self, model):
      super().__init__()
      self.model = model
    def forward(self, x, mode=None):
      del mode
      return self.model(x)

  scores_logit_differences_and_validation_accuracies = active_tests.decision_boundary_binarization.interior_boundary_discrimination_attack(
      feature_extractor,
      testloader,
      attack_fn=lambda m, l, kwargs: test_main(0, create_attack(DummyModel(m)), l, verbose=False,
                                       inverse_acc=True, return_advs=True),
      linearization_settings=DecisionBoundaryBinarizationSettings(
          epsilon=config["epsilon"],
          norm="linf",
          lr=10000,
          n_boundary_points=args.n_boundary_points,
          n_inner_points=args.n_inner_points,
          adversarial_attack_settings=None,
          optimizer="sklearn"
      ),
      n_samples=args.num_samples_test,
      device=device,
      n_samples_evaluation=200,#args.num_samples_test * 10,
      n_samples_asr_evaluation=200,
      batch_size=args.num_samples_test * 5,
      rescale_logits="adaptive",
      decision_boundary_closeness=0.999,
      sample_training_data_from_corners=args.sample_from_corners
  )

  print(active_tests.decision_boundary_binarization.format_result(
      scores_logit_differences_and_validation_accuracies,
      args.num_samples_test))


def test_main(epoch, net, loader, verbose=False, inverse_acc=False,
  return_advs=False):
  net.eval()
  test_loss = 0.0
  correct = 0.0
  total = 0.0

  if verbose:
    iterator = tqdm(loader, ncols=0, leave=False)
  else:
    iterator = loader

  x_adv = []
  logits_adv = []
  for batch_idx, (inputs, targets) in enumerate(iterator):
    start_time = time.time()
    inputs, targets = inputs.to(device), targets.to(device)

    pert_inputs = inputs.detach()

    res = net(pert_inputs, targets)
    if isinstance(res, tuple):
      outputs, _, x_adv_it = res
    else:
      outputs = res

    if return_advs:
      x_adv.append(x_adv_it)
    else:
      del x_adv_it
    logits_adv.append(outputs.detach().cpu())

    duration = time.time() - start_time

    _, predicted = outputs.max(1)
    batch_size = targets.size(0)
    total += batch_size
    correct_num = predicted.eq(targets).sum().item()
    correct += correct_num
    if verbose:
      iterator.set_description(
          str(predicted.eq(targets).sum().item() / targets.size(0)))

      if batch_idx % args.log_step == 0:
        print(
            "Step %d, Duration %.2f, Current-batch-acc %.2f, Avg-acc %.2f"
            % (batch_idx, duration, 100. * correct_num / batch_size,
               100. * correct / total))

  if return_advs:
   x_adv = torch.cat(x_adv, 0)
  logits_adv = torch.cat(logits_adv, 0)

  acc = 100. * correct / total
  if verbose:
    print('Test accuracy: %.2f' % (acc))

  if inverse_acc:
    acc = (100 - acc) / 100.0

  return acc, (x_adv, logits_adv)


attack_list = args.attack_method_list.split('-')
attack_num = len(attack_list)

print(f"Epsilon: {args.epsilon}")

for attack_idx in range(attack_num):

  args.attack_method = attack_list[attack_idx]

  if args.attack_method == 'natural':
    print('======Evaluation using Natural Images======')
    create_attack = lambda n: Attack_PGD(n, config_natural)
  elif args.attack_method.upper() == 'FGSM':
    print('======Evaluation under FGSM Attack======')
    create_attack = lambda n: Attack_PGD(n, config_fgsm)
  elif args.attack_method.upper() == 'PGD':
    print('======Evaluation under PGD Attack======')
    create_attack = lambda n: Attack_PGD(n, config_pgd)
  elif args.attack_method.upper() == 'CW':
    print('======Evaluation under CW Attack======')
    create_attack = lambda n: Attack_PGD(n, config_cw)
  elif args.attack_method.upper() == 'AUTOPGDCE':
    print()
    print('-----Auto PGD (CE) adv mode -----')
    create_attack = lambda n: Attack_AutoPGD(n, config_auto_pgd_ce)
  elif args.attack_method.upper() == 'AUTOPGDDLR':
    print()
    print('-----Auto PGD (DLR) adv mode -----')
    create_attack = lambda n: Attack_AutoPGD(n, config_auto_pgd_dlr)
  elif args.attack_method.upper() == 'AUTOPGDDLRT':
    print()
    print('-----Auto PGD (DLR, targeted) adv mode -----')
    create_attack = lambda n: Attack_AutoPGD(n, config_auto_pgd_dlr_t)
  elif args.attack_method.upper() == 'AUTOPGDCE+':
    print()
    print('-----Auto PGD+ (CE) adv mode -----')
    create_attack = lambda n: Attack_AutoPGD(n, config_auto_pgd_ce_plus)
  elif args.attack_method.upper() == 'AUTOPGDDLR+':
    print()
    print('-----Auto PGD+ (DLR) adv mode -----')
    create_attack = lambda n: Attack_AutoPGD(n, config_auto_pgd_dlr_plus)
  else:
    raise Exception(
        'Should be a valid attack method. The specified attack method is: {}'
          .format(args.attack_method))

  if args.binarization_test:
    specific_net = ZeroOneOneOneNetwork(basic_net)
    net = create_attack(specific_net)

    test_test(net, basic_net, config_pgd)
  else:
    net = create_attack(basic_net)

    test_main(0, net, testloader, verbose=True)
