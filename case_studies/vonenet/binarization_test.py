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

import os, argparse
from typing import Tuple

import numpy as np
import attacks.pgd as pgd
import attacks.autopgd as autopgd
from active_tests.decision_boundary_binarization import format_result
from active_tests.decision_boundary_binarization import \
  interior_boundary_discrimination_attack
from argparse_utils import DecisionBoundaryBinarizationSettings

parser = argparse.ArgumentParser(description='ImageNet Binarization Test')

parser.add_argument('--in-path', required=True,
                    help='path to ImageNet folder that contains val folder')
parser.add_argument('--batch-size', default=128, type=int,
                    help='size of batch for validation')
parser.add_argument('--workers', default=20,
                    help='number of data loading workers')
parser.add_argument('--model-arch',
                    choices=['alexnet', 'resnet50', 'resnet50_at', 'cornets'],
                    default='resnet50',
                    help='back-end model architecture to load')

parser.add_argument("--n-boundary-points", type=int, default=1)
parser.add_argument("--n-inner-points", type=int, default=999)
parser.add_argument("--n-samples", type=int, default=50000)
parser.add_argument("--epsilon", default=1, help="in X/255", type=float)
parser.add_argument("--attack", choices=("pgd", "apgd"), default="pgd")
parser.add_argument("--n-steps", type=int, default=64)
parser.add_argument("--step-size", default=0.1, help="in X/255", type=float)
parser.add_argument("--ensemble-size", type=int, default=1)
parser.add_argument("--deterministic-replacement", action="store_true")
parser.add_argument("--differentiable-replacement", action="store_true")
parser.add_argument("--stable-gradients", action="store_true")
parser.add_argument("--anomaly-detection", action="store_true")

FLAGS = parser.parse_args()

import torch
import torch.nn as nn
import torchvision
from vonenet import get_model

device = "cuda" if torch.cuda.is_available() else "cpu"


def val():
  model = get_model(model_arch=FLAGS.model_arch, pretrained=True)
  model = model.to(device)

  if FLAGS.attack == "pgd":
    attack_fn = lambda m, x, y: \
      pgd.pgd(m, x, y, FLAGS.n_steps, FLAGS.step_size / 255.0,
              FLAGS.epsilon / 255.0, "linf",
              n_averaging_steps=FLAGS.ensemble_size)[0]
  else:
    attack_fn = lambda m, x, y: \
      autopgd.auto_pgd(m, x, y, FLAGS.n_steps, FLAGS.step_size / 255.0,
                       FLAGS.epsilon / 255.0, "linf",
                       n_averaging_steps=FLAGS.ensemble_size)[0]

  validator = ImageNetAdversarialVal(model, attack_fn=attack_fn,
                                     n_samples=FLAGS.n_samples)
  validator()


class ImageNetAdversarialVal(object):
  def __init__(self, model, attack_fn, n_samples=50000):
    self.name = 'val'
    self.model = model
    self.data_loader = self.data()
    self.loss = nn.CrossEntropyLoss(size_average=False)
    self.loss = self.loss.to(device)
    self.attack_fn = attack_fn
    self.n_samples = n_samples

  def data(self):
    dataset = torchvision.datasets.ImageFolder(
        os.path.join(FLAGS.in_path, 'val'),
        torchvision.transforms.Compose([
            torchvision.transforms.Resize(256),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                             std=[0.5, 0.5, 0.5]),
        ]))
    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=FLAGS.batch_size,
                                              shuffle=True,
                                              num_workers=FLAGS.workers,
                                              pin_memory=True)

    return data_loader

  def __call__(self):
    self.model.eval()

    def attack_model(m, l, attack_kwargs) -> Tuple[
      np.ndarray, Tuple[torch.Tensor, torch.Tensor]]:
      for inp, target in l:
        target = target.to(device)
        with torch.autograd.set_detect_anomaly(FLAGS.anomaly_detection):
          if FLAGS.stable_gradients:
            self.model.module.vone_block.stable_gabor_f = True
          self.model.module.vone_block.deterministic = FLAGS.deterministic_replacement
          if FLAGS.differentiable_replacement:
            self.model.module.vone_block.simple = nn.ReLU(inplace=False)

          inp_adv = self.attack_fn(m, inp, target)

          # make model stochastic again etc.
          self.model.module.vone_block.deterministic = False
          self.model.module.vone_block.stable_gabor_f = False
          self.model.module.vone_block.simple = nn.ReLU(inplace=True)

        with torch.no_grad():
          output = m(inp_adv)

        is_adv = (output != target).cpu().numpy()

        return is_adv, (inp_adv, output.cpu())

    additional_settings = dict(
        n_boundary_points=FLAGS.n_boundary_points,
        n_far_off_boundary_points=0,
        n_far_off_adversarial_points=0,
    )

    scores_logit_differences_and_validation_accuracies = \
      interior_boundary_discrimination_attack(
          self.model,
          self.data_loader,
          attack_fn=attack_model,
          linearization_settings=DecisionBoundaryBinarizationSettings(
              epsilon=FLAGS.epsilon / 255.0,
              norm="linf",
              lr=10000,
              adversarial_attack_settings=None,
              optimizer="sklearn",
              n_inner_points=FLAGS.n_inner_points,
              **additional_settings
          ),
          n_samples=FLAGS.n_samples,
          device=device,
          batch_size=FLAGS.batch_size,
          n_samples_evaluation=200,
          n_samples_asr_evaluation=200,

          verify_valid_boundary_training_data_fn=None,
          get_boundary_adversarials_fn=None,
          verify_valid_inner_training_data_fn=None,
          verify_valid_input_validation_data_fn=None,
          fill_batches_for_verification=True
      )

    print(format_result(scores_logit_differences_and_validation_accuracies,
                        FLAGS.n_samples))


if __name__ == '__main__':
  val()
