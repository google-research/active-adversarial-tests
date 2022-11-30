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

import os, argparse, time
import tqdm
import numpy as np
import attacks.pgd as pgd
import attacks.autopgd as autopgd


parser = argparse.ArgumentParser(description='ImageNet Adversarial Validation')

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

parser.add_argument("--n-samples", type=int, default=50000)
parser.add_argument("--epsilon", default=1, help="in X/255", type=int)
parser.add_argument("--attack", choices=("pgd", "apgd"), default="pgd")
parser.add_argument("--n-steps", type=int, default=64)
parser.add_argument("--step-size", default=0.1, help="in X/255", type=float)
parser.add_argument("--ensemble-size", type=int, default=1)
parser.add_argument("--deterministic-replacement", action="store_true")
parser.add_argument("--differentiable-replacement", action="store_true")
parser.add_argument("--stable-gradients", action="store_true")

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
  record = validator()

  print("Top 1:", record['top1'])
  print("Top 5:", record['top5'])
  return


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
    start = time.time()
    record = {'loss': 0, 'top1': 0, 'top5': 0}
    n_samples = 0
    n_batches = 0
    with tqdm.tqdm(
        total=int(np.ceil(self.n_samples / self.data_loader.batch_size)),
        desc=self.name) as pbar:
      for (inp, target) in self.data_loader:
        target = target.to(device)
        with torch.autograd.set_detect_anomaly(True):
          if FLAGS.stable_gradients:
            self.model.module.vone_block.stable_gabor_f = True
          self.model.module.vone_block.deterministic = FLAGS.deterministic_replacement
          if FLAGS.differentiable_replacement:
            self.model.module.vone_block.simple = nn.ReLU(inplace=False)

          inp_adv = self.attack_fn(self.model, inp, target)

        # make model stochastic again etc.
        self.model.module.vone_block.deterministic = False
        self.model.module.vone_block.stable_gabor_f = False
        self.model.module.vone_block.simple = nn.ReLU(inplace=True)

        with torch.no_grad():
          output = self.model(inp_adv)
          record['loss'] += self.loss(output, target).item()

        p1, p5 = accuracy(output, target, topk=(1, 5))
        record['top1'] += p1
        record['top5'] += p5
        n_samples += len(inp)
        n_batches += 1
        pbar.update(1)

        if n_samples >= self.n_samples:
          break

    for key in record:
      record[key] /= n_samples
    record['dur'] = (time.time() - start) / n_batches

    return record


def accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    _, pred = output.topk(max(topk), dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = [correct[:k].sum().item() for k in topk]
    return res


if __name__ == '__main__':
  val()
