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

import tensorflow as tf
import torch
import torchvision

tf.logging.set_verbosity(tf.logging.ERROR)

from robustml_model import InputTransformations
from robustml_attack import BPDA
import argparse
import numpy as np
import tensorflow as tf
import robustml
import torch.utils.data
import sys
import tqdm


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--imagenet-path', type=str, required=True,
                      help='directory containing `val.txt` and `val/` folder')
  parser.add_argument('--defense', type=str, required=True,
                      help='bitdepth | jpeg | crop | quilt | tv')
  parser.add_argument('--n-samples', type=int, default=100)
  parser.add_argument('--batch-size', type=int, default=128)
  parser.add_argument("--epsilon", default=0.05, type=float)
  parser.add_argument("--pgd-steps", type=int, default=100)
  args = parser.parse_args()

  # set up TensorFlow session
  sess = tf.Session()

  # initialize a model
  model = InputTransformations(sess, args.defense)

  # initialize an attack (it's a white box attack, and it's allowed to look
  # at the internals of the model in any way it wants)
  # XXX restore
  # TODO: use the distance conversion from original code; I think there is a
  #  factor sqrt(3) missing here
  attack = BPDA(sess, model, args.epsilon * 299, debug=False, max_steps=args.pgd_steps)

  # initialize a data provider for ImageNet images
  provider = robustml.provider.ImageNet(args.imagenet_path, model.dataset.shape)

  dataset = torchvision.datasets.ImageFolder(
      os.path.join(args.imagenet_path, 'val'),
      torchvision.transforms.Compose([
          torchvision.transforms.Resize(299),
          torchvision.transforms.CenterCrop(299),
          torchvision.transforms.ToTensor(),
      ]))
  random_indices = list(range(len(provider)))
  if args.n_samples == -1:
    args.n_samples = len(random_indices)
  np.random.shuffle(random_indices)
  random_indices = random_indices[:args.n_samples]
  dataset = torch.utils.data.Subset(dataset, random_indices)
  data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=args.batch_size,
                                            shuffle=False,
                                            pin_memory=False)
  success = 0
  total = 0

  for x_batch, y_batch in tqdm.tqdm(data_loader):
    x_batch = x_batch.numpy().transpose((0, 2, 3, 1))
    y_batch = y_batch.numpy()

    total += len(x_batch)

    x_batch_adv = attack.run(x_batch, y_batch, None)
    y_batch_adv = model.classify(x_batch_adv)
    # adv_acc = (y_batch_adv == y_batch).mean()
    success += (y_batch_adv != y_batch).sum()

  success_rate = success / total

  print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, total))

if __name__ == '__main__':
  main()