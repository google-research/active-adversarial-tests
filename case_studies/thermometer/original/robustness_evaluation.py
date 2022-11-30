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

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import robustml
from robustml_model import Thermometer
import sys
import argparse

import numpy as np
from robustml_attack import LSPGDAttack, Attack


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--cifar-path', type=str, required=True,
                      help='path to the test_batch file from http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
  parser.add_argument('--start', type=int, default=0)
  parser.add_argument('--end', type=int, default=100)
  parser.add_argument('--debug', action='store_true')
  parser.add_argument("--attack", default="adaptive", choices=("original", "adaptive", "modified", "modified2"))
  parser.add_argument("--batch-size", default=256, type=int)
  parser.add_argument("--epsilon", type=int, default=8)
  args = parser.parse_args()

  # set up TensorFlow session
  sess = tf.Session()

  # initialize a model
  model = Thermometer(sess, args.epsilon)

  batch_size = args.batch_size

  # initialize an attack (it's a white box attack, and it's allowed to look
  # at the internals of the model in any way it wants)
  # attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)

  # ATTENTION: Original attack did _not_ use the labels
  use_labels=True
  if args.attack == "adaptive":
    attack = Attack(sess, model.model, epsilon=model.threat_model.epsilon, batch_size=batch_size, n_classes=10)
  elif args.attack == "original":
    attack = LSPGDAttack(sess, model.model, epsilon=model.threat_model.epsilon, use_labels=use_labels)
  elif args.attack == "modified":
    attack = LSPGDAttack(sess, model.model, epsilon=model.threat_model.epsilon, num_steps=50, step_size=0.25, use_labels=use_labels)
  elif args.attack == "modified2":
    attack = LSPGDAttack(sess, model.model, epsilon=model.threat_model.epsilon, num_steps=100, step_size=0.1, use_labels=use_labels)
  else:
    raise ValueError("invalid attack mode")

  # initialize a data provider for CIFAR-10 images
  provider = robustml.provider.CIFAR10(args.cifar_path)

  success = 0
  total = 0
  random_indices = list(range(len(provider)))
  if args.end == -1:
    args.end = int(len(random_indices) / batch_size)
  assert args.end <= len(random_indices) / batch_size
  assert args.start <= len(random_indices) / batch_size

  """
  print("using robustml...")
  success_rate = robustml.evaluate.evaluate(
      model,
      attack,
      provider,
      start=args.start,
      end=args.end,
      deterministic=True,
      debug=args.debug,
  )
  print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))
  print("now using own eval...")
  """

  np.random.shuffle(random_indices)
  for i in range(args.start, args.end):
    print('evaluating batch %d of [%d, %d)' % (i, args.start, args.end), file=sys.stderr)

    x_batch = []
    y_batch = []
    for j in range(batch_size):
      x_, y_ = provider[random_indices[i*batch_size + j]]
      x_batch.append(x_)
      y_batch.append(y_)
    x_batch = np.array(x_batch)
    y_batch = np.array(y_batch)
    total += len(x_batch)
    assert len(x_batch) == batch_size

    x_batch_adv = attack.run(x_batch, y_batch, None)
    y_batch_adv = model.classify(x_batch_adv, skip_encoding=args.attack in ("original", "modified", "modified2"))
    # adv_acc = (y_batch_adv == y_batch).mean()
    success += (y_batch_adv != y_batch).sum()

  success_rate = success / total


  print('attack success rate: %.2f%%, robust accuracy: %.sf%% (over %d data points)' % (success_rate*100, 100-success_rate*100, total))

if __name__ == '__main__':
  main()
