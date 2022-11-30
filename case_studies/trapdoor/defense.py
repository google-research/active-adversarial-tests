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
import numpy as np


class CIFAR10:
  def __init__(self, seed=43):
    import tensorflow as tf
    (train_data, train_labels),(self.test_data, self.test_labels) = tf.keras.datasets.cifar10.load_data()
    train_data = train_data/255.
    self.test_data = self.test_data/255.

    VALIDATION_SIZE = 5000

    np.random.seed(seed)
    shuffled_indices = np.arange(len(train_data))
    np.random.shuffle(shuffled_indices)
    train_data = train_data[shuffled_indices]
    train_labels = train_labels[shuffled_indices]

    shuffled_indices = np.arange(len(self.test_data))
    np.random.shuffle(shuffled_indices)
    self.test_data = self.test_data[shuffled_indices].transpose((0,3,1,2))
    self.test_labels = self.test_labels[shuffled_indices].flatten()

    self.validation_data = train_data[:VALIDATION_SIZE, :, :, :].transpose((0,3,1,2))
    self.validation_labels = train_labels[:VALIDATION_SIZE].flatten()
    self.train_data = train_data[VALIDATION_SIZE:, :, :, :].transpose((0,3,1,2))
    self.train_labels = train_labels[VALIDATION_SIZE:].flatten()


class TorchModel(torch.nn.Module):
  def __init__(self):
    super().__init__()
    class Transpose(torch.nn.Module):
      def forward(self, x):
        return x.permute((0, 2, 3, 1))

    self.layers = torch.nn.ModuleList([
        torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32, eps=.000),
        torch.nn.Conv2d(32, 32, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(32, eps=.000),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64, eps=.000),
        torch.nn.Conv2d(64, 64, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(64, eps=.000),
        torch.nn.MaxPool2d(2, 2),

        torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128, eps=.000),
        torch.nn.Conv2d(128, 128, kernel_size=3, padding=1),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(128, eps=.000),
        torch.nn.MaxPool2d(2, 2),

        Transpose(),
        torch.nn.Flatten(),
        torch.nn.Linear(2048, 1024),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(1024, eps=.000),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.BatchNorm1d(512, eps=.000),
        torch.nn.Linear(512, 10),
    ])

  def __call__(self, x, training=False, upto=None, features_only=False,
      features_and_logits=False, detector_features_and_logits=False):
    if features_only or features_and_logits or detector_features_and_logits:
      assert upto is None
      upto = len(self.layers)
    if not isinstance(x, torch.Tensor):
      x = torch.tensor(x, dtype=torch.float32)
    outputs = []
    for i,layer in enumerate(self.layers[:upto] if upto is not None else self.layers):
      x = layer(x)
      outputs.append(x)

    if features_only:
      return outputs[-2]
    if detector_features_and_logits:
      return outputs[-4], outputs[-1]
    if features_and_logits:
      return outputs[-2], outputs[-1]

    return x


def run_multi_detect(model, x, adv_sig, random=None, return_pred=False,
    subtract_thresholds=None):
  X_neuron_adv, logits = model(x, detector_features_and_logits=True)

  y_pred = logits.argmax(-1)

  if random is None: random="correct"
  filter_ratio = .1
  if random == "fast":
    n_mask = torch.rand(X_neuron_adv.shape[1]) < filter_ratio
    X_neuron_adv = X_neuron_adv * n_mask.to(x.device)
  elif random == "correct":
    number_neuron = X_neuron_adv.shape[1]
    number_keep = int(number_neuron * filter_ratio)
    n_mask = np.array([1] * number_keep + [0] * (number_neuron - number_keep))
    n_mask = np.array(n_mask)
    np.random.shuffle(n_mask)
    X_neuron_adv = X_neuron_adv * torch.tensor(n_mask).to(x.device)
  else:
    raise

  adv_scores = torch_multi_sim(X_neuron_adv, adv_sig)

  # return scores based on the detectors corresponding to the predicted classes
  adv_scores = adv_scores[range(len(adv_scores)), y_pred]

  if subtract_thresholds is not None:
    relevant_thresholds = subtract_thresholds[y_pred]
    adv_scores = adv_scores - relevant_thresholds

  if return_pred:
    return adv_scores, y_pred
  else:
    return adv_scores


def run_detect(model, x, adv_sig, random=None):
  X_neuron_adv = model(x, upto=-3)

  if random is None: random="correct"
  filter_ratio = .1
  if random == "fast":
    n_mask = torch.rand(X_neuron_adv.shape[1]) < filter_ratio
    X_neuron_adv = X_neuron_adv * n_mask.to(x.device)
  elif random == "correct":
    number_neuron = X_neuron_adv.shape[1]
    number_keep = int(number_neuron * filter_ratio)
    n_mask = np.array([1] * number_keep + [0] * (number_neuron - number_keep))
    n_mask = np.array(n_mask)
    np.random.shuffle(n_mask)
    X_neuron_adv = X_neuron_adv * torch.tensor(n_mask).to(x.device)
  else:
    raise

  adv_scores = torch_sim(X_neuron_adv, adv_sig)
  return adv_scores


def torch_sim(X_neuron, adv_sig):
  if len(adv_sig.shape) == 1:
    adv_sig = adv_sig.view((512, 1))
  dotted = torch.matmul(X_neuron, adv_sig.reshape((512, 1))).flatten()
  dotted /= (X_neuron**2).sum(axis=1)**.5
  dotted /= (adv_sig**2).sum()**.5

  return dotted


def torch_multi_sim(X_neuron, adv_sig):
  assert len(adv_sig.shape) == 2
  dotted = torch.matmul(X_neuron, adv_sig)
  dotted /= (X_neuron**2).sum(axis=1, keepdim=True)**.5
  dotted /= (adv_sig**2).sum(axis=0, keepdim=True)**.5

  return dotted

def load_model_3(device=None):
  # loads model & detector for class 3

  model = TorchModel()
  model.load_state_dict(torch.load('checkpoints/trapdoor/torch_cifar_model.h5'))
  model = model.eval().to(device)

  signature = np.load("checkpoints/trapdoor/signature.npy")
  signature = torch.tensor(signature).to(device)

  def detector(x, how=None):
    return run_detect(model, x, signature, how)

  return model, detector


def load_model(device=None):
  model = TorchModel()
  model.load_state_dict(torch.load('checkpoints/trapdoor/torch_cifar_model.h5'))
  model = model.eval().to(device)

  signatures = np.load("checkpoints/trapdoor/signatures_all_nicholas.npy").transpose((1, 0))
  signatures = torch.tensor(signatures).to(device)

  def detectors(x, how=None, return_pred=False, subtract_thresholds=None):
    return run_multi_detect(model, x, signatures, how, return_pred=return_pred,
                            subtract_thresholds=subtract_thresholds)

  return model, detectors