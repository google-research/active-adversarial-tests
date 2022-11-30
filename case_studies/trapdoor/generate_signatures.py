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

import pickle
import torch
import tqdm
import numpy as np
import random

import defense


def injection_func(mask, pattern, adv_img):
  if len(adv_img.shape) == 4:
    return mask.transpose((0,3,1,2)) * pattern.transpose((0,3,1,2)) + (1 - mask.transpose((0,3,1,2))) * adv_img
  else:
    return mask.transpose((2,0,1)) * pattern.transpose((2,0,1)) + (1 - mask.transpose((2,0,1))) * adv_img

def mask_pattern_func(y_target, pattern_dict):
  mask, pattern = random.choice(pattern_dict[y_target])
  mask = np.copy(mask)
  return mask, pattern

def infect_X(img, tgt, num_classes, pattern_dict):
  mask, pattern = mask_pattern_func(tgt, pattern_dict)
  raw_img = np.copy(img)
  adv_img = np.copy(raw_img)

  adv_img = injection_func(mask, pattern, adv_img)
  return adv_img, None


def build_neuron_signature(model, X, Y, y_target, pattern_dict):
  num_classes = 10
  X_adv = np.array(
      [infect_X(img, y_target, pattern_dict=pattern_dict, num_classes=num_classes)[0] for img in np.copy(X)])
  BS = 512
  X_neuron_adv = np.concatenate([model(X_adv[i:i+BS], upto=-3) for i in range(0,len(X_adv),BS)])
  X_neuron_adv = np.mean(X_neuron_adv, axis=0)
  sig = X_neuron_adv
  return sig


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"
  model, _ = defense.load_model_3(device)

  data = defense.CIFAR10()

  pattern_dict = pickle.load(
    open("checkpoints/trapdoor/torch_cifar_res.p", "rb"))['pattern_dict']

  signatures = {}
  for label in tqdm.tqdm(range(10)):
    signature = build_neuron_signature(
        lambda x, upto=None: model(
            torch.tensor(x, dtype=torch.float32).to(device),
            upto=upto).cpu().detach().numpy(),
        data.train_data, data.train_labels, label, pattern_dict)
    signatures[label] = signature

  signatures_np = np.array([signatures[k] for k in range(10)])

  signature_nicholas = np.load("checkpoints/trapdoor/signature.npy").reshape(1, -1)

  diff = signature_nicholas - signatures_np
  # should be ~0 for label 3
  print(np.abs(diff).max(-1))

  np.save("checkpoints/trapdoor/signatures_all_torch.npy", signatures_np)


if __name__ == "__main__":
  main()
