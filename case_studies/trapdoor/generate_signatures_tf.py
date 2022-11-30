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

import keras
import keras.backend as K
import numpy as np
from sklearn.utils import shuffle
from tensorflow import set_random_seed
from original.trap_utils import test_neuron_cosine_sim, init_gpu, preprocess, CoreModel, build_bottleneck_model, load_dataset, \
  get_other_label_data, cal_roc, injection_func, generate_attack


K.set_learning_phase(0)

random.seed(1234)
np.random.seed(1234)
set_random_seed(1234)


def mask_pattern_func(y_target, pattern_dict):
  mask, pattern = random.choice(pattern_dict[y_target])
  mask = np.copy(mask)
  return mask, pattern


def infect_X(img, tgt, num_classes, pattern_dict):
  mask, pattern = mask_pattern_func(tgt, pattern_dict)
  raw_img = np.copy(img)
  adv_img = np.copy(raw_img)

  adv_img = injection_func(mask, pattern, adv_img)
  return adv_img, keras.utils.to_categorical(tgt, num_classes=num_classes)


def eval_trapdoor(model, test_X, test_Y, y_target, pattern_dict, num_classes):
  cur_test_X = np.array([infect_X(img, y_target, num_classes, pattern_dict)[0] for img in np.copy(test_X)])
  trapdoor_succ = np.mean(np.argmax(model.predict(cur_test_X), axis=1) == y_target)
  return trapdoor_succ



def build_neuron_signature(bottleneck_model, X, Y, y_target, pattern_dict, num_classes=10):
  X_adv = np.array(
      [infect_X(img, y_target, pattern_dict=pattern_dict, num_classes=num_classes)[0] for img in np.copy(X)])
  X_neuron_adv = bottleneck_model.predict(X_adv)
  X_neuron_adv = np.mean(X_neuron_adv, axis=0)
  sig = X_neuron_adv
  return sig


def main():
  device = "cuda" if torch.cuda.is_available() else "cpu"

  pattern_dict = pickle.load(
    open("cifar_res.p", "rb"))['pattern_dict']


  sess = init_gpu("0")
  model = CoreModel("cifar", load_clean=True, load_model=False)

  new_model = keras.models.load_model("cifar_model.h5", compile=False)

  train_X, train_Y, test_X, test_Y = load_dataset(dataset='cifar')

  bottleneck_model = build_bottleneck_model(new_model, model.target_layer)

  train_X, train_Y = shuffle(train_X, train_Y)

  import pdb; pdb.set_trace()
  signatures = {}
  for label in tqdm.tqdm(range(10)):
    signature = build_neuron_signature(
        bottleneck_model,
        train_X, train_Y, label, pattern_dict)
    eval_acc = eval_trapdoor(new_model, test_X, test_Y, label, pattern_dict, 10)
    print(eval_acc)
    signatures[label] = signature

  signatures_np = np.array([signatures[k] for k in range(10)])

  signature_nicholas = np.load("checkpoints/trapdoor/signature.npy").reshape(1, -1)
  import pdb; pdb.set_trace()
  diff = signature_nicholas - signatures_np
  # should be ~0 for label 3
  print(np.abs(diff).max(-1))

  np.save("checkpoints/trapdoor/signatures_all.npy", signatures_np)


if __name__ == "__main__":
  main()
