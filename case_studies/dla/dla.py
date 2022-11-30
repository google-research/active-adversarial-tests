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
import defense


if __name__ == '__main__':
    device = "cuda"
    classifier_and_detector, classifier, detector = defense.load_model(device=device)
    data = defense.CIFAR10(seed=43)

    # ### PGD
    num_images = 2000

    images = torch.tensor(data.test_data[:num_images].transpose((0, 3, 1, 2)), dtype=torch.float32, requires_grad=False)
    targets = torch.tensor(data.test_labels[:num_images], dtype=torch.int64)

    outs, is_ok = classifier_and_detector(images.to(device))
    outs = outs.cpu()
    is_ok = is_ok.cpu()
    import pdb; pdb.set_trace()

    correct = data.test_labels[:num_images] == outs.argmax(1).detach().numpy()
    is_ok = is_ok.detach().numpy()
    print('acc', correct.mean())
    print('fraction bad', (is_ok > 0).mean())
    print("successful attacks", ((~correct) & (is_ok < 0)).mean())
