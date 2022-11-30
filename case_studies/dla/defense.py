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


class TorchAlarm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = torch.nn.ModuleList([
            torch.nn.Linear(28682, 112),
            torch.nn.ReLU(),
            torch.nn.Linear(112, 100),
            torch.nn.ReLU(),
            torch.nn.Linear(100, 300),
            torch.nn.ReLU(),
            torch.nn.Linear(300, 200),
            torch.nn.ReLU(),
            torch.nn.Linear(200, 77),
            torch.nn.ReLU(),
            torch.nn.Linear(77, 1),
        ])

    def __call__(self, x, training=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        for layer in self.layers:
            x = layer(x)
        return x


class resnet_layer_torch(torch.nn.Module):
    def __init__(self,
                 prior_filters=16,
                 num_filters=16,
                 kernel_size=3,
                 strides=1):
        super().__init__()
        self.a = torch.nn.Conv2d(prior_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.b = torch.nn.BatchNorm2d(num_filters, eps=.000)
        self.c = torch.nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.d = torch.nn.BatchNorm2d(num_filters, eps=.000)
        self.layers = [self.a, self.b, self.c, self.d]
    def forward(self, inputs):
        x1 = self.a(inputs)
        x2 = self.b(x1)
        x3 = torch.nn.ReLU()(x2)
        x4 = self.c(x3)
        x5 = self.d(x4)
        x6 = x5 + inputs
        return x6, x2, x5


class resnet_layer2_torch(torch.nn.Module):
    def __init__(self,
                 prior_filters=16,
                 num_filters=16,
                 kernel_size=3,
                 strides=1):
        super().__init__()
        self.a = torch.nn.Conv2d(prior_filters, num_filters, kernel_size=kernel_size, padding=0, stride=(2,2))
        self.b = torch.nn.BatchNorm2d(num_filters, eps=.000)
        self.c = torch.nn.Conv2d(num_filters, num_filters, kernel_size=kernel_size, padding=1)
        self.c2 = torch.nn.Conv2d(prior_filters, num_filters, kernel_size=1, padding=0, stride=(2,2))
        self.d = torch.nn.BatchNorm2d(num_filters, eps=.000)
        self.layers = [self.a, self.b, self.c, self.c2, self.d]

    def forward(self, x):
        xp = torch.nn.functional.pad(x, (0, 1, 0, 1), "constant", 0)
        y = self.a(xp)
        y = self.b(y)
        y = torch.nn.ReLU()(y)
        y = self.c(y)
        z = self.c2(x)
        y = self.d(y)
        x = z+y
        return x


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        class Transpose(torch.nn.Module):
            def forward(self, x):
                return x.permute((0, 2, 3, 1))

        self.layers = torch.nn.ModuleList([
            torch.nn.Conv2d(3, 16, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(16, eps=.000),
            torch.nn.ReLU(),
            # AAA                                                                                                                                                                                                                                                                              

            resnet_layer_torch(16, 16),
            torch.nn.ReLU(),
            resnet_layer_torch(16, 16),
            torch.nn.ReLU(),
            resnet_layer_torch(16, 16),
            torch.nn.ReLU(),


            resnet_layer2_torch(16, 32),
            torch.nn.ReLU(),

            resnet_layer_torch(32, 32),
            torch.nn.ReLU(),
            resnet_layer_torch(32, 32),
            torch.nn.ReLU(),

            resnet_layer2_torch(32, 64),
            torch.nn.ReLU(),

            resnet_layer_torch(64, 64),
            torch.nn.ReLU(),
            resnet_layer_torch(64, 64),
            torch.nn.ReLU(),

            torch.nn.AvgPool2d(8),
            #                                                                                                                                                                                                                                                                                  
            Transpose(),
            torch.nn.Flatten(),
            torch.nn.Linear(64, 10),


            ])

    def __call__(self, x, training=False):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        extra = []
        for i,layer in enumerate(self.layers):
            if isinstance(layer, resnet_layer_torch):
                x,y,z = layer(x)
                if i == 11:
                    extra.append(y)
                if i == 19:
                    extra.append(z)
            else:
                x = layer(x)
            if i == 1:
                extra.append(x)

        extra = torch.cat([x.permute((0, 2, 3, 1)).reshape((x.shape[0], -1)) for x in extra] + [x], axis=1)
        return x, extra


class TorchWithDetect:
    def __init__(self, model, alarm):
        self.model = model
        self.alarm = alarm
        
    def __call__(self, x):
        out, hidden = self.model(x)
        is_ok = self.alarm(hidden)
        return out, is_ok


#def load_model(path_model_weights='checkpoints/dla/dla_cifar_classifier.h5',
#    path_detector_weights='checkpoints/dla/dla_cifar_detector.h5', device=None):
def load_model(path_model_weights='/home/AUTHOR/dla/dla/cifar_model.h5',
    path_detector_weights='/home/AUTHOR/dla/dla/cifar_alarm.h5', device=None):
    torch_model = TorchModel()
    torch_model.load_state_dict(
        torch.load(path_model_weights))
    torch_model.eval().to(device)
    
    torch_alarm = TorchAlarm()
    torch_alarm.load_state_dict(
            torch.load(path_detector_weights))
    torch_alarm.eval().to(device)

    return TorchWithDetect(torch_model, torch_alarm), \
           lambda x: torch_model(x)[0], \
           lambda x, how=None: torch_alarm(torch_model(x)[1]).flatten()
