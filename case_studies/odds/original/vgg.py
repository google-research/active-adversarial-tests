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

'''
Modified from https://github.com/pytorch/vision.git
'''
import math

import torch.nn as nn
import torch.nn.init as init

__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19', 'vgg11_nd', 'vgg11_nd_s', 'vgg13_nd', 'vgg13_nd_s', 'vgg16_nd', 'vgg16_nd_s', 'vgg19_nd', 'vgg19_nd_s',
    'vgg11_nd_ss', 'vgg13_nd_ss', 'vgg16_nd_ss', 'vgg19_nd_ss',
]


class VGG(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, features, dropout=True, small=False, supersmall=False):
        super(VGG, self).__init__()
        self.features = features
        cls_layers = []
        if dropout or supersmall:
            cls_layers.append(nn.Dropout())
        if not (small or supersmall):
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
            if dropout:
                cls_layers.append(nn.Dropout())
        if not supersmall:
            cls_layers.append(nn.Linear(512, 512))
            cls_layers.append(nn.ReLU())
        cls_layers.append(nn.Linear(512, 10))

        self.classifier = nn.Sequential(*cls_layers)
         # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=False)]
            else:
                layers += [conv2d, nn.ReLU(inplace=False)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))

def vgg11_nd():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False)

def vgg11_nd_s():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True)

def vgg11_nd_ss():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']), dropout=False, small=True, supersmall=True)


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))

def vgg13_nd():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False)

def vgg13_nd_s():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True)

def vgg13_nd_ss():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']), dropout=False, small=True, supersmall=True)


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))

def vgg16_nd():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False)

def vgg16_nd_s():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True)

def vgg16_nd_ss():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']), dropout=False, small=True, supersmall=True)


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))

def vgg19_nd():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False)

def vgg19_nd_s():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False, small=True)

def vgg19_nd_ss():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']), dropout=False, small=True, supersmall=True)



def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
