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

from io import BytesIO
from typing import Any
from typing import Callable
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms.functional as torchvision_functional
from PIL import Image
from torch.nn import init
from jpeg import DifferentiableJPEG
from torchvision.models.resnet import resnet50
from torchvision.models.inception import inception_v3


class Lambda(nn.Module):
  def __init__(self, function: Callable):
    super().__init__()
    self.function = function

  def forward(self, x, **kwargs):
    return self.function(x, **kwargs)


class InputNormalization(nn.Module):
  def __init__(self, module: nn.Module, mean: torch.Tensor, std: torch.Tensor):
    super().__init__()
    self.module = module
    self.register_buffer("mean", mean[..., None, None])
    self.register_buffer("std", std[..., None, None])

  def forward(self, x, *args, **kwargs):
    return self.module(
      torchvision_functional.normalize(x, self.mean, self.std, False), *args,
      **kwargs)


class Detector(nn.Module):
  def __init__(self, encoder: Optional[nn.Module] = None,
      n_features_encoder: int = 0, classifier: Optional[nn.Module] = None,
      n_features_classifier: int = 0, ):
    super().__init__()
    assert encoder is not None or classifier is not None

    self.encoder = encoder
    self.classifier = classifier
    n_features = n_features_encoder + n_features_classifier
    self.head = nn.Sequential(
        nn.Linear(n_features, n_features * 4),
        nn.ReLU(),
        nn.Linear(n_features * 4, n_features * 4),
        nn.ReLU(),
        nn.Linear(n_features * 4, n_features * 4),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(n_features * 4, n_features),
        nn.ReLU(),
        nn.Linear(n_features, 2),
    )

  def train(self, mode: bool = True) -> nn.Module:
    if self.encoder is not None:
      self.encoder.train(mode)
    self.head.train(mode)
    self.training = mode
    # keep classifier always in test mode
    if self.classifier is not None:
      self.classifier.train(False)

    return self

  def forward(self, x):
    features = []
    if self.encoder is not None:
      features.append(self.encoder(x))
    if self.classifier is not None:
      features.append(self.classifier(x))
    if len(features) > 1:
      features = torch.cat(features, 1)
    else:
      features = features[0]

    return self.head(features)


class ScaledLogitsModule(nn.Module):
  def __init__(self, module: nn.Module, scale: float):
    super().__init__()
    self.module = module
    self.scale = scale

  def forward(self, *args, **kwargs):
    return self.module(*args, **kwargs) * self.scale


class GaussianNoiseInputModule(nn.Module):
  def __init__(self, module: nn.Module, stddev: float):
    super().__init__()
    self.stddev = stddev
    self.module = module

  def forward(self, x, *args, **kwargs):
    x = x + torch.randn_like(x) * self.stddev
    return self.module(x, *args, **kwargs)


class __GaussianNoiseGradientFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx, input, stddev):
    ctx.intermediate_results = stddev
    return input

  @staticmethod
  def backward(ctx, grad_output):
    stddev = ctx.intermediate_results
    grad_input = grad_output + torch.randn_like(grad_output) * stddev
    return grad_input, None


gaussian_noise_gradient = __GaussianNoiseGradientFunction.apply


class GaussianNoiseGradientModule(nn.Module):
  def __init__(self, module: nn.Module, stddev: float):
    super().__init__()
    self.module = module
    self.stddev = stddev

  def forward(self, x, *args, **kwargs):
    return gaussian_noise_gradient(self.module(x, *args, **kwargs), self.stddev)


class __JPEGForwardIdentityBackwardFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, input: torch.Tensor, quality: int) -> torch.Tensor:
    res = []
    for x in input.permute(0, 2, 3, 1).detach().cpu().numpy():
      output = BytesIO()
      x = (np.clip(x, 0, 1) * 255).astype(np.uint8)
      Image.fromarray(x).save(output, 'JPEG', quality=quality)
      x = Image.open(output)
      res.append(np.array(x).transpose(2, 0, 1) / 255.0)
    res = torch.Tensor(np.array(res)).to(input.device)

    return res

  @staticmethod
  def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output, None


jpeg_forward_identity_backward = __JPEGForwardIdentityBackwardFunction.apply


class __LambdaForwardIdentityBackward(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, input: torch.Tensor,
      function: Callable) -> torch.Tensor:
    return function(input)

  @staticmethod
  def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output, None, None


lambda_forward_identity_backward = __LambdaForwardIdentityBackward.apply


class JPEGForwardIdentityBackwardModule(nn.Module):
  def __init__(self, module: nn.Module, quality: int, size: int, legacy=False):
    super().__init__()
    self.module = module

    if legacy:
      self.jpeg = lambda x: jpeg_forward_identity_backward(x, quality)
    else:
      self.jpeg_module = DifferentiableJPEG(size, size, True, quality=quality)
      self.jpeg = lambda x: lambda_forward_identity_backward(x,
                                                             self.jpeg_module)

  def forward(self, x, *args, **kwargs):
    return self.module(self.jpeg(x), *args, **kwargs)


class DifferentiableJPEGModule(nn.Module):
  def __init__(self, module: nn.Module, quality: int, size: int):
    super().__init__()
    self.module = module
    self.jpeg = DifferentiableJPEG(size, size, True, quality=quality)

  def forward(self, x, *args, **kwargs):
    return self.module(self.jpeg(x), *args, **kwargs)


class __GausianBlurForwardIdentityBackwardFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, input: torch.Tensor, kernel_size: int,
      stddev: float) -> torch.Tensor:
    return torchvision_functional.gaussian_blur(input, kernel_size, stddev)

  @staticmethod
  def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    return grad_output, None, None


gaussian_blur_forward_identity_backward = __GausianBlurForwardIdentityBackwardFunction.apply


class GausianBlurForwardIdentityBackwardModule(nn.Module):
  def __init__(self, module: nn.Module, kernel_size: int, stddev: float):
    super().__init__()
    self.module = module
    self.kernel_size = kernel_size
    self.stddev = stddev

  def forward(self, x, *args, **kwargs):
    return self.module(
        gaussian_blur_forward_identity_backward(x, self.kernel_size,
                                                self.stddev), *args, **kwargs)


class __UniversalSingularValueThresholding(torch.autograd.Function):
  """Universal Singular Value Thresholding (USVT) """

  @staticmethod
  def forward(ctx: Any, input: torch.Tensor, me_channel_concat: bool = True,
      maskp: float = 0.5, svdprob: float = 0.8):
    device = input.device
    batch_num, c, h, w = input.size()

    output = torch.zeros_like(input).cpu().numpy()

    for i in range(batch_num):
      img = (input[i] * 2 - 1).cpu().numpy()

      if me_channel_concat:
        img = np.concatenate((np.concatenate((img[0], img[1]), axis=1), img[2]),
                             axis=1)
        mask = np.random.binomial(1, maskp, h * w * c).reshape(h, w * c)
        p_obs = len(mask[mask == 1]) / (h * w * c)

        if svdprob is not None:
          u, sigma, v = np.linalg.svd(img * mask)
          S = np.zeros((h, w))
          for j in range(int(svdprob * h)):
            S[j][j] = sigma[j]
          S = np.concatenate((S, np.zeros((h, w * 2))), axis=1)
          W = np.dot(np.dot(u, S), v) / p_obs
          W[W < -1] = -1
          W[W > 1] = 1
          est_matrix = (W + 1) / 2
          for channel in range(c):
            output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]
        else:
          est_matrix = ((img * mask) + 1) / 2
          for channel in range(c):
            output[i, channel] = est_matrix[:, channel * h:(channel + 1) * h]

      else:
        mask = np.random.binomial(1, maskp, h * w).reshape(h, w)
        p_obs = len(mask[mask == 1]) / (h * w)
        for channel in range(c):
          u, sigma, v = np.linalg.svd(img[channel] * mask)
          S = np.zeros((h, w))
          for j in range(int(svdprob * h)):
            S[j][j] = sigma[j]
          W = np.dot(np.dot(u, S), v) / p_obs
          W[W < -1] = -1
          W[W > 1] = 1
          output[i, channel] = (W + 1) / 2

    output = torch.from_numpy(output).float().to(device)

    return output

  @staticmethod
  def backward(ctx: Any, grad_output: torch.Tensor):
    return grad_output, None, None, None


universal_singular_value_thresholding = __UniversalSingularValueThresholding.apply


class UVSTModule(nn.Module):
  """Apply Universal Singular Value Thresholding as suggested in ME-Net:
  Chatterjee, S. et al. Matrix estimation by universal singular value thresholding. 2015."""

  def __init__(self, module: nn.Module, me_channel_concat: bool = True,
      maskp: float = 0.5, svdprob: float = 0.8):
    super().__init__()
    self.module = module
    self.me_channel_concat = me_channel_concat
    self.maskp = maskp
    self.svdprob = svdprob

  def forward(self, x, *args, **kwargs):
    x = universal_singular_value_thresholding(x, self.me_channel_concat,
                                              self.maskp, self.svdprob)
    return self.module(x, *args, **kwargs)


class _ThermometerEncodingFunction(torch.autograd.Function):
  @staticmethod
  def forward(ctx: Any, input: torch.Tensor, l: int) -> torch.Tensor:
    ctx.intermediate_results = input, l
    return _ThermometerEncodingFunction.tau(input, l)

  @staticmethod
  def tau_hat(x, l):
    x_hat = torch.unsqueeze(x, 2)
    k = torch.arange(l, dtype=x.dtype, device=x.device)
    k = k.view((1, 1, -1, 1, 1))
    y = torch.minimum(torch.maximum(x_hat - k / l, torch.zeros_like(x_hat)),
                      torch.ones_like(x_hat))

    shape = list(x.shape)
    shape[1] = -1
    y = y.view(shape)

    return y

  @staticmethod
  def tau(x, l):
    return torch.ceil(_ThermometerEncodingFunction.tau_hat(x, l))

  @staticmethod
  def backward(ctx: Any, grad_output: torch.Tensor) -> torch.Tensor:
    input, l = ctx.intermediate_results
    with torch.enable_grad():
      value_input = _ThermometerEncodingFunction.tau_hat(input.requires_grad_(),
                                                         l)
      grad_output = torch.autograd.grad(
          (value_input,), (input,), (grad_output,))[0].detach()

    return grad_output, None


thermometer_encoding = _ThermometerEncodingFunction.apply


class ThermometerEncodingModule(nn.Module):
  def __init__(self, l: int, differentiable: bool):
    super().__init__()
    self._l = l
    self.differentaible = differentiable
    if differentiable:
      self.apply_fn = lambda x: thermometer_encoding(x, l)
    else:
      # TODO
      # self.apply_fn = lambda y: lambda_forward_identity_backward(
      #    y, lambda x: thermometer_encoding(x, l))
      self.apply_fn = lambda x: thermometer_encoding(x, l)

  @property
  def l(self):
    return self._l

  def forward(self, x):
    if self.differentaible:
      with torch.no_grad():
        return self.apply_fn(x)
    else:
      return self.apply_fn(x)


'''ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F


class _CifarResNetBasicBlock(nn.Module):
  expansion = 1

  def __init__(self, in_planes, planes, stride=1):
    super(_CifarResNetBasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(
        in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.bn2(self.conv2(out))
    out += self.shortcut(x)
    out = F.relu(out)
    return out


class _CifarResNetBottleneck(nn.Module):
  expansion = 4

  def __init__(self, in_planes, planes, stride=1):
    super().__init__()
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
    self.bn1 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1,
                           bias=False)
    self.bn3 = nn.BatchNorm2d(self.expansion * planes)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != self.expansion * planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1,
                    stride=stride, bias=False),
          nn.BatchNorm2d(self.expansion * planes)
      )

  def forward(self, x, fake_relu=False):
    out = F.relu(self.bn1(self.conv1(x)))
    out = F.relu(self.bn2(self.conv2(out)))
    out = self.bn3(self.conv3(out))
    out += self.shortcut(x)
    return F.relu(out)


class _CifarResNet(nn.Module):
  def __init__(self, block, num_blocks, num_classes=10, n_input_channels=3):
    super(_CifarResNet, self).__init__()
    self.in_planes = 64

    self.conv1 = nn.Conv2d(n_input_channels, 64, kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn1 = nn.BatchNorm2d(64)
    self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
    self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
    self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
    self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
    self.linear = nn.Linear(512 * block.expansion, num_classes)

  def _make_layer(self, block, planes, num_blocks, stride):
    strides = [stride] + [1] * (num_blocks - 1)
    layers = []
    for stride in strides:
      layers.append(block(self.in_planes, planes, stride))
      self.in_planes = planes * block.expansion
    return nn.Sequential(*layers)

  def forward(self, x, features_only=False, features_and_logits=False):
    out = F.relu(self.bn1(self.conv1(x)))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = F.avg_pool2d(out, 4)
    out = out.view(out.size(0), -1)
    if features_and_logits:
      return out, self.linear(out)
    if not features_only:
      out = self.linear(out)
    return out


def cifar_resnet18(num_classes=10):
  """Resnet18 architecture adapted for small resolutions
  Taken from https://github.com/kuangliu/pytorch-cifar"""
  return _CifarResNet(_CifarResNetBasicBlock, [2, 2, 2, 2],
                      num_classes=num_classes)


def cifar_resnet50(num_classes=10):
  """Resnet50 architecture adapted for small resolutions
  Taken from https://github.com/kuangliu/pytorch-cifar"""
  return _CifarResNet(_CifarResNetBottleneck, [3, 4, 6, 3],
                      num_classes=num_classes)


class _ThermometerCifarResNet(nn.Module):
  def __init__(self, num_classes: int, l: int, differentiable: bool):
    super().__init__()
    self.encoder = ThermometerEncodingModule(l, differentiable)
    self.model = _CifarResNet(_CifarResNetBasicBlock, [2, 2, 2, 2],
                              num_classes=num_classes, n_input_channels=l * 3)

  @property
  def l(self):
    return self.encoder.l

  def forward(self, x, features_only: bool = False, skip_encoder: bool = False):
    if not skip_encoder:
      x = self.encoder(x)
    return self.model(x, features_only)


# Taken from https://github.com/meliketoy/wide-resnet.pytorch
class WideResNetBasicBlock(nn.Module):
  def __init__(self, in_planes, planes, dropout_rate, stride=1):
    super(WideResNetBasicBlock, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_planes)
    self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1,
                           bias=True)
    self.dropout = nn.Dropout(p=dropout_rate)
    self.bn2 = nn.BatchNorm2d(planes)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                           padding=1, bias=True)

    self.shortcut = nn.Sequential()
    if stride != 1 or in_planes != planes:
      self.shortcut = nn.Sequential(
          nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
      )

  def forward(self, x):
    out = self.dropout(self.conv1(F.relu(self.bn1(x))))
    out = self.conv2(F.relu(self.bn2(out)))
    out += self.shortcut(x)

    return out


# Taken from https://github.com/meliketoy/wide-resnet.pytorch
class WideResNet(nn.Module):
  def __init__(self, depth, widen_factor, dropout_rate, num_classes=10,
      n_input_channels=3):
    super(WideResNet, self).__init__()
    self.in_planes = 16

    assert ((depth - 4) % 6 == 0), 'WideResNet depth should be 6n+4'
    n = (depth - 4) / 6
    k = widen_factor

    nStages = [16, 16 * k, 32 * k, 64 * k]

    self.conv1 = WideResNet.conv3x3(n_input_channels, nStages[0])
    self.layer1 = self._wide_layer(WideResNetBasicBlock, nStages[1], n,
                                   dropout_rate, stride=1)
    self.layer2 = self._wide_layer(WideResNetBasicBlock, nStages[2], n,
                                   dropout_rate, stride=2)
    self.layer3 = self._wide_layer(WideResNetBasicBlock, nStages[3], n,
                                   dropout_rate, stride=2)
    self.bn1 = nn.BatchNorm2d(nStages[3], momentum=0.9)
    self.linear = nn.Linear(nStages[3], num_classes)

    # initialize weights
    self.apply(WideResNet.__conv_init)

  @staticmethod
  def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=True)

  @staticmethod
  def __conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
      init.xavier_uniform_(m.weight, gain=np.sqrt(2))
      init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
      init.constant_(m.weight, 1)
      init.constant_(m.bias, 0)

  def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
    strides = [stride] + [1] * (int(num_blocks) - 1)
    layers = []

    for stride in strides:
      layers.append(block(self.in_planes, planes, dropout_rate, stride))
      self.in_planes = planes

    return nn.Sequential(*layers)

  def forward(self, x, features_only: bool = False):
    out = self.conv1(x)
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = F.relu(self.bn1(out))
    out = F.avg_pool2d(out, 8)
    out = out.view(out.size(0), -1)

    if not features_only:
      out = self.linear(out)

    return out


class _ThermometerCifarWideResNet344(nn.Module):
  def __init__(self, num_classes: int, l: int, differentiable: bool):
    super().__init__()
    self.encoder = ThermometerEncodingModule(l, differentiable)
    self.model = WideResNet(depth=34, widen_factor=4, dropout_rate=0.3,
                            num_classes=num_classes, n_input_channels=l * 3)

  @property
  def l(self):
    return self.encoder.l

  def forward(self, x, features_only: bool = False, skip_encoder: bool = False):
    if not skip_encoder:
      x = self.encoder(x)
    return self.model(x, features_only)


def thermometer_encoding_cifar_resnet18(num_classes=10, l=10,
    differentiable=True):
  """Resnet18 architecture adapted for small resolutions
  Taken from https://github.com/kuangliu/pytorch-cifar"""
  return _ThermometerCifarResNet(num_classes=num_classes, l=l,
                                 differentiable=differentiable)


def thermometer_encoding_cifar_wideresnet344(num_classes=10, l=10,
    differentiable=True):
  """WideResnet architecture.
  Taken from https://github.com/meliketoy/wide-resnet.pytorch"""
  return _ThermometerCifarWideResNet344(num_classes=num_classes, l=l,
                                        differentiable=differentiable)


def non_differentiable_10_thermometer_encoding_cifar_resnet18(num_classes=10):
  return thermometer_encoding_cifar_resnet18(num_classes=num_classes,
                                             l=10, differentiable=False)


def differentiable_10_thermometer_encoding_cifar_resnet18(num_classes=10):
  return thermometer_encoding_cifar_resnet18(num_classes=num_classes,
                                             l=10, differentiable=True)


def non_differentiable_16_thermometer_encoding_cifar_resnet18(num_classes=10):
  return thermometer_encoding_cifar_resnet18(num_classes=num_classes,
                                             l=16, differentiable=False)


def differentiable_16_thermometer_encoding_cifar_resnet18(num_classes=10):
  return thermometer_encoding_cifar_resnet18(num_classes=num_classes,
                                             l=16, differentiable=True)


def non_differentiable_16_thermometer_encoding_cifar_wideresnet344(
    num_classes=10):
  return thermometer_encoding_cifar_wideresnet344(num_classes=num_classes,
                                                  l=16, differentiable=False)


def differentiable_16_thermometer_encoding_cifar_wideresnet344(num_classes=10):
  return thermometer_encoding_cifar_wideresnet344(num_classes=num_classes,
                                                  l=16, differentiable=True)
