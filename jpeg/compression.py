# Copyright 2022 The Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#         https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MIT License
#
# Copyright (c) 2021 Michael R Lomnitz
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import itertools

import numpy as np
# PyTorch
import torch
import torch.nn as nn

# Local
from . import utils


class rgb_to_ycbcr_jpeg(nn.Module):
    """Converts RGB image to YCbCr
    Args:
          image(tensor): batch x 3 x height x width
    Returns:
          result(tensor): batch x height x width x 3
    """

    def __init__(self):
        super(rgb_to_ycbcr_jpeg, self).__init__()
        matrix = np.array(
              [[0.299, 0.587, 0.114], [-0.168736, -0.331264, 0.5],
               [0.5, -0.418688, -0.081312]], dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0., 128., 128.]),
                                                  requires_grad=False)
        #
        self.matrix = nn.Parameter(torch.from_numpy(matrix),
                                                   requires_grad=False)

    def forward(self, image):
        image = image.permute(0, 2, 3, 1)
        result = torch.tensordot(image, self.matrix, dims=1) + self.shift
        #        result = torch.from_numpy(result)
        result.view(image.shape)
        return result


class chroma_subsampling(nn.Module):
    """Chroma subsampling on CbCv channels
    Args:
          image(tensor): batch x height x width x 3
    Returns:
          y(tensor): batch x height x width
          cb(tensor): batch x height/2 x width/2
          cr(tensor): batch x height/2 x width/2
    """

    def __init__(self):
        super(chroma_subsampling, self).__init__()

    def forward(self, image):
        image_2 = image.permute(0, 3, 1, 2).clone()
        avg_pool = nn.AvgPool2d(kernel_size=2, stride=(2, 2),
                                count_include_pad=False)
        cb = avg_pool(image_2[:, 1, :, :].unsqueeze(1))
        cr = avg_pool(image_2[:, 2, :, :].unsqueeze(1))
        cb = cb.permute(0, 2, 3, 1)
        cr = cr.permute(0, 2, 3, 1)
        return image[:, :, :, 0], cb.squeeze(3), cr.squeeze(3)


class block_splitting(nn.Module):
    """ Splitting image into patches
    Input:
          image(tensor): batch x height x width
    Output:
          patch(tensor):    batch x h*w/64 x h x w
    """

    def __init__(self):
        super(block_splitting, self).__init__()
        self.k = 8

    def forward(self, image):
        height, width = image.shape[1:3]
        batch_size = image.shape[0]
        image_reshaped = image.view(batch_size, height // self.k, self.k, -1,
                                    self.k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, -1, self.k, self.k)


class dct_8x8(nn.Module):
    """Discrete Cosine Transformation
    Args:
          image(tensor): batch x height x width
    Returns:
          dcp(tensor): batch x height x width
    """

    def __init__(self):
        super(dct_8x8, self).__init__()
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
          tensor[x, y, u, v] = np.cos((2 * x + 1) * u * np.pi / 16) * np.cos(
                  (2 * y + 1) * v * np.pi / 16)
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        #
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float(),
                                   requires_grad=False)
        self.scale = nn.Parameter(
          torch.from_numpy(np.outer(alpha, alpha) * 0.25).float(),
          requires_grad=False)

    def forward(self, image):
        image = image - 128
        result = self.scale * torch.tensordot(image, self.tensor, dims=2)
        result.view(image.shape)
        return result


class y_quantize(nn.Module):
    """JPEG Quantization for Y channel
    Args:
          image(tensor): batch x height x width
          rounding(function): rounding function to use
          factor(float): Degree of compression
    Returns:
          image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(y_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.y_table = utils.y_table

    def forward(self, image):
        image = image.float() / (self.y_table * self.factor)
        image = self.rounding(image)
        return image


class c_quantize(nn.Module):
    """JPEG Quantization for CrCb channels
    Args:
          image(tensor): batch x height x width
          rounding(function): rounding function to use
          factor(float): Degree of compression
    Returns:
          image(tensor): batch x height x width
    """

    def __init__(self, rounding, factor=1):
        super(c_quantize, self).__init__()
        self.rounding = rounding
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        image = image.float() / (self.c_table * self.factor)
        image = self.rounding(image)
        return image


class compress_jpeg(nn.Module):
    """Full JPEG compression algortihm
    Args:
          imgs(tensor): batch x 3 x height x width
          rounding(function): rounding function to use
          factor(float): Compression factor
    Returns:
          compressed(dict(tensor)): batch x h*w/64 x 8 x 8
    """

    def __init__(self, rounding=torch.round, factor=1):
        super(compress_jpeg, self).__init__()
        self.l1 = nn.Sequential(
              rgb_to_ycbcr_jpeg(),
              chroma_subsampling()
        )
        self.l2 = nn.Sequential(
              block_splitting(),
              dct_8x8()
        )
        self.c_quantize = c_quantize(rounding=rounding, factor=factor)
        self.y_quantize = y_quantize(rounding=rounding, factor=factor)

    def forward(self, image):
        y, cb, cr = self.l1(image * 255)
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
          comp = self.l2(components[k])
          if k in ('cb', 'cr'):
              comp = self.c_quantize(comp)
          else:
              comp = self.y_quantize(comp)

          components[k] = comp

        return components['y'], components['cb'], components['cr']
