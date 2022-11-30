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


class y_dequantize(nn.Module):
    """Dequantize Y channel
    Args:
        image(tensor): batch x height x width
        factor(float): compression factor
    Returns:
        batch x height x width
    """

    def __init__(self, factor=1):
        super(y_dequantize, self).__init__()
        self.y_table = utils.y_table
        self.factor = factor

    def forward(self, image):
        return image * (self.y_table * self.factor)


class c_dequantize(nn.Module):
    """Dequantize CbCr channel
    Args:
        image(tensor): batch x height x width
        factor(float): compression factor
    Returns:
        batch x height x width
    """

    def __init__(self, factor=1):
        super(c_dequantize, self).__init__()
        self.factor = factor
        self.c_table = utils.c_table

    def forward(self, image):
        return image * (self.c_table * self.factor)


class idct_8x8(nn.Module):
    """Inverse discrete Cosine Transformation
    Args:
        dcp(tensor): batch x height x width
    Returns:
        batch x height x width
    """

    def __init__(self):
        super(idct_8x8, self).__init__()
        alpha = np.array([1. / np.sqrt(2)] + [1] * 7)
        self.alpha = nn.Parameter(torch.from_numpy(np.outer(alpha, alpha)).float(),
                                                            requires_grad=False)
        tensor = np.zeros((8, 8, 8, 8), dtype=np.float32)
        for x, y, u, v in itertools.product(range(8), repeat=4):
            tensor[x, y, u, v] = np.cos((2 * u + 1) * x * np.pi / 16) * np.cos(
                    (2 * v + 1) * y * np.pi / 16)
        self.tensor = nn.Parameter(torch.from_numpy(tensor).float(),
                                                             requires_grad=False)

    def forward(self, image):
        image = image * self.alpha
        result = 0.25 * torch.tensordot(image, self.tensor, dims=2) + 128
        result.view(image.shape)
        return result


class block_merging(nn.Module):
    """Merge pathces into image
    Args:
        patches(tensor) batch x height*width/64, height x width
        height(int)
        width(int)
    Returns:
        batch x height x width
    """

    def __init__(self):
        super(block_merging, self).__init__()

    def forward(self, patches, height, width):
        k = 8
        batch_size = patches.shape[0]
        image_reshaped = patches.view(batch_size, height // k, width // k, k, k)
        image_transposed = image_reshaped.permute(0, 1, 3, 2, 4)
        return image_transposed.contiguous().view(batch_size, height, width)


class chroma_upsampling(nn.Module):
    """Upsample chroma layers
    Args:
        y(tensor): y channel image
        cb(tensor): cb channel
        cr(tensor): cr channel
    Returns:
        batch x height x width x 3
    """

    def __init__(self):
        super(chroma_upsampling, self).__init__()

    def forward(self, y, cb, cr):
        def repeat(x, k=2):
            height, width = x.shape[1:3]
            x = x.unsqueeze(-1)
            x = x.repeat(1, 1, k, k)
            x = x.view(-1, height * k, width * k)
            return x

        cb = repeat(cb)
        cr = repeat(cr)

        return torch.cat([y.unsqueeze(3), cb.unsqueeze(3), cr.unsqueeze(3)], dim=3)


class ycbcr_to_rgb_jpeg(nn.Module):
    """Converts YCbCr image to RGB JPEG"""

    def __init__(self):
        super(ycbcr_to_rgb_jpeg, self).__init__()

        matrix = np.array(
                [[1., 0., 1.402], [1, -0.344136, -0.714136], [1, 1.772, 0]],
                dtype=np.float32).T
        self.shift = nn.Parameter(torch.tensor([0, -128., -128.]),
                                                            requires_grad=False)
        self.matrix = nn.Parameter(torch.from_numpy(matrix), requires_grad=False)

    def forward(self, image):
        result = torch.tensordot(image + self.shift, self.matrix, dims=1)
        # result = torch.from_numpy(result)
        result.view(image.shape)
        return result.permute(0, 3, 1, 2)


class decompress_jpeg(nn.Module):
    """Full JPEG decompression algortihm
    Args:
        compressed(dict(tensor)): batch x h*w/64 x 8 x 8
        rounding(function): rounding function to use
        factor(float): Compression factor
    Returns:
        batch x 3 x height x width
    """

    def __init__(self, height, width, rounding=torch.round, factor=1):
        super(decompress_jpeg, self).__init__()
        self.c_dequantize = c_dequantize(factor=factor)
        self.y_dequantize = y_dequantize(factor=factor)
        self.idct = idct_8x8()
        self.merging = block_merging()
        self.chroma = chroma_upsampling()
        self.colors = ycbcr_to_rgb_jpeg()

        self.height, self.width = height, width

    def forward(self, y, cb, cr):
        components = {'y': y, 'cb': cb, 'cr': cr}
        for k in components.keys():
            if k in ('cb', 'cr'):
                comp = self.c_dequantize(components[k])
                height, width = int(self.height / 2), int(self.width / 2)
            else:
                comp = self.y_dequantize(components[k])
                height, width = self.height, self.width
            comp = self.idct(comp)
            components[k] = self.merging(comp, height, width)

        image = self.chroma(components['y'], components['cb'], components['cr'])
        image = self.colors(image)

        image = torch.min(255 * torch.ones_like(image),
                          torch.max(torch.zeros_like(image), image))
        return image / 255
