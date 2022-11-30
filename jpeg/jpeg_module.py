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

import torch
import torch.nn as nn
# Local
import torchvision.transforms

from jpeg.compression import compress_jpeg
from jpeg.decompression import decompress_jpeg
from jpeg.utils import diff_round, quality_to_factor


class DifferentiableJPEG(nn.Module):
  def __init__(self, height, width, differentiable=True, quality=80):
    """ Initialize the DiffJPEG layer
    Args:
        height: Original image height
        width: Original image width
        differentiable: If true uses custom differentiable
            rounding function, if false uses standard torch.round
        quality: Quality factor for jpeg compression scheme.
    """
    super(DifferentiableJPEG, self).__init__()
    if differentiable:
      rounding = diff_round
    else:
      rounding = torch.round
    factor = quality_to_factor(quality)
    self.compress = compress_jpeg(rounding=rounding, factor=factor)
    self.decompress = decompress_jpeg(height, width, rounding=rounding,
                                      factor=factor)

  def forward(self, x):
    y, cb, cr = self.compress(x)
    recovered = self.decompress(y, cb, cr)

    return recovered
