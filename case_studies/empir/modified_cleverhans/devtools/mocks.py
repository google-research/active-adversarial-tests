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

"""Utility functions for mocking up tests.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def random_feed_dict(rng, placeholders):
    """
    Returns random data to be used with `feed_dict`.
    :param rng: A numpy.random.RandomState instance
    :param placeholders: List of tensorflow placeholders
    :return: A dict mapping placeholders to random numpy values
    """

    output = {}

    for placeholder in placeholders:
        if placeholder.dtype != 'float32':
            raise NotImplementedError()
        value = rng.randn(*placeholder.shape).astype('float32')
        output[placeholder] = value

    return output
