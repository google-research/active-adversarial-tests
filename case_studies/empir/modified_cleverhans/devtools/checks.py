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

"""Functionality for building tests.

We have to call this file "checks" and not anything with "test" as a
substring or nosetests will execute it.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import unittest


class CleverHansTest(unittest.TestCase):

    def setUp(self):
        self.test_start = time.time()
        # seed the randomness
        np.random.seed(1234)

    def tearDown(self):
        print(self.id(), "took", time.time() - self.test_start, "seconds")

    def assertClose(self, x, y, *args, **kwargs):
        # self.assertTrue(np.allclose(x, y)) doesn't give a useful message
        # on failure
        assert np.allclose(x, y, *args, **kwargs), (x, y)
