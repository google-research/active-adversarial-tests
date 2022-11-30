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

"""
Utility functions for keeping track of the version of CleverHans.

These functions provide a finer level of granularity than the
manually specified version string attached to each release.
"""
import hashlib
from cleverhans.devtools.list_files import list_files


def dev_version():
    """
    Returns a hexdigest of all the python files in the module.
    """

    m = hashlib.md5()
    py_files = sorted(list_files(suffix=".py"))
    for filename in py_files:
        with open(filename, 'rb') as f:
            content = f.read()
        m.update(content)
    return m.hexdigest()
