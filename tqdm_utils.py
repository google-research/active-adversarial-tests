# Copyright 2022 The Authors
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#   https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Makes print and tqdm work better together.
Based on the idea presented in https://stackoverflow.com/a/37243211
"""

import contextlib
import sys
import time
import warnings

import tqdm
from tqdm import tqdm

__all__ = ["tqdm_print"]


class __DummyFile(object):
    file = None

    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            with tqdm.external_write_mode():
                tqdm.write(x, file=self.file)

    def flush(self):
        pass


@contextlib.contextmanager
def tqdm_print(include_warnings=True):
    """Makes sure printing text/showing warnings does not interrupt a
    progressbar but just moves it to the bottom by wrapping stdout and
    passing all write statements through tqdm.write."""

    save_stdout = sys.stdout
    sys.stdout = __DummyFile(sys.stdout)

    if include_warnings:
        def redirected_showwarning(message, category, filename, lineno,
                file=sys.stdout, line=None):
            if file is None:
                file = sys.stdout
            save_showwarning(message, category, filename, lineno, file, line)

        save_showwarning = warnings.showwarning
        warnings.showwarning = redirected_showwarning

    try:
        yield
    finally:
        # restore stdout
        sys.stdout = save_stdout
        if include_warnings:
            warnings.showwarning = save_showwarning
