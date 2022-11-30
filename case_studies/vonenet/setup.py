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

#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "torch>=0.4.0+",
    "torchvision",
    "numpy",
    "pandas",
    "scipy",
    "tqdm",
    "fire",
    "requests",
]

setup(
    name='vonenet',
    version='0.1.0',
    description="CNNs with a Primary Visual Cortex Front-End ",
    long_description=readme,
    author="Tiago Marques, Joel Dapello",
    author_email='tmarques@mit.edu, dapello@mit.edu',
    url='https://github.com/dicarlolab/vonenet',
    packages=['vonenet'],
    include_package_data=True,
    install_requires=requirements,
    license="GNU GPL v3",
    zip_safe=False,
    keywords='VOneNet, Robustness, Primary Visual Cortex',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU GPL v3',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.6'
    ],
)
