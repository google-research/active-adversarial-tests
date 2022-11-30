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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import subprocess

directory = "models/modelGTP_cifar10"
subprocess.run("python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name fgsm --save_eval_log --num_steps 1 --no-random_start --step_size 8 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name fgsm --save_eval_log --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds5 --save_eval_log --num_steps 5 --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds10 --save_eval_log --num_steps 10 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds10 --save_eval_log --num_steps 10 --model_dir {} ; python PGD_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {} ; python run_attack.py -d cifar10 --data_path datasets/cifar10 --attack_name pgds20 --save_eval_log --num_steps 20 --model_dir {}".format(directory, directory, directory, directory, directory, directory, directory, directory, directory, directory), shell=True)
print("{}: Ended evaluation on fgsm and pgd  attacks".format(datetime.now()))

