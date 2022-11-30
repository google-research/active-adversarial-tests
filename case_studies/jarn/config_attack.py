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

import configargparse
import pdb

def pair(arg):
    return [float(x) for x in arg.split(',')]

def get_args():
    parser = configargparse.ArgParser(default_config_files=[])
    parser.add("--model_dir", type=str, default="models/adv_trained", help="Path to save/load the checkpoints, default=models/model")
    parser.add("--data_path", type=str, default="datasets/cifar10", help="Path to dataset, default=datasets/cifar10")
    parser.add("--tf_seed", type=int, default=451760341, help="Random seed for initializing tensor-flow variables to rule out the effect of randomness in experiments, default=45160341") 
    parser.add("--np_seed", type=int, default=216105420, help="Random seed for initializing numpy variables to rule out the effect of randomness in experiments, default=216105420") 
    parser.add("--num_eval_examples", type=int, default=10000, help="Number of eval samples, default=10000")
    parser.add("--eval_batch_size", type=int, default=100, help="Eval batch size, default=100")
    parser.add("--epsilon", "-e", type=float, default=8.0, help="Epsilon (Lp Norm distance from the original image) for generating adversarial examples, default=8.0")
    parser.add("--num_steps", type=int, default=10, help="Number of steps to PGD attack, default=10")
    parser.add("--step_size", "-s", type=float, default=2.0, help="Step size in PGD attack for generating adversarial examples in each step, default=2.0")
    parser.add("--random_start", dest="random_start", action="store_true", help="Random start for PGD attack default=True")
    parser.add("--no-random_start", dest="random_start", action="store_false", help="No random start for PGD attack default=True")
    parser.set_defaults(random_start=True)
    parser.add("--loss_func", "-f", type=str, default="xent", choices=["xent", "target_task_xent", "cw"], help="Loss function for the model, choices are [xent, cw], default=xent")
    parser.add("--attack_norm", type=str, default="inf", choices=["inf", "2"], help="Lp norm type for attacks, choices are [inf, 2], default=inf")
    parser.add("--dataset", "-d", type=str, default="cifar10", choices=["cifar10", "cifar100", "imagenet"], help="Path to load dataset, default=cifar10")
    parser.add("--store_adv_path", type=str, default=None, help="Path to save adversarial examples, default=None")
    parser.add("--attack_name", type=str, default=None, help="Path to save adversarial examples, default=''")
    parser.add("--save_eval_log", dest="save_eval_log", action="store_true", help="Save txt file for attack eval")
    parser.add("--no-save_eval_log", dest="save_eval_log", action="store_false", help="Save txt file for attack eval")
    parser.set_defaults(save_eval_log=False)

    parser.add("--xfer_attack", dest="xfer_attack", action="store_true", help="Adversarial transfer attack")
    parser.add("--no-xfer_attack", dest="xfer_attack", action="store_false", help="not adversarial transfer attack")
    parser.set_defaults(xfer_attack=False)
    parser.add("--custom_output_model_name", type=str, default=None, help="Custom model name, default=None")

    parser.add("--n_inner_points", type=int, default=999, help="")
    parser.add("--n_boundary_points", type=int, default=1, help="")
    parser.add("--sample_from_corners", type=bool, default=False, help="")
    parser.add("--attack", type=str, default="pgd", help="either pgd or apgd")

    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()
