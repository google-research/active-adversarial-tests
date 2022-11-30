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
    parser.add("--config", type=str, is_config_file=True, help="You can store all the config args in a config file and pass the path here")
    parser.add("--model_dir", type=str, default="models/model", help="Path to save/load the checkpoints, default=models/model")
    parser.add("--data_dir", type=str, default="datasets/", help="Path to load datasets from, default=datasets")
    parser.add("--model_suffix", type=str, default="", help="Suffix to append to model name, default=''")
    parser.add("--dataset", "-d", type=str, default="cifar10", choices=["cifar10", "cifar100", "svhn"], help="Path to load dataset, default=cifar10")
    parser.add("--tf_seed", type=int, default=451760341, help="Random seed for initializing tensor-flow variables to rule out the effect of randomness in experiments, default=45160341") 
    parser.add("--np_seed", type=int, default=216105420, help="Random seed for initializing numpy variables to rule out the effect of randomness in experiments, default=216105420") 
    parser.add("--train_steps", type=int, default=80000, help="Maximum number of training steps, default=80000")
    parser.add("--out_steps", "-o", type=int, default=100, help="Number of output steps, default=100")
    parser.add("--summary_steps", type=int, default=500, help="Number of summary steps, default=500") 
    parser.add("--checkpoint_steps", "-c", type=int, default=1000, help="Number of checkpoint steps, default=1000")
    parser.add("--train_batch_size", "-b", type=int, default=128, help="The training batch size, default=128")
    parser.add("--step_size_schedule", nargs='+', type=pair, default=[[0, 0.1], [40000, 0.01], [60000, 0.001]], help="The step size scheduling, default=[[0, 0.1], [40000, 0.01], [60000, 0.001]], use like: --stepsize 0,0.1 40000,0.01 60000,0.001") 
    parser.add("--weight_decay", "-w", type=float, default=0.0002, help="The weight decay parameter, default=0.0002")
    parser.add("--momentum", type=float, default=0.9, help="The momentum parameter, default=0.9")
    parser.add("--replay_m", "-m", type=int, default=8, help="Number of steps to repeat training on the same batch, default=8")
    parser.add("--eval_examples", type=int, default=10000, help="Number of evaluation examples, default=10000")
    parser.add("--eval_size", type=int, default=128, help="Evaluation batch size, default=128")
    parser.add("--eval_cpu", dest='eval_cpu', action='store_true', help="Set True to do evaluation on CPU instead of GPU, default=False")
    parser.set_defaults(eval_cpu=False)
    # attack params
    parser.add("--epsilon", "-e", type=float, default=8.0, help="Epsilon (Lp Norm distance from the original image) for generating adversarial examples, default=8.0")
    parser.add("--pgd_steps", "-k", type=int, default=20, help="Number of steps to PGD attack, default=20")
    parser.add("--step_size", "-s", type=float, default=2.0, help="Step size in PGD attack for generating adversarial examples in each step, default=2.0")
    parser.add("--loss_func", "-f", type=str, default="xent", choices=["xent", "cw"], help="Loss function for the model, choices are [xent, cw], default=xent")
    parser.add("--num_restarts", type=int, default=1, help="Number of resets for the PGD attack, default=1")
    parser.add("--random_start", dest="random_start", action="store_true", help="Random start for PGD attack default=True")
    parser.add("--no-random_start", dest="random_start", action="store_false", help="No random start for PGD attack default=True")
    parser.set_defaults(random_start=True)
    # input grad generation param
    parser.add("--randinit_repeat", type=int, default=1, help="Number of randinit grad to generate, default=1")
    parser.add("--num_gen_grad", type=int, default=0, help="Number of input grad samples to generate, 0 means all data default=0")
    parser.add("--num_gen_act", type=int, default=0, help="Number of activation samples to generate, 0 means all data default=0")
    # input grad reg params
    parser.add("--beta", type=float, default=1, help="Weight of input gradient regularization, default=1")
    parser.add("--gamma", type=float, default=1, help="Weight of disc xent term on encoder opt, default=1")
    parser.add("--alpha", type=float, default=0, help="Weight of image-input gradient l2 norm regularization, default=0")
    parser.add("--disc_update_steps", type=int, default=5, help="Number of classifier opt steps between each disc opt step, default=5")
    parser.add("--adv_update_steps_per_iter", type=int, default=1, help="Number of classifier adv opt steps per classification xent opt step, default=1")
    parser.add("--disc_layers", type=int, default=5, help="Number of conv layers in disc model, default=5")
    parser.add("--disc_base_channels", type=int, default=16, help="Number of channels in first disc conv layer, default=16")
    parser.add("--steps_before_adv_opt", type=int, default=0, help="Number of training steps to wait before training on adv loss, default=0")
    parser.add("--adv_encoder_type", type=str, default='simple', help="Type of input grad encoder for adv training, default=simple")
    parser.add("--enc_output_activation", type=str, default='tanh', help="Activation function of encoder output default=None")
    parser.add("--sep_opt_version", type=int, default=1, choices=[0, 1, 2], help="Sep opt version 0: train_jan.py,  1: train_jan_sep_opt-CD.py,  2: train_jan_sep_opt2-CD.py default=1")
    parser.add("--grad_image_ratio", type=float, default=1, help="Ratio of input grad to mix with image default=1")
    parser.add("--final_grad_image_ratio", type=float, default=0, help="Final ratio of input grad to mix with image, set to 0 for static ratio default=0")
    parser.add("--num_grad_image_ratios", type=int, default=5, help="Number of times to adjust grad_image_ratio default=4")
    
    parser.add("--eval_adv_attack", dest="eval_adv_attack", action="store_true", help="Evaluate trained model on adv attack after training default=True")
    parser.add("--no-eval_adv_attack", dest="eval_adv_attack", action="store_false", help="Evaluate trained model on adv attack after training default=True")
    parser.set_defaults(eval_adv_attack=True)

    parser.add("--normalize_zero_mean", dest="normalize_zero_mean", action="store_true", help="Normalize classifier input to zero mean default=True")
    parser.add("--no-normalize_zero_mean", dest="normalize_zero_mean", action="store_false", help="Normalize classifier input to zero mean default=True")
    parser.set_defaults(normalize_zero_mean=True)

    parser.add("--same_optimizer", dest="same_optimizer", action="store_true", help="Train classifier and disc with same optimizer configuration default=True")
    parser.add("--no-same_optimizer", dest="same_optimizer", action="store_false", help="Train classifier and disc with same optimizer configuration default=True")
    parser.set_defaults(same_optimizer=True)

    parser.add("--only_fully_connected", dest="only_fully_connected", action="store_true", help="Fully connected disc model default=False")
    parser.add("--no-only_fully_connected", dest="only_fully_connected", action="store_false", help="Fully connected disc model default=False")
    parser.set_defaults(only_fully_connected=False)

    parser.add("--img_random_pert", dest="img_random_pert", action="store_true", help="Random start image pertubation augmentation default=False")
    parser.add("--no-img_random_pert", dest="img_random_pert", action="store_false", help="No random start image pertubation augmentation default=False")
    parser.set_defaults(img_random_pert=False)

    args = parser.parse_args()
    return args


if __name__ == "__main__": 
    print(get_args())
    pdb.set_trace()

# TODO Default for model_dir
# TODO Need to update the helps
