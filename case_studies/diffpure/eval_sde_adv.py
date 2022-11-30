# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for DiffPure. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------

import argparse
import logging
import yaml
import os
import time

import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from autoattack import AutoAttack
from attacks.autopgd import fix_autoattack as fix_autoattack_autopgd

from active_tests import decision_boundary_binarization as dbb
from argparse_utils import DecisionBoundaryBinarizationSettings
from stadv_eot.attacks import StAdvAttack

import dp_utils
import utils
from dp_utils import str2bool, get_accuracy, get_image_classifier, load_data

from runners.diffpure_ddpm import Diffusion
from runners.diffpure_guided import GuidedDiffusion
from runners.diffpure_sde import RevGuidedDiffusion
from runners.diffpure_ode import OdeGuidedDiffusion
from runners.diffpure_ldsde import LDGuidedDiffusion


def patch_robustbench_models():
    import robustbench.model_zoo.architectures as arch
    def wide_resnet_forward(self, x, features_only=False, features_and_logits=False):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        if features_only:
            return out
        l = self.fc(out)
        if features_and_logits:
            return out, l
        return l

    def robust_wide_resnet_forward(self, x, features_only=False, features_and_logits=False):
        out = self.stem_conv(x)
        for i, block in enumerate(self.blocks):
            out = block(out)
        out = self.relu(self.bn1(out))
        out = self.global_pooling(out)
        out = out.view(-1, self.fc_size)
        if features_only:
            return out
        l = self.fc(out)
        if features_and_logits:
            return out, l
        return out

    def cifar_resnext_forward(self, x, features_only=False, features_and_logits=False):
        x = self.conv_1_3x3(x)
        x = F.relu(self.bn_1(x), inplace=True)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        if features_only:
            return x
        l = self.classifier(x)
        if features_and_logits:
            return x, l
        return l

    def preact_resenet_forward(self, x, features_only=False, features_and_logits=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.bn_before_fc:
            out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if features_only:
            return out
        l = self.linear(out)
        if features_and_logits:
            return out, l
        return l

    def resnet_forward(self, x, features_only=False, features_and_logits=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if features_only:
            return out
        l = self.linear(out)
        if features_and_logits:
            out, l
        return l

    def dm_preact_resnet_forward(self, x, features_only=False, features_and_logits=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.conv_2d(out)
        out = self.layer_0(out)
        out = self.layer_1(out)
        out = self.layer_2(out)
        out = self.layer_3(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        if features_only:
            return out
        l = self.logits(out)
        if features_and_logits:
            return out, l
        return l

    def dm_resnet_forward(self, x, features_only=False, features_and_logits=False):
        if self.padding > 0:
            x = F.pad(x, (self.padding,) * 4)
        out = (x - self.mean) / self.std
        out = self.init_conv(out)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        if features_only:
            return out
        l = self.logits(out)
        if features_and_logits:
            return out, l
        return l

    arch.wide_resnet.WideResNet.forward = wide_resnet_forward
    arch.robust_wide_resnet.RobustWideResNet.forward = robust_wide_resnet_forward
    arch.resnext.CifarResNeXt.forward = cifar_resnext_forward
    arch.resnet.PreActResNet.forward = preact_resenet_forward
    arch.resnet.ResNet.forward = resnet_forward
    arch.dm_wide_resnet.DMPreActResNet.forward = dm_preact_resnet_forward
    arch.dm_wide_resnet.DMWideResNet.forward = dm_resnet_forward

    print("Patched RobustBench classifiers.")


class SDE_Adv_Model(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args

        # image classifier
        self.classifier = get_image_classifier(args.classifier_name).to(config.device)

        # diffusion model
        print(f'diffusion_type: {args.diffusion_type}')
        if args.diffusion_type == 'ddpm':
            self.runner = GuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'sde':
            self.runner = RevGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ode':
            self.runner = OdeGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'ldsde':
            self.runner = LDGuidedDiffusion(args, config, device=config.device)
        elif args.diffusion_type == 'celebahq-ddpm':
            self.runner = Diffusion(args, config, device=config.device)
        else:
            raise NotImplementedError('unknown diffusion type')

        self.register_buffer('counter', torch.zeros(1, device=config.device))
        self.tag = None

    def reset_counter(self):
        self.counter = torch.zeros(1, dtype=torch.int, device=config.device)

    def set_tag(self, tag=None):
        self.tag = tag

    def forward(self, x, features_only=False, features_and_logits=False):
        counter = self.counter.item()
        if counter % 10 == 0:
            print(f'diffusion times: {counter}')

        # imagenet [3, 224, 224] -> [3, 256, 256] -> [3, 224, 224]
        if 'imagenet' in self.args.domain:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)

        start_time = time.time()
        x_re = self.runner.image_editing_sample((x - 0.5) * 2, bs_id=counter, tag=self.tag)
        minutes, seconds = divmod(time.time() - start_time, 60)

        if 'imagenet' in self.args.domain:
            x_re = F.interpolate(x_re, size=(224, 224), mode='bilinear', align_corners=False)

        if counter % 10 == 0:
            print(f'x shape (before diffusion models): {x.shape}')
            print(f'x shape (before classifier): {x_re.shape}')
            print("Sampling time per batch: {:0>2}:{:05.2f}".format(int(minutes), seconds))

        out = self.classifier((x_re + 1) * 0.5, features_only=features_only, features_and_logits=features_and_logits)

        self.counter += 1

        return out


def eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    attack_version = args.attack_version  # ['standard', 'rand', 'custom']
    if attack_version == 'standard':
        attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
    elif attack_version == 'rand':
        attack_list = ['apgd-ce', 'apgd-dlr']
    elif attack_version == 'custom':
        attack_list = args.attack_type.split(',')
    else:
        raise NotImplementedError(f'Unknown attack version: {attack_version}!')
    print(f'attack_version: {attack_version}, attack_list: {attack_list}')  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']

    # ---------------- apply the attack to classifier ----------------
    print(f'apply the attack to classifier [{args.lp_norm}]...')
    classifier = get_image_classifier(args.classifier_name).to(config.device)
    adversary_resnet = AutoAttack(classifier, norm=args.lp_norm, eps=args.adv_eps,
                                  version=attack_version, attacks_to_run=attack_list,
                                  log_path=f'{log_dir}/log_resnet.txt', device=config.device)
    if attack_version == 'custom':
        adversary_resnet.apgd.n_restarts = 1
        adversary_resnet.fab.n_restarts = 1
        adversary_resnet.apgd_targeted.n_restarts = 1
        adversary_resnet.fab.n_target_classes = 9
        adversary_resnet.apgd_targeted.n_target_classes = 9
        adversary_resnet.square.n_queries = 5000
    if attack_version == 'rand':
        adversary_resnet.apgd.eot_iter = args.eot_iter
        print(f'[classifier] rand version with eot_iter: {adversary_resnet.apgd.eot_iter}')
    print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

    x_adv_resnet = adversary_resnet.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
    print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
    torch.save([x_adv_resnet, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')

    # ---------------- apply the attack to sde_adv ----------------
    print(f'apply the attack to sde_adv [{args.lp_norm}]...')
    model_.reset_counter()
    adversary_sde = AutoAttack(model, norm=args.lp_norm, eps=args.adv_eps,
                               version=attack_version, attacks_to_run=attack_list,
                               log_path=f'{log_dir}/log_sde_adv.txt', device=config.device)
    if attack_version == 'custom':
        adversary_sde.apgd.n_restarts = 1
        adversary_sde.fab.n_restarts = 1
        adversary_sde.apgd_targeted.n_restarts = 1
        adversary_sde.fab.n_target_classes = 9
        adversary_sde.apgd_targeted.n_target_classes = 9
        adversary_sde.square.n_queries = 5000
    if attack_version == 'rand':
        adversary_sde.apgd.eot_iter = args.eot_iter
        print(f'[adv_sde] rand version with eot_iter: {adversary_sde.apgd.eot_iter}')
    print(f'{args.lp_norm}, epsilon: {args.adv_eps}')

    x_adv_sde = adversary_sde.run_standard_evaluation(x_val, y_val, bs=adv_batch_size)
    print(f'x_adv_sde shape: {x_adv_sde.shape}')
    torch.save([x_adv_sde, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')


def eval_stadv(args, config, model, x_val, y_val, adv_batch_size, log_dir):
    ngpus = torch.cuda.device_count()
    model_ = model
    if ngpus > 1:
        model_ = model.module

    x_val, y_val = x_val.to(config.device), y_val.to(config.device)
    print(f'bound: {args.adv_eps}')

    # apply the attack to resnet
    print(f'apply the stadv attack to resnet...')
    resnet = get_image_classifier(args.classifier_name).to(config.device)

    start_time = time.time()
    init_acc = get_accuracy(resnet, x_val, y_val, bs=adv_batch_size)
    print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    adversary_resnet = StAdvAttack(resnet, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter)

    start_time = time.time()
    x_adv_resnet = adversary_resnet(x_val, y_val)

    robust_acc = get_accuracy(resnet, x_adv_resnet, y_val, bs=adv_batch_size)
    print('robust accuracy: {:.2%}, time elapsed: {:.2f}s'.format(robust_acc, time.time() - start_time))

    print(f'x_adv_resnet shape: {x_adv_resnet.shape}')
    torch.save([x_adv_resnet, y_val], f'{log_dir}/x_adv_resnet_sd{args.seed}.pt')

    # apply the attack to sde_adv
    print(f'apply the stadv attack to sde_adv...')

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag('no_adv')
    init_acc = get_accuracy(model, x_val, y_val, bs=adv_batch_size)
    print('initial accuracy: {:.2%}, time elapsed: {:.2f}s'.format(init_acc, time.time() - start_time))

    adversary_sde = StAdvAttack(model, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter)

    start_time = time.time()
    model_.reset_counter()
    model_.set_tag()
    x_adv_sde = adversary_sde(x_val, y_val)

    model_.reset_counter()
    model_.set_tag('sde_adv')
    robust_acc = get_accuracy(model, x_adv_sde, y_val, bs=adv_batch_size)
    print('robust accuracy: {:.2%}, time elapsed: {:.2f}s'.format(robust_acc, time.time() - start_time))

    print(f'x_adv_sde shape: {x_adv_sde.shape}')
    torch.save([x_adv_sde, y_val], f'{log_dir}/x_adv_sde_sd{args.seed}.pt')


def binarization_eval(args, config):
    print("Running binarization test.")

    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard',
                                                                                                  'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed) + "_" +
                           str(args.test_samples_idx_start) + "_" + str(args.test_samples_idx_end))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    x_val, y_val = load_data(args, adv_batch_size, binarization_test=True)
    testloader = utils.build_dataloader_from_arrays(x_val.detach().cpu().numpy(), y_val.detach().cpu().numpy())
    adv_batch_size = args.adv_batch_size * ngpus

    print('adv_batch_size', adv_batch_size)
    print('x_val', x_val.shape)

    if args.attack_version in ['standard', 'rand', 'custom']:
        attack_version = args.attack_version  # ['standard', 'rand', 'custom']
        if attack_version == 'standard':
            attack_list = ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        elif attack_version == 'rand':
            attack_list = ['apgd-ce', 'apgd-dlr']
        elif attack_version == 'custom':
            attack_list = args.attack_type.split(',')
        else:
            raise NotImplementedError(f'Unknown attack version: {attack_version}!')
        print(
            f'attack_version: {attack_version}, attack_list: {attack_list}')  # ['apgd-ce', 'apgd-t', 'fab-t', 'square']
        print(f'{args.lp_norm}, epsilon: {args.adv_eps}')
    elif args.attack_version == 'stadv':
        print("Using StAdv attack.")
    else:
        raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

    def eval(m, x, y):
        model.reset_counter()

        if args.attack_version in ['standard', 'rand', 'custom']:
            adversary_sde = AutoAttack(m, norm=args.lp_norm, eps=args.adv_eps,
                                       version=attack_version, attacks_to_run=attack_list,
                                       device=config.device)
            # Fix loss functions of APGD such that they are properly defined for binary classification problems.
            fix_autoattack_autopgd(adversary_sde)
            if attack_version == 'custom':
                adversary_sde.apgd.n_restarts = 1
                adversary_sde.fab.n_restarts = 1
                adversary_sde.apgd_targeted.n_restarts = 1
                adversary_sde.fab.n_target_classes = 1
                adversary_sde.apgd_targeted.n_target_classes = 1
                adversary_sde.square.n_queries = 5000
            if attack_version == 'rand':
                adversary_sde.apgd.eot_iter = args.eot_iter
            x_adv_sde = adversary_sde.run_standard_evaluation(x, y, bs=len(x))
        elif args.attack_version == 'stadv':
            adversary_sde = StAdvAttack(m, bound=args.adv_eps, num_iterations=100, eot_iter=args.eot_iter)
            x_adv_sde = adversary_sde(x, y)
        else:
            raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

        x_adv_logits = m(x_adv_sde)
        robust_acc = get_accuracy(m, x_adv_sde, y, bs=adv_batch_size)

        return robust_acc, (x_adv_sde, x_adv_logits)

    scores_logit_differences_and_validation_accuracies = dbb.interior_boundary_discrimination_attack(
        model,
        testloader,
        attack_fn=lambda m, l, kwargs: eval(m, l.dataset.tensors[0].to(config.device),
                                            l.dataset.tensors[1].to(config.device)),
        linearization_settings=DecisionBoundaryBinarizationSettings(
            epsilon=args.adv_eps,
            norm="linf",
            lr=10000,
            n_boundary_points=args.n_boundary_points,
            n_inner_points=args.n_inner_points,
            adversarial_attack_settings=None,
            optimizer="sklearn"
        ),
        n_samples=len(x_val),
        device=config.device,
        n_samples_evaluation=200,  # args.num_samples_test * 10,
        n_samples_asr_evaluation=200,
        batch_size=args.batch_size * ngpus,
        rescale_logits="adaptive",
        decision_boundary_closeness=0.999,
        sample_training_data_from_corners=args.sample_from_corners
    )

    print(dbb.format_result(scores_logit_differences_and_validation_accuracies, len(x_val)))


def robustness_eval(args, config):
    middle_name = '_'.join([args.diffusion_type, args.attack_version]) if args.attack_version in ['stadv', 'standard', 'rand'] \
        else '_'.join([args.diffusion_type, args.attack_version, args.attack_type])
    log_dir = os.path.join(args.image_folder, args.classifier_name, middle_name,
                           'seed' + str(args.seed), 'data' + str(args.data_seed))
    os.makedirs(log_dir, exist_ok=True)
    args.log_dir = log_dir
    logger = dp_utils.Logger(file_name=f'{log_dir}/log.txt', file_mode="w+", should_flush=True)

    ngpus = torch.cuda.device_count()
    adv_batch_size = args.adv_batch_size * ngpus
    print(f'ngpus: {ngpus}, adv_batch_size: {adv_batch_size}')

    # load model
    print('starting the model and loader...')
    model = SDE_Adv_Model(args, config)
    if ngpus > 1:
        model = torch.nn.DataParallel(model)
    model = model.eval().to(config.device)

    # load data
    x_val, y_val = load_data(args, adv_batch_size)

    # eval classifier and sde_adv against attacks
    if args.attack_version in ['standard', 'rand', 'custom']:
        eval_autoattack(args, config, model, x_val, y_val, adv_batch_size, log_dir)
    elif args.attack_version == 'stadv':
        eval_stadv(args, config, model, x_val, y_val, adv_batch_size, log_dir)
    else:
        raise NotImplementedError(f'unknown attack_version: {args.attack_version}')

    logger.close()


def parse_args_and_config():
    parser = argparse.ArgumentParser(description=globals()['__doc__'])
    # diffusion models
    parser.add_argument('--config', type=str, required=True, help='Path to the config file')
    parser.add_argument('--data_seed', type=int, default=0, help='Random seed')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--exp', type=str, default='exp', help='Path for saving running related data.')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('-i', '--image_folder', type=str, default='images', help="The folder name of samples")
    parser.add_argument('--ni', action='store_true', help="No interaction. Suitable for Slurm Job launcher")
    parser.add_argument('--sample_step', type=int, default=1, help='Total sampling steps')
    parser.add_argument('--t', type=int, default=400, help='Sampling noise scale')
    parser.add_argument('--t_delta', type=int, default=15, help='Perturbation range of sampling noise scale')
    parser.add_argument('--rand_t', type=str2bool, default=False, help='Decide if randomize sampling noise scale')
    parser.add_argument('--diffusion_type', type=str, default='ddpm', help='[ddpm, sde]')
    parser.add_argument('--score_type', type=str, default='guided_diffusion', help='[guided_diffusion, score_sde]')
    parser.add_argument('--eot_iter', type=int, default=20, help='only for rand version of autoattack')
    parser.add_argument('--use_bm', action='store_true', help='whether to use brownian motion')

    # LDSDE
    parser.add_argument('--sigma2', type=float, default=1e-3, help='LDSDE sigma2')
    parser.add_argument('--lambda_ld', type=float, default=1e-2, help='lambda_ld')
    parser.add_argument('--eta', type=float, default=5., help='LDSDE eta')
    parser.add_argument('--step_size', type=float, default=1e-2, help='step size for ODE Euler method')

    # adv
    parser.add_argument('--domain', type=str, default='celebahq', help='which domain: celebahq, cat, car, imagenet')
    parser.add_argument('--classifier_name', type=str, default='Eyeglasses', help='which classifier to use')
    parser.add_argument('--partition', type=str, default='val')
    parser.add_argument('--adv_batch_size', type=int, default=64)
    parser.add_argument('--attack_type', type=str, default='square')
    parser.add_argument('--lp_norm', type=str, default='Linf', choices=['Linf', 'L2'])
    parser.add_argument('--attack_version', type=str, default='custom')

    parser.add_argument('--num_sub', type=int, default=1000, help='imagenet subset')
    parser.add_argument('--adv_eps', type=float, default=0.07)

    # Binarization Test
    parser.add_argument("--binarization-test", action="store_true")
    parser.add_argument("--batch-size", default=64, type=int)
    parser.add_argument("--n_inner_points", default=999, type=int)
    parser.add_argument("--n_boundary_points", default=1, type=int)
    parser.add_argument("--sample-from-corners", action="store_true")
    parser.add_argument("--test-samples-idx-start", default=0, type=int)
    parser.add_argument("--test-samples-idx-end", default=64, type=int)
    # parser.add_argument('--gpu_ids', type=str, default='0')

    args = parser.parse_args()

    # parse config file
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    new_config = dp_utils.dict2namespace(config)

    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.setLevel(level)

    args.image_folder = os.path.join(args.exp, args.image_folder)
    os.makedirs(args.image_folder, exist_ok=True)

    # add device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    torch.backends.cudnn.benchmark = True

    return args, new_config


if __name__ == '__main__':
    patch_robustbench_models()
    args, config = parse_args_and_config()
    # os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    if args.binarization_test:
        binarization_eval(args, config)
    else:
        robustness_eval(args, config)
