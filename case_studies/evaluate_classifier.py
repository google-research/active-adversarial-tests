# Copyright 2022 The Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import copy
import textwrap
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchvision
import tqdm
from torchvision import transforms
from typing_extensions import Literal
import warnings

import active_tests.decision_boundary_binarization as dbl
import argparse_utils
import argparse_utils as aut
import networks
from attacks import adaptive_kwta_attack
from attacks import autopgd
from attacks import fab
from attacks import pgd
from attacks import thermometer_ls_pgd


LossType = Union[Literal["ce"], Literal["logit-diff"]]


def parse_arguments():
    """Parse arguments."""
    parser = argparse.ArgumentParser("Evaluation Script")
    parser.add_argument(
        "-ds", "--dataset", choices=("cifar10", "imagenet"), default="cifar10"
    )
    parser.add_argument("-bs", "--batch-size", default=128, type=int)
    parser.add_argument("-ns", "--n-samples", default=512, type=int)
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("-d", "--device", default=None, type=str)
    parser.add_argument(
        "-c",
        "--classifier",
        default="networks.cifar_resnet18",
        type=argparse_utils.parse_classifier_argument,
    )
    parser.add_argument("-cin", "--classifier-input-noise", default=0.0, type=float)
    parser.add_argument("-cgn", "--classifier-gradient-noise", default=0.0, type=float)
    parser.add_argument("-cls", "--classifier-logit-scale", default=1.0, type=float)
    parser.add_argument(
        "-cinorm", "--classifier-input-normalization", action="store_true"
    )
    parser.add_argument(
        "-cijq",
        "--classifier-input-jpeg-quality",
        default=100,
        type=int,
        help="Setting a negative value leads to a differentiable JPEG version",
    )
    parser.add_argument(
        "-cigb", "--classifier-input-gaussian-blur-stddev", default=0.0, type=float
    )

    parser.add_argument(
        "-a",
        "--adversarial-attack",
        type=aut.parse_adversarial_attack_argument,
        default=None,
    )

    parser.add_argument(
        "-dbl",
        "--decision-boundary-binarization",
        type=aut.parse_decision_boundary_binarization_argument,
        default=None,
    )
    parser.add_argument("--dbl-sample-from-corners", action="store_true")

    parser.add_argument("-nfs", "--n-final-softmax", default=1, type=int)
    parser.add_argument("-ciusvt", "--classifier-input-usvt", action="store_true")
    parser.add_argument("--no-ce-loss", action="store_true")
    parser.add_argument("--no-logit-diff-loss", action="store_true")
    parser.add_argument("--no-clean-evaluation", action="store_true")
    args = parser.parse_args()

    assert not (
        args.no_ce_loss and args.no_logit_diff_loss
    ), "Only one loss can be disabled"

    print("Detected type of tests to run:")
    if args.adversarial_attack is not None:
        print("\tadversarial attack:", args.adversarial_attack)

    if args.decision_boundary_binarization is not None:
        print(
            "\tinterior-vs-boundary discrimination:",
            args.decision_boundary_binarization,
        )

    print()

    return args


def setup_dataloader(
    dataset: Union[Literal["cifar10", "imagenet"]], batch_size: int
) -> torch.utils.data.DataLoader:
    if dataset == "cifar10":
        transform_test = transforms.Compose([transforms.ToTensor()])
        create_dataset_fn = lambda download: torchvision.datasets.CIFAR10(
            root="./data/cifar10",
            train=False,
            download=download,
            transform=transform_test,
        )
    elif dataset == "imagenet":
        transform_test = transforms.Compose(
            [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
        )
        create_dataset_fn = lambda _: torchvision.datasets.ImageNet(
            root="./data/imagenet", split="val", transform=transform_test
        )
    else:
        raise ValueError("Invalid value for dataset.")
    try:
        testset = create_dataset_fn(False)
    except:
        testset = create_dataset_fn(True)

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=True, num_workers=8
    )

    return testloader


def main():
    args = parse_arguments()

    if args.input == "pretrained":
        assert args.dataset == "imagenet"
        classifier = args.classifier(pretrained=True)
        print("Base Classifier:", args.classifier.__name__)
    else:
        classifier = args.classifier(
            **({"pretrained": False} if args.dataset == "imagenet" else {})
        )
        print("Base Classifier:", args.classifier.__name__)

        print("Loading checkpoint:", args.input)
        state_dict = torch.load(args.input, map_location="cpu")
        if "classifier" in state_dict:
            classifier_state_dict = state_dict["classifier"]
        else:
            classifier_state_dict = state_dict
        try:
            classifier.load_state_dict(classifier_state_dict)
        except RuntimeError as ex:
            print(
                f"Could not load weights due to error: "
                f"{textwrap.shorten(str(ex), width=50, placeholder='...')}"
            )
            print("Trying to remap weights by removing 'module.' namespace")
            modified_classifier_state_dict = {
                (k[len("module.") :] if k.startswith("module.") else k): v
                for k, v in classifier_state_dict.items()
            }
            try:
                classifier.load_state_dict(modified_classifier_state_dict)
                print("Successfully loaded renamed weights.")
            except RuntimeError:
                print("Remapping weights did also not work. Initial error:")
                raise ex
    classifier.train(False)

    test_loader = setup_dataloader(args.dataset, args.batch_size)
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    classifier = classifier.to(args.device)

    if args.classifier_input_normalization:
        if args.dataset == "cifar10":
            classifier = networks.InputNormalization(
                classifier,
                torch.tensor([0.4914, 0.4822, 0.4465]),
                torch.tensor([0.2023, 0.1994, 0.2010]),
            )
        elif args.dataset == "imagenet":
            classifier = networks.InputNormalization(
                classifier,
                torch.tensor([0.485, 0.456, 0.406]),
                torch.tensor([0.229, 0.224, 0.225]),
            )
        else:
            raise ValueError("Unknown dataset.")

    if args.classifier_input_noise > 0:
        classifier = networks.GaussianNoiseInputModule(
            classifier, args.classifier_input_noise
        )
    if args.classifier_gradient_noise > 0:
        classifier = networks.GaussianNoiseGradientModule(
            classifier, args.classifier_gradient_noise
        )

    if args.classifier_input_jpeg_quality != 100:
        if args.classifier_input_jpeg_quality > 0:
            classifier = networks.JPEGForwardIdentityBackwardModule(
                classifier,
                args.classifier_input_jpeg_quality,
                size=32 if args.dataset == "cifar10" else 224,
                legacy=True,
            )
            print("Using (slow) legacy JPEG mode")
        else:
            classifier = networks.DifferentiableJPEGModule(
                classifier,
                args.classifier_input_jpeg_quality,
                size=32 if args.dataset == "cifar10" else 224,
            )
        classifier = classifier.to(args.device)

    if args.classifier_input_gaussian_blur_stddev > 0:
        classifier = networks.GausianBlurForwardIdentityBackwardModule(
            classifier, 3, args.classifier_input_gaussian_blur_stddev
        )

    if args.classifier_input_usvt:
        classifier = networks.UVSTModule(classifier)

    if args.classifier_logit_scale > 1:
        classifier = networks.ScaledLogitsModule(
            classifier, args.classifier_logit_scale
        )

    if args.n_final_softmax > 1:
        classifier = torch.nn.Sequential(
            classifier, *[torch.nn.Softmax() for _ in range(args.n_final_softmax)]
        )

    classifier = classifier.to(args.device)

    if not args.no_clean_evaluation:
        print(
            "clean evaluation, Accuracy: {0}\n\tclass accuracy: {1}\n\tclass histogram: {3}".format(
                *run_clean_evaluation(classifier, test_loader, args.device)
            )
        )

    if args.adversarial_attack is not None:
        print("adversarial evaluation:")
        if not args.no_ce_loss:
            print(
                "\tadversarial evaluation (ce loss), ASR:",
                run_adversarial_evaluation(
                    classifier,
                    test_loader,
                    "ce",
                    args.adversarial_attack,
                    args.n_samples,
                    args.device,
                ),
            )
        if not args.no_logit_diff_loss:
            print(
                "\tadversarial evaluation (logit-diff loss), ASR:",
                run_adversarial_evaluation(
                    classifier,
                    test_loader,
                    "logit-diff",
                    args.adversarial_attack,
                    args.n_samples,
                    args.device,
                ),
            )

        max_eps_adversarial_attack_settings = copy.deepcopy(args.adversarial_attack)
        # set epsilon to 0.5
        # then rescale the step size so that it relatively to epsilon stays the same
        max_eps_adversarial_attack_settings.epsilon = 0.50
        max_eps_adversarial_attack_settings.step_size = (
            args.adversarial_attack.step_size / args.adversarial_attack.epsilon * 0.5
        )
        if not args.no_ce_loss:
            print(
                "\tadversarial evaluation (ce loss, eps = 0.5), ASR:",
                run_adversarial_evaluation(
                    classifier,
                    test_loader,
                    "ce",
                    max_eps_adversarial_attack_settings,
                    args.n_samples,
                    args.device,
                ),
            )
        if not args.no_logit_diff_loss:
            print(
                "\tadversarial evaluation (logit-diff loss, eps = 0.5), ASR:",
                run_adversarial_evaluation(
                    classifier,
                    test_loader,
                    "logit-diff",
                    max_eps_adversarial_attack_settings,
                    args.n_samples,
                    args.device,
                ),
            )

    if args.decision_boundary_binarization is not None:
        print("decision boundary binarization:")
        if not args.no_ce_loss:
            print(
                run_decision_boundary_binarization(
                    classifier,
                    test_loader,
                    "ce",
                    args.decision_boundary_binarization,
                    args.n_samples,
                    args.device,
                    args.batch_size,
                    "interior-vs-boundary discrimination (ce loss)",
                    args.dbl_sample_from_corners,
                )
            )
        if not args.no_logit_diff_loss:
            print(
                run_decision_boundary_binarization(
                    classifier,
                    test_loader,
                    "logit-diff",
                    args.decision_boundary_binarization,
                    args.n_samples,
                    args.device,
                    args.batch_size,
                    "interior-vs-boundary discrimination (logit-diff loss)",
                    args.dbl_sample_from_corners,
                )
            )


def run_clean_evaluation(
    classifier: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    device: str,
    n_classes: int = 10,
) -> Tuple[float, List[float], np.ndarray, List[int]]:
    """
    Perform evaluation of classifier on clean data.

    Args:
        classifier: Classifier to evaluate.
        test_loader: Dataloader to perform evaluation on.
        device: torch device
        n_classes: Number of classes in the dataset.
    Returns
        Accuracy, Accuracy per class, Correctly classified per sample,
        Histogram of predicted labels
    """
    n_correct = 0
    n_total = 0
    class_histogram_correct = {}
    class_histogram_total = {}
    class_histogram_predicted = {}

    if n_classes is not None:
        for i in range(n_classes):
            class_histogram_correct[i] = 0
            class_histogram_total[i] = 0
            class_histogram_predicted[i] = 0
    correctly_classified = []
    pbar = tqdm.tqdm(test_loader, leave=False)
    for x, y in pbar:
        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            y_pred = classifier(x).argmax(-1)
            n_correct += (y_pred == y).long().sum().item()
            n_total += len(x)
            correctly_classified.append((y_pred == y).detach().cpu())

            for y_, y_pred_ in zip(
                y.detach().cpu().numpy(), y_pred.detach().cpu().numpy()
            ):
                if y_ not in class_histogram_correct:
                    class_histogram_correct[y_] = 0
                class_histogram_correct[y_] += int(y_ == y_pred_)
                if y_ not in class_histogram_total:
                    class_histogram_total[y_] = 0
                class_histogram_total[y_] += 1
                if y_pred_ not in class_histogram_predicted:
                    class_histogram_predicted[y_pred_] = 0
                class_histogram_predicted[y_pred_] += 1
        pbar.set_description(f"Accuracy = {n_correct / n_total:.4f}")
    correctly_classified = torch.cat(correctly_classified).numpy()
    class_histogram_correct = [
        class_histogram_correct[k] for k in sorted(class_histogram_correct.keys())
    ]
    class_histogram_total = [
        class_histogram_total[k] for k in sorted(class_histogram_total.keys())
    ]

    class_histogram_accuracy = [
        a / b if b > 0 else np.nan
        for a, b in zip(class_histogram_correct, class_histogram_total)
    ]

    class_histogram_predicted = [
        class_histogram_predicted[k] for k in sorted(class_histogram_predicted.keys())
    ]

    return (
        n_correct / n_total,
        class_histogram_accuracy,
        correctly_classified,
        class_histogram_predicted,
    )


def run_decision_boundary_binarization(
    classifier: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss: LossType,
    linearization_settings: aut.DecisionBoundaryBinarizationSettings,
    n_samples: int,
    device: str,
    batch_size: int,
    title: str = "interior-vs-boundary discimination",
    sample_training_data_from_corners: bool = False,
) -> float:
    """Perform the binarization test for a classifier.

    Args:
        classifier: Classifier to evaluate.
        test_loader: Test dataloader.
        loss: Loss to use in the adversarial attack during the test.
        linearization_settings: Settings of the test.
        n_samples: Number of samples to perform test on.
        device: Torch device.
        batch_size: Batch size.
        title: Name of the experiment that will be shown in log.
        sample_training_data_from_corners: Sample boundary samples from
            corners or surfaces.
    Returns:
        String summarizing the results of the test.
    """
    def attack_fn(
        model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, attack_kwargs
    ):
        result = run_adversarial_evaluation(
            model,
            data_loader,
            loss,
            linearization_settings.adversarial_attack_settings,
            n_samples=1,
            device=device,
            return_samples=True,
            n_classes=2,
            early_stopping=True,
        )
        # return ASR, (x_adv, logits(x_adv))
        return result[0], (result[1][1], result[1][2])

    scores_logit_differences_and_validation_accuracies_and_asr = dbl.interior_boundary_discrimination_attack(
        classifier,
        test_loader,
        attack_fn,
        linearization_settings,
        n_samples,
        device,
        n_samples_evaluation=200,  # was set to n_samples
        n_samples_asr_evaluation=linearization_settings.adversarial_attack_settings.n_steps,
        rescale_logits="adaptive",
        decision_boundary_closeness=0.9999,
        sample_training_data_from_corners=sample_training_data_from_corners,
        batch_size=batch_size,
    )

    return dbl.format_result(
        scores_logit_differences_and_validation_accuracies_and_asr,
        n_samples,
        title=title,
    )


def run_adversarial_evaluation(
    classifier: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    loss: LossType,
    adversarial_attack_settings: aut.AdversarialAttackSettings,
    n_samples: int,
    device: str,
    randomly_targeted: bool = False,
    n_classes: int = 10,
    return_samples: bool = False,
    early_stopping: bool = True,
) -> Tuple[float, ...]:
    """
    Perform an adversarial evaluation of a classifier.

    Args:
        classifier: Classifier to evaluate.
        test_loader: Test dataloader.
        loss: Loss to use in adversarial attack.
        adversarial_attack_settings: Settings of adversarial evaluation.
        n_samples: Number of samples to evaluate robustness on.
        device: Torch device:
        randomly_targeted: Whether to use random targets for attack.
        n_classes: Number of classes in the dataset (relevant for random targets)
        return_samples: Returns clean and perturbed samples?
        early_stopping: Stop once all samples have successfully been attacked
    Returns:
        Either only Attack Success Rate (ASR) or Tuple containing ASR and
        clean/perturbed samples as well as their logits.
    """

    loss_per_sample = adversarial_attack_settings.attack == "kwta"

    if loss == "ce":
        sign = 1 if randomly_targeted else -1
        if loss_per_sample:
            reduction = "none"
        else:
            reduction = "sum"
        loss_fn = lambda x, y: sign * F.cross_entropy(x, y, reduction=reduction)
    elif loss == "logit-diff":
        sign = -1 if randomly_targeted else 1

        def loss_fn(logits, y):
            gt_logits = logits[range(len(y)), y]
            other = torch.max(
                logits - 2 * torch.max(logits) * F.one_hot(y, logits.shape[-1]), -1
            )[0]
            value = sign * (gt_logits - other)
            if not loss_per_sample:
                value = value.sum()
            return value

    if adversarial_attack_settings.attack == "kwta":
        if loss != "logit-diff":
            warnings.warn(
                "Adaptive attack for kWTA originally uses logit "
                "differences and not CE loss",
                RuntimeWarning,
            )

    n_attacked = 0
    attack_successful = []
    clean_samples = []
    perturbed_samples = []
    clean_or_target_labels = []
    predicted_logits = []
    for x, y in test_loader:
        x = x[: max(1, min(len(x), n_samples - n_attacked))]
        y = y[: max(1, min(len(y), n_samples - n_attacked))]

        x = x.to(device)
        y = y.to(device)

        if randomly_targeted:
            y = (y + torch.randint_like(y, 0, n_classes)) % n_classes
        if adversarial_attack_settings.attack == "pgd":
            x_adv = pgd.general_pgd(
                loss_fn=lambda x, y: loss_fn(classifier(x), y),
                is_adversarial_fn=lambda x, y: classifier(x).argmax(-1) == y
                if randomly_targeted
                else classifier(x).argmax(-1) != y,
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                step_size=adversarial_attack_settings.step_size,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                early_stopping=early_stopping,
                n_averaging_steps=adversarial_attack_settings.n_averages,
                random_start=adversarial_attack_settings.random_start,
            )[0]
        elif adversarial_attack_settings.attack == "autopgd":
            temp = autopgd.auto_pgd(
                model=classifier,
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                targeted=randomly_targeted,
                n_averaging_steps=adversarial_attack_settings.n_averages,
            )
            x_adv = temp[0]
            if randomly_targeted:
                y = temp[-1]
        elif adversarial_attack_settings.attack == "autopgd+":
            temp = autopgd.auto_pgd(
                model=classifier,
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                # from https://github.com/fra31/auto-attack/blob/
                # 6482e4d6fbeeb51ae9585c41b16d50d14576aadc/autoattack/
                # autoattack.py#L281
                n_restarts=4,
                targeted=randomly_targeted,
                n_averaging_steps=adversarial_attack_settings.n_averages,
            )
            x_adv = temp[0]
            if randomly_targeted:
                y = temp[-1]
        elif adversarial_attack_settings.attack == "fab":
            temp = fab.fab(
                model=classifier,
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                targeted=randomly_targeted,
                n_restarts=5,
            )
            x_adv = temp[0]
            if randomly_targeted:
                y = temp[-1]
        elif adversarial_attack_settings.attack == "kwta":
            x_adv = adaptive_kwta_attack.gradient_estimator_pgd(
                model=classifier,
                loss_fn=lambda x, y: loss_fn(classifier(x), y),
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                step_size=adversarial_attack_settings.step_size,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                random_start=True,
                early_stopping=early_stopping,
                targeted=randomly_targeted,
            )[0]
        elif adversarial_attack_settings.attack == "thermometer-lspgd":
            if hasattr(classifier, "l"):
                l = classifier.l
            else:
                l = 16
                warnings.warn(
                    "Could not determine thermometer parameter l; "
                    "using default of 16",
                    RuntimeWarning,
                )

            x_adv = thermometer_ls_pgd.general_thermometer_ls_pgd(
                loss_fn=lambda x, y: loss_fn(classifier(x, skip_encoder=True), y),
                is_adversarial_fn=lambda x, y: classifier(x).argmax(-1) == y
                if randomly_targeted
                else classifier(x).argmax(-1) != y,
                x=x,
                y=y,
                n_steps=adversarial_attack_settings.n_steps,
                step_size=adversarial_attack_settings.step_size,
                epsilon=adversarial_attack_settings.epsilon,
                norm=adversarial_attack_settings.norm,
                random_start=True,
                early_stopping=early_stopping,
                temperature=1.0,
                annealing_factor=1.0,  # 1.0/1.2,
                n_restarts=0,
                l=l,
            )[0]
        else:
            raise ValueError(
                f"Unknown adversarial attack "
                f"({adversarial_attack_settings.attack})."
            )

        with torch.no_grad():
            logits = classifier(x_adv)
            if randomly_targeted:
                correctly_classified = logits.argmax(-1) == y
                attack_successful += (
                    correctly_classified.cpu().detach().numpy().tolist()
                )
            else:
                incorrectly_classified = logits.argmax(-1) != y
                attack_successful += (
                    incorrectly_classified.cpu().detach().numpy().tolist()
                )

        clean_samples.append(x.cpu())
        perturbed_samples.append(x_adv.cpu())
        clean_or_target_labels.append(y.cpu())
        predicted_logits.append(logits.cpu())

        n_attacked += len(x)

        if n_attacked >= n_samples:
            break
    attack_successful = np.array(attack_successful)
    clean_samples = np.concatenate(clean_samples, 0)
    perturbed_samples = np.concatenate(perturbed_samples, 0)
    clean_or_target_labels = np.concatenate(clean_or_target_labels, 0)
    predicted_logits = np.concatenate(predicted_logits, 0)

    attack_successful = attack_successful[:n_samples]
    clean_samples = clean_samples[:n_samples]
    perturbed_samples = perturbed_samples[:n_samples]
    clean_or_target_labels = clean_or_target_labels[:n_samples]
    predicted_logits = predicted_logits[:n_samples]

    result = [np.mean(attack_successful).astype(np.float32)]

    if return_samples:
        result += [
            (clean_samples, perturbed_samples, predicted_logits, clean_or_target_labels)
        ]

    return tuple(result)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning, module="torch")
    main()
