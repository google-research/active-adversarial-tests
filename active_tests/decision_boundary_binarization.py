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

import functools
import traceback
import sys
import warnings
from typing import Callable
from typing import List

from torch.utils.data import SequentialSampler
from typing_extensions import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
import torch
import torch.utils.data
from torch.utils.data import DataLoader
import tqdm

import argparse_utils as aut
import networks
import utils as ut

__all__ = ["interior_boundary_discrimination_attack"]


LogitRescalingType = Optional[
    Union[Literal["fixed"], Literal["adaptive"], Literal["tight"]]
]
SolutionGoodnessType = Union[Literal["perfect"], Literal["good"], None]
OptimizerType = Union[
    Literal["sklearn"], Literal["sklearn-svm"], Literal["sgd"], Literal["adam"]
]


class __KwargsSequential(torch.nn.Sequential):
    """
    Modification of a torch.nn.Sequential model that allows kwargs in the
    forward pass. These will be passed to the first module of the network.
    """

    def forward(self, x, **kwargs):
        for idx, module in enumerate(self):
            if idx == 0:
                x = module(x, **kwargs)
            else:
                x = module(x)
        return x


def _create_raw_data(
    x: torch.Tensor,
    y: torch.Tensor,
    n_inner_points: int,
    n_boundary_points: int,
    n_boundary_adversarial_points: int,
    n_far_off_boundary_points: int,
    n_far_off_adversarial_points: int,
    batch_size: int,
    fill_batches_for_verification: bool,
    verify_valid_inner_input_data_fn: Optional[Callable],
    verify_valid_boundary_input_data_fn: Optional[Callable],
    get_boundary_adversarials_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, int, float], torch.Tensor]
    ],
    device: str,
    epsilon: float,
    norm: ut.NormType,
    n_boundary_classes: int = 1,
    eta: float = 0.95,
    xi: float = 1.50,
    include_original: bool = True,
    rejection_resampling_max_repetitions: int = 10,
    sample_boundary_from_corners: bool = False,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    """Creates the raw training data in image space. Label 0 corresponds to
    inner points and label 1 to boundary points."""

    def _sample_inner_points(n_samples):
        # We want to keep the original data point -> only generate n-1 new points
        x_inner = torch.repeat_interleave(torch.unsqueeze(x, 0), n_samples, 0)

        if norm == "linf":
            # Random noise in [-1, 1].
            delta_inner = 2 * torch.rand_like(x_inner) - 1.0
            # Random noise in [-eps*eta, eps*eta]
            delta_inner = delta_inner * eta * epsilon
        elif norm == "l2":
            # sample uniformly in ball with max radius eta*epsilon
            delta_inner = torch.randn_like(x_inner)
            delta_inner /= torch.norm(delta_inner, p=2, dim=[1, 2, 3], keepdim=True)
            delta_inner *= torch.pow(
                torch.rand(
                    (len(delta_inner), 1, 1, 1),
                    dtype=delta_inner.dtype,
                    device=delta_inner.device,
                ),
                1 / np.prod(x.shape[1:]),
            )
            delta_inner *= epsilon * eta
        else:
            raise ValueError

        if norm != "linf":
            _, delta_inner = ut.clipping_aware_rescaling(
                x_inner,
                delta_inner,
                target_distance=epsilon * eta,
                norm=norm,
                shrinking=True,
                return_delta=True,
            )

        x_inner = torch.clamp(x_inner + delta_inner, 0, 1)
        y_inner = torch.zeros(len(x_inner), dtype=torch.long, device=device)

        return x_inner, y_inner

    def _sample_boundary_points(n_samples, distance=epsilon):
        x_boundary = torch.unsqueeze(x, 0).repeat(
            tuple([n_samples] + [1] * len(x.shape))
        )

        if norm == "linf":
            if sample_boundary_from_corners:
                delta_boundary = torch.randint(
                    0,
                    2,
                    size=x_boundary.shape,
                    device=x_boundary.device,
                    dtype=x_boundary.dtype,
                )
                delta_boundary = (delta_boundary * 2.0 - 1.0) * distance
            else:
                delta_boundary = (torch.rand_like(x_boundary) * 2.0 - 1.0) * distance
        elif norm == "l2":
            # sample uniformly on sphere with radius epsilon
            delta_boundary = torch.randn_like(x_boundary)
            delta_boundary /= torch.norm(
                delta_boundary, p=2, dim=[1, 2, 3], keepdim=True
            )
            delta_boundary *= distance
        else:
            raise ValueError

        if not sample_boundary_from_corners:
            _, delta_boundary = ut.clipping_aware_rescaling(
                x_boundary,
                delta_boundary,
                target_distance=distance,
                norm=norm,
                growing=True,
                shrinking=True,
                return_delta=True,
            )

        x_boundary = torch.clamp(x_boundary + delta_boundary, 0, 1)
        y_boundary = torch.ones(len(x_boundary), dtype=torch.long, device=device)

        return x_boundary, y_boundary

    def _create_boundary_data():
        # TODO(zimmerrol): Extend this logic for multiple boundary classes
        n_random_boundary_samples = n_boundary_points - n_boundary_adversarial_points
        if n_random_boundary_samples == 0:
            x_random, y_random = None, None
        else:
            if verify_valid_boundary_input_data_fn is None:
                x_random, y_random = _sample_boundary_points(n_random_boundary_samples)
            else:
                x_random, y_random = _rejection_resampling(
                    _sample_boundary_points,
                    n_random_boundary_samples,
                    verify_valid_boundary_input_data_fn,
                    n_repetitions=rejection_resampling_max_repetitions,
                )

        if n_random_boundary_samples == n_boundary_points:
            # do not have to add any special adversarial points anymore
            x_total, y_total = x_random, y_random
        else:
            x_adv = get_boundary_adversarials_fn(
                x.clone(), y, n_boundary_adversarial_points, epsilon
            )
            y_adv = torch.ones(len(x_adv), dtype=y.dtype, device=y.device)
            if x_random is not None:
                x_total = torch.cat((x_random, x_adv))
                y_total = torch.cat((y_random, y_adv))
            else:
                x_total, y_total = x_adv, y_adv

        if n_boundary_classes > 1:
            raise NotImplementedError("n_boundary_classes > 1 is not yet implemented.")

        if n_far_off_boundary_points > 0:
            # add examples that have magnitude larger than epsilon but can be used
            # e.g., by logit matching attacks as a reference point
            n_random_far_off_samples = (
                n_far_off_boundary_points - n_far_off_adversarial_points
            )
            if n_random_boundary_samples == 0:
                x_faroff_random, y_faroff_random = None, None
            else:
                if verify_valid_boundary_input_data_fn is None:
                    x_faroff_random, y_faroff_random = _sample_boundary_points(
                        n_random_far_off_samples * n_boundary_classes,
                        distance=xi * epsilon,
                    )
                else:
                    x_faroff_random, y_faroff_random = _rejection_resampling(
                        functools.partial(
                            _sample_boundary_points, distance=xi * epsilon
                        ),
                        n_random_far_off_samples * n_boundary_classes,
                        verify_valid_boundary_input_data_fn,
                        n_repetitions=rejection_resampling_max_repetitions,
                    )
                if n_boundary_classes > 1:
                    raise NotImplementedError(
                        "n_boundary_classes > 1 is not yet implemented."
                    )

            if n_far_off_adversarial_points > 0:
                x_faroff_adv = get_boundary_adversarials_fn(
                    x.clone(), y, n_far_off_adversarial_points, epsilon
                )
                y_faroff_adv = torch.ones(
                    len(x_faroff_adv), dtype=y.dtype, device=y.device
                )
                if x_faroff_random is not None:
                    x_faroff = torch.cat((x_faroff_random, x_faroff_adv))
                    y_faroff = torch.cat((y_faroff_random, y_faroff_adv))
                else:
                    x_faroff, y_faroff = x_faroff_adv, y_faroff_adv
            else:
                x_faroff, y_faroff = x_faroff_random, y_faroff_random

            x_total = torch.cat((x_total, x_faroff))
            y_total = torch.cat((y_total, y_faroff))

        return x_total, y_total

    def _create_inner_data():
        if include_original:
            n_random_points = n_inner_points - 1
        else:
            n_random_points = n_inner_points

        if n_random_points > 0:
            if verify_valid_inner_input_data_fn is None:
                x_random, y_random = _sample_inner_points(n_inner_points)
            else:
                x_random, y_random = _rejection_resampling(
                    _sample_inner_points,
                    n_inner_points,
                    verify_valid_inner_input_data_fn,
                    n_repetitions=rejection_resampling_max_repetitions,
                )

            if include_original:
                x_total = torch.cat((torch.unsqueeze(x, 0), x_random))
                y_total = torch.zeros(
                    len(y_random) + 1, dtype=y_random.dtype, device=y_random.device
                )
            else:
                x_total, y_total = x_random, y_random
        else:
            x_total = torch.unsqueeze(x, 0)
            y_total = torch.zeros(1, dtype=y_boundary.dtype, device=y_boundary.device)

        return x_total, y_total

    def _rejection_resampling(
        sampling_fn, n_samples, verify_valid_input_data_fn, n_repetitions=10
    ):
        """Resample & replace until all samples returned by the sampling_fn are
        valid according to verify_valid_input_data_fn."""
        # do not waste time but running a non-full batch
        if fill_batches_for_verification:
            n_sampling_samples = max(n_samples, batch_size)
        else:
            n_sampling_samples = n_samples

        x, y = sampling_fn(n_sampling_samples)
        x_valid_mask = verify_valid_input_data_fn(x)
        for i in range(n_repetitions + 1):
            if np.sum(x_valid_mask) >= n_samples:
                # found enough samples
                # now restrict x to the valid samples
                # and x and y such that their length matches n_samples
                x = x[x_valid_mask]
                x = x[:n_samples]
                y = y[:n_samples]
                return x, y

            if i == n_repetitions:
                raise RuntimeError(
                    f"Rejection resampling failed after {n_repetitions} " f"rounds."
                )

            # check how many samples to be replaced
            n_x_invalid = len(x_valid_mask) - np.sum(x_valid_mask)
            # generate new samples
            c = sampling_fn(n_sampling_samples)[0]
            # check how many of them are valid and are needed
            c_valid_mask = verify_valid_input_data_fn(c)
            c = c[c_valid_mask][:n_x_invalid]
            c_valid_mask = c_valid_mask[c_valid_mask][:n_x_invalid]
            n_x_invalid_c_valid = min(n_x_invalid, len(c))
            # replace samples and update the mask
            x[~x_valid_mask][:n_x_invalid_c_valid] = c
            x_valid_mask[~x_valid_mask][:n_x_invalid_c_valid] = c_valid_mask

    if not n_inner_points > 0:
        raise ValueError("n_inner_points must be > 0.")
    if not n_boundary_points > 0:
        raise ValueError("n_boundary_points must be > 0.")
    if not n_boundary_classes == 1:
        raise NotImplementedError("More than 1 boundary class is not yet supported.")
    if not n_far_off_adversarial_points >= 0:
        raise ValueError("n_far_off_adversarial_points must not be negative.")
    if not n_far_off_boundary_points >= 0:
        raise ValueError("n_far_off_boundary_points must not be negative.")
    if not n_boundary_adversarial_points >= 0:
        raise ValueError("n_boundary_adversarial_points must not be negative.")

    x = x.to(device)
    y = y.to(device)
    (x_boundary, y_boundary) = _create_boundary_data()
    (x_inner, y_inner) = _create_inner_data()

    x = torch.cat((x_inner, x_boundary))
    y = torch.cat((y_inner, y_boundary))

    dataset = torch.utils.data.TensorDataset(x, y)
    dataset_boundary = torch.utils.data.TensorDataset(x_boundary, y_boundary)
    dataset_inner = torch.utils.data.TensorDataset(x_inner, y_inner)

    dataloader = torch.utils.data.DataLoader(
        dataset, shuffle=False, batch_size=batch_size
    )
    dataloader_boundary = torch.utils.data.DataLoader(
        dataset_boundary, shuffle=False, batch_size=batch_size
    )
    dataloader_inner = torch.utils.data.DataLoader(
        dataset_inner, shuffle=False, batch_size=batch_size
    )

    return dataloader, dataloader_boundary, dataloader_inner, len(x)


def _get_data_features_and_maybe_logits(
    classifier: Callable,
    raw_data_loader: torch.utils.data.DataLoader,
    get_logits: bool,
    device: str,
    include_raw_data: bool = False,
    raw_data_loader_boundary: Optional[torch.utils.data.DataLoader] = None,
    raw_data_loader_inner: Optional[torch.utils.data.DataLoader] = None,
    n_repetitions_boundary: Optional[int] = None,
    n_repetitions_inner: Optional[int] = None,
) -> Tuple[torch.utils.data.DataLoader, torch.Tensor, int, int]:
    """
    Collects the intermediate features for a classifier and creates a new data
    loader consisting only of these features.

    Args:
        classifier: Classifier to use as a feature extractor.
        raw_data_loader: Data loader that contains images which
            shall be mapped to intermediate features.
        get_logits: Extract not only features but also logits
        device: torch device.
        include_raw_data: Include raw images in the data loader.
    Returns:
        Data loader mapping intermediate features to class labels.
    """
    all_features = []
    all_logits = [] if get_logits else None
    all_labels = []
    all_images = []

    def _process_dataloader(dataloader: DataLoader):
        with torch.no_grad():
            for x, y in dataloader:
                x_ = x.to(device)
                if get_logits:
                    features, logits = classifier(x_, features_and_logits=True)
                    all_logits.append(logits)
                else:
                    features = classifier(x_, features_only=True)

                all_features.append(features.detach())
                all_labels.append(y)
                if include_raw_data:
                    all_images.append(x)

    _process_dataloader(raw_data_loader)

    if n_repetitions_boundary is not None:
        raw_data_loader_boundary = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.repeat_interleave(
                    raw_data_loader_boundary.dataset.tensors[0],
                    n_repetitions_boundary,
                    0,
                ),
                torch.repeat_interleave(
                    raw_data_loader_boundary.dataset.tensors[1],
                    n_repetitions_boundary,
                    0,
                ),
            ),
            batch_size=raw_data_loader_boundary.batch_size,
        )
        _process_dataloader(raw_data_loader_boundary)
    if n_repetitions_inner is not None:
        raw_data_loader_inner = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.repeat_interleave(
                    raw_data_loader_inner.dataset.tensors[0], n_repetitions_inner, 0
                ),
                torch.repeat_interleave(
                    raw_data_loader_inner.dataset.tensors[1], n_repetitions_inner, 0
                ),
            ),
            batch_size=raw_data_loader_inner.batch_size,
        )
        _process_dataloader(raw_data_loader_inner)

    all_features = torch.cat(all_features, 0)

    if get_logits:
        all_logits = torch.cat(all_logits, 0)
    all_labels = torch.cat(all_labels, 0)
    if include_raw_data:
        all_images = torch.cat(all_images)

    if len(all_features.shape) > 2:
        warnings.warn(
            f"Features are not vectors but higher dimensional "
            f"({len(all_features.shape) - 1})"
        )

    if include_raw_data:
        dataset = torch.utils.data.TensorDataset(all_features, all_labels, all_images)
    else:
        dataset = torch.utils.data.TensorDataset(all_features, all_labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=not isinstance(raw_data_loader.sampler, SequentialSampler),
        batch_size=raw_data_loader.batch_size,
    )

    return dataloader, all_logits, all_features.shape[-1], all_features.shape[0]


def _train_logistic_regression_classifier(
    n_features: int,
    train_loader: DataLoader,
    classifier_logits: Optional[torch.Tensor],
    optimizer: OptimizerType,
    lr: float,
    device: str,
    n_classes: int = 2,
    rescale_logits: LogitRescalingType = "fixed",
    decision_boundary_closeness: Optional[float] = None,
    solution_goodness: SolutionGoodnessType = "perfect",
    class_weight: Optional[Union[Literal["balanced"], dict]] = None,
) -> torch.nn.Module:
    """
    Trains a logistic regression model.

    Args:
        n_features: Feature dimensionality.
        train_loader: Data loader containing the data to fit model on.
        classifier_logits: Logits of the underlying classifier; will be used for logit
          rescaling.
        optimizer: Type of optimizer to use.
        lr: Learning rate (only applies to explicit gradient-descent optimizer).
        device: torch device.
        rescale_logits: Rescale weights of model such that the logits have
          at most unit absolute magnitude.
        decision_boundary_closeness: (optional) The larger this value, the will
          the decision boundary be placed to the boundary sample(s).
    Returns:
        Logistic regression model.
    """

    if rescale_logits == "adaptive" and classifier_logits is None:
        raise ValueError("classifier_logits must be supplied for adaptive rescaling")

    def get_accuracy() -> Tuple[float, float, float]:
        """

        :return: (total accuracy, accuracy for inner samples, for outer samples)
        """
        # calculate accuracy
        n_correct = {0: 0, 1: 0}
        n_total = {0: 0, 1: 0}
        with torch.no_grad():
            for x, y in train_loader:
                x = x.to(device)
                logits = binary_classifier(x)
                for k in n_total.keys():
                    n_correct[k] += (
                        (logits.argmax(-1).cpu() == y.cpu())
                        .float()[y == k]
                        .sum()
                        .item()
                    )
                    n_total[k] += len(x[y == k])

        accuracy = (n_correct[0] + n_correct[1]) / (n_total[0] + n_total[1])
        accuracy_inner = n_correct[0] / n_total[0]
        accuracy_outer = n_correct[1] / n_total[1]

        return accuracy, accuracy_inner, accuracy_outer

    if not n_classes == 2:
        raise NotImplementedError("Currently only supports 1 boundary class")

    if optimizer.startswith("sklearn"):
        if optimizer == "sklearn":
            # Use logistic regression of sklearn to speed up fitting.
            regression = LogisticRegression(
                penalty="none",
                max_iter=max(1000, int(lr)),
                multi_class="multinomial",
                class_weight=class_weight,
            )
        elif optimizer == "sklearn-svm":
            # Since the problem should be perfectly possible to solve, C should not
            # have any effect.
            regression = LinearSVC(
                penalty="l2", C=10e5, max_iter=max(1000, int(lr)), multi_class="ovr"
            )
        else:
            raise ValueError("Invalid optimizer choice.")

        regression.fit(
            train_loader.dataset.tensors[0].cpu().numpy(),
            train_loader.dataset.tensors[1].cpu().numpy(),
        )

        binary_classifier = torch.nn.Linear(n_features, n_classes)

        binary_classifier.weight.data = torch.Tensor(
            np.concatenate((-regression.coef_, regression.coef_), 0)
        )
        binary_classifier.bias.data = torch.Tensor(
            np.concatenate((-regression.intercept_, regression.intercept_), 0)
        )

        binary_classifier = binary_classifier.to(device)

        accuracy, accuracy_inner, accuracy_outer = get_accuracy()

        if solution_goodness is not None and accuracy < 1.0:
            raise_error = solution_goodness == "perfect"
            raise_error |= (
                solution_goodness == "good"
                and accuracy_inner == 0
                or accuracy_outer == 0
            )

            message = (
                f"sklearn solver failed to find perfect solution, "
                f"Accuracy = {accuracy:.4f} instead of 1.0; "
                f"{accuracy_inner:.4f} and {accuracy_outer:.4f} for "
                f"inner and boundary points."
            )

            if raise_error:
                raise RuntimeError(message)
            else:
                warnings.warn(message)
    else:
        binary_classifier = torch.nn.Linear(n_features, n_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = {"sgd": torch.optim.SGD, "adam": torch.optim.Adam}[optimizer](
            lr=lr, params=binary_classifier.parameters()
        )
        epoch = 0
        while True:
            epoch += 1
            for x, y in train_loader:
                optimizer.zero_grad()
                x = x.to(device)
                y = y.to(device)
                logits = binary_classifier(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()

            if epoch > 50000:
                raise RuntimeError(
                    f"Could not fit binary discriminator in 50k iterations "
                    f"(Loss = {loss.item()}."
                    "Consider using different settings for the optimizer."
                )

            accuracy = get_accuracy()
            # stop training once perfect accuracy is achieved
            if accuracy == 1.0:
                break

    if rescale_logits is not None or decision_boundary_closeness is not None:
        # Get value range of binarized logits.
        with torch.no_grad():
            logits = binary_classifier(
                train_loader.dataset.tensors[0].to(device)
            ).detach()

        if decision_boundary_closeness is not None:
            logit_differences = logits[:, 0] - logits[:, 1]
            lowest_difference = np.min(logit_differences.cpu().numpy())
            binary_classifier.bias.data[0] -= (
                decision_boundary_closeness * lowest_difference / 2
            )
            binary_classifier.bias.data[1] += (
                decision_boundary_closeness * lowest_difference / 2
            )

        if rescale_logits is not None:
            binarized_logit_range = (
                logits.detach().cpu().numpy().max()
                - logits.detach().cpu().numpy().min()
            )

            if rescale_logits == "fixed":
                target_logit_range = 1.0
                logit_offset = 0.0
                logit_rescaling_factor = binarized_logit_range / target_logit_range
            elif rescale_logits == "adaptive":
                # Rescale the binarized logits such that they have the same value range
                # i.e., min and max value match.
                target_logit_range = (
                    classifier_logits.detach().cpu().numpy().max()
                    - classifier_logits.detach().cpu().numpy().min()
                )
                logit_rescaling_factor = binarized_logit_range / target_logit_range
                logit_offset = (
                    classifier_logits.detach().cpu().numpy().min()
                    - logits.detach().cpu().numpy().min() / logit_rescaling_factor
                )
            elif rescale_logits == "tight":
                # Rescale/shift weights such that the distance between the decision
                # boundary and the boundary data is small.

                # Calculate distance of boundary points to decision boundary.
                distances = binary_classifier(
                    train_loader.dataset.tensors[0].to(device)[
                        train_loader.dataset.tensors[1].to(device) != 0
                    ]
                )[:, 1:].cpu()
                min_distance = distances.min()
                # Move decision boundary close to true boundary points
                logit_rescaling_factor = 1.0
                logit_offset = torch.tensor(
                    [+0.999 * min_distance.item(), -0.999 * min_distance.item()],
                    device=device,
                )
            else:
                raise ValueError(f"Invalid value for rescale_logits: {rescale_logits}")

            binary_classifier.bias.data /= logit_rescaling_factor
            binary_classifier.bias.data += logit_offset
            binary_classifier.weight.data /= logit_rescaling_factor

    return binary_classifier


def _get_interior_boundary_discriminator_and_dataloaders(
    classifier: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    linearization_settings: aut.DecisionBoundaryBinarizationSettings,
    device: str,
    batch_size: int = 512,
    rescale_logits: LogitRescalingType = "fixed",
    n_samples_evaluation: int = 0,
    n_samples_asr_evaluation: int = 0,
    verify_valid_inner_training_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_boundary_training_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_inner_validation_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_boundary_validation_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    get_boundary_adversarials_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, float], np.ndarray]
    ] = None,
    fill_batches_for_verification: bool = False,
    far_off_distance: float = 1.25,
    rejection_resampling_max_repetitions: int = 10,
    train_classifier_fn: Callable[
        [int, DataLoader, DataLoader, torch.Tensor, str, LogitRescalingType],
        torch.nn.Module,
    ] = None,
    decision_boundary_closeness: Optional[float] = None,
    n_inference_repetitions_boundary: Optional[int] = None,
    n_inference_repetitions_inner: Optional[int] = None,
    relative_inner_boundary_gap: Optional[float] = 0.05,
    sample_training_data_from_corners: bool = False,
) -> Tuple[
    Tuple[torch.nn.Module, torch.nn.Module],
    Tuple[DataLoader, DataLoader],
    Tuple[float, float, float, float, bool, bool],
]:
    """
    Creates a number of perturbed images, obtains the features for these images
    and trains a linear, binary discriminator of these samples.

    Args:
        classifier: The classifier that will be used as a feature encoder.
        x: The single clean image to apply apply the method on.
        y: The ground-truth classification label of x.
        linearization_settings: How to construct the binary classifier.
        device: torch device
        batch_size: Max batch size allowed to use
        rescale_logits: Rescale weights of linear classifier such that logits
          have a max scale of 1
        n_samples_evaluation: Number of random samples to use for evaluation
        verify_valid_inner_training_data_fn: Can be used for e.g. detector-based defenses.
          Check whether all input points used for training/testing are actually valid
          and won't get filtered out be the model/detector.
        verify_valid_boundary_training_data_fn: See verify_valid_inner_training_data_fn but for boundary samples.
        verify_valid_inner_validation_data_fn: See
          verify_valid_inner_training_data_fn but for calculating the validation
          scores, i.e. the random ASR.
        verify_valid_boundary_validation_data_fn: See
          verify_valid_boundary_training_data_fn but for calculating the validation
          scores, i.e. the random ASR.
        get_boundary_adversarials_fn: If given, use this function to
          generate all but one of the boundary samples. This can be used for
          evaluating detector-based evaluation functions.
        fill_batches_for_verification: If computational cost of verification_fn
          does not depend on the batch size, set this to True.
        far_off_distance: Relative multiplier (in terms of epsilon) controlling
          the distance between clean and far off training samples.
        decision_boundary_closeness: (optional) The larger this value, the will
          the decision boundary be placed to the boundary sample(s).
        n_inference_repetitions_boundary: (optional) How often to repeat
          inference for boundary samples for obtaining their features.
        n_inference_repetitions_inner: (optional) How often to repeat
          inference for inner samples for obtaining their features.
        relative_inner_boundary_gap: (optional) Gap between interior and
          boundary data relative to epsilon (i.e. a value of 0 means boundary points
          can lie directly next to inner points)
        sample_training_data_from_corners: Sample training data from the corners
          of the epsilon ball; this setting is only possible when using linf norm.
    Returns:
        Tuple containing ((binary discriminator between interior and boundary points,
            binary readout only),
          (dataset of perturbed images, dataset of features of perturbed images),
          (validation accuracies of inner/boundary/boundary surface/boundary corner points,
           random attack success rate of surface/corner points))
    """

    if sample_training_data_from_corners and linearization_settings.norm != "linf":
        raise ValueError("Corners are only defined for linf norm.")

    # Check if model is compatible with this check.
    try:
        with torch.no_grad():
            if rescale_logits == "adaptive":
                classifier(
                    torch.ones((1, 1, 1, 1), device=device), features_and_logits=True
                )
            else:
                classifier(torch.ones((1, 1, 1, 1), device=device), features_only=True)
    except TypeError as e:
        message = str(e)
        if "unexpected keyword argument 'features_only'" in message:
            raise ValueError(
                "model does not support `features_only` flag in forward pass."
            )
        if "unexpected keyword argument 'features_and_logits'" in message:
            raise ValueError(
                "model does not support `features_and_logits` flag in forward pass."
            )
    except Exception:
        pass

    (
        raw_train_loader,
        raw_train_loader_boundary,
        raw_train_loader_inner,
        n_raw_training_samples,
    ) = _create_raw_data(
        x,
        y,
        linearization_settings.n_inner_points,
        linearization_settings.n_boundary_points,
        linearization_settings.n_boundary_adversarial_points,
        linearization_settings.n_far_off_boundary_points,
        linearization_settings.n_far_off_adversarial_points,
        batch_size=batch_size,
        fill_batches_for_verification=fill_batches_for_verification,
        verify_valid_inner_input_data_fn=verify_valid_inner_training_data_fn,
        verify_valid_boundary_input_data_fn=verify_valid_boundary_training_data_fn,
        get_boundary_adversarials_fn=get_boundary_adversarials_fn,
        device=device,
        epsilon=linearization_settings.epsilon,
        norm=linearization_settings.norm,
        n_boundary_classes=1,
        include_original=True,
        xi=far_off_distance,
        eta=1.0 - relative_inner_boundary_gap,
        rejection_resampling_max_repetitions=rejection_resampling_max_repetitions,
        sample_boundary_from_corners=sample_training_data_from_corners,
    )

    # Get features to train binary classifier on.
    (
        train_loader,
        logits,
        n_features,
        n_training_samples,
    ) = _get_data_features_and_maybe_logits(
        classifier,
        raw_train_loader,
        rescale_logits == "adaptive",
        device,
        include_raw_data=False,
        n_repetitions_boundary=n_inference_repetitions_boundary,
        n_repetitions_inner=n_inference_repetitions_inner,
        raw_data_loader_boundary=raw_train_loader_boundary,
        raw_data_loader_inner=raw_train_loader_inner,
    )

    if not (n_features > n_training_samples):
        warnings.warn(
            f"Feature dimension ({n_features}) should not be smaller than the "
            f"number of training samples ({n_training_samples})",
            RuntimeWarning,
        )

    # finally train new binary classifier on features
    if train_classifier_fn is None:
        binary_classifier = _train_logistic_regression_classifier(
            n_features,
            train_loader,
            logits,
            linearization_settings.optimizer,
            linearization_settings.lr,
            device,
            class_weight=linearization_settings.class_weight,
            rescale_logits=rescale_logits,
            decision_boundary_closeness=decision_boundary_closeness,
        )
        linearized_model = __KwargsSequential(
            networks.Lambda(
                lambda x, **kwargs: classifier(x, features_only=True, **kwargs)
            ),
            binary_classifier,
        )

    else:
        binary_classifier = None
        linearized_model = train_classifier_fn(
            n_features,
            train_loader,
            raw_train_loader,
            logits,
            device,
            rescale_logits=rescale_logits,
        )

    # evaluate on another set of random samples (we are only interested in the
    # performance of points inside the epsilon ball)
    if n_samples_evaluation > 0:
        raw_validation_loader, _, _, _ = _create_raw_data(
            x,
            y,
            n_samples_evaluation,
            n_samples_evaluation,
            0,
            0,
            0,
            batch_size=batch_size,
            fill_batches_for_verification=fill_batches_for_verification,
            # TODO(zimmerrol): check if this makes sense. The motivation to remove this here
            #   was that the moved the check down to when the accuracy is calculated
            verify_valid_boundary_input_data_fn=None,
            verify_valid_inner_input_data_fn=None,
            # verify_valid_input_data_fn=verify_valid_input_validation_data_fn,
            get_boundary_adversarials_fn=get_boundary_adversarials_fn,
            device=device,
            epsilon=linearization_settings.epsilon,
            norm=linearization_settings.norm,
            n_boundary_classes=1,
            include_original=False,
            xi=far_off_distance,
            rejection_resampling_max_repetitions=rejection_resampling_max_repetitions,
            eta=1.0,
            sample_boundary_from_corners=False,
        )

        _, raw_validation_loader_corners, _, _ = _create_raw_data(
            x,
            y,
            1,
            n_samples_evaluation,
            0,
            0,
            0,
            batch_size=batch_size,
            fill_batches_for_verification=fill_batches_for_verification,
            verify_valid_boundary_input_data_fn=None,
            verify_valid_inner_input_data_fn=None,
            get_boundary_adversarials_fn=get_boundary_adversarials_fn,
            device=device,
            epsilon=linearization_settings.epsilon,
            norm=linearization_settings.norm,
            n_boundary_classes=1,
            include_original=False,
            xi=far_off_distance,
            rejection_resampling_max_repetitions=rejection_resampling_max_repetitions,
            eta=1.0,
            sample_boundary_from_corners=True,
        )

        raw_validation_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.cat(
                    (
                        raw_validation_loader.dataset.tensors[0],
                        raw_validation_loader_corners.dataset.tensors[0],
                    ),
                    0,
                ),
                torch.cat(
                    (
                        raw_validation_loader.dataset.tensors[1],
                        raw_validation_loader_corners.dataset.tensors[1],
                    ),
                    0,
                ),
            ),
            batch_size=raw_validation_loader.batch_size,
            shuffle=False,
        )

        # Get features to test binary classifier on.
        validation_loader, _, _, _ = _get_data_features_and_maybe_logits(
            classifier, raw_validation_loader, False, device, include_raw_data=True
        )

        inner_correctly_classified = []
        boundary_correctly_classified = []
        for it, (x_features, y, x_images) in enumerate(validation_loader):
            x_features = x_features.to(device)
            x_images = x_images.to(device)

            # If we use a custom train method use the raw images for validation as
            # it is possible that the classifier has no simple linear readout.

            # TODO(zimmerrol): also allow detector-like models here
            #  if the verify_valid_input_data_fn is used, this shouldn't be a
            #  concern anymore since all samples generated here have already passed
            #  the detector
            with torch.no_grad():
                if binary_classifier is not None:
                    y_pred = binary_classifier(x_features).argmax(-1).to("cpu")
                else:
                    y_pred = linearized_model(x_images).argmax(-1).to("cpu")
                # flag predictions for invalid data points such they are not
                # counted as correctly classified samples
                if verify_valid_inner_validation_data_fn is not None:
                    is_valid_input = verify_valid_inner_validation_data_fn(
                        x_images[y == 0]
                    )
                    y_pred[y == 0][~is_valid_input] = -1
                if verify_valid_boundary_validation_data_fn is not None:
                    is_valid_input = verify_valid_boundary_validation_data_fn(
                        x_images[y == 1]
                    )
                    y_pred[y == 1][~is_valid_input] = -1

            inner_correctly_classified += (y_pred[y == 0] == 0).numpy().tolist()
            boundary_correctly_classified += (y_pred[y == 1] == 1).numpy().tolist()

        inner_correctly_classified = np.array(inner_correctly_classified)
        boundary_correctly_classified = np.array(boundary_correctly_classified)

        validation_accuracy_inner = float(np.mean(inner_correctly_classified))
        validation_accuracy_boundary = float(np.mean(boundary_correctly_classified))

        validation_accuracy_boundary_surface = float(
            np.mean(boundary_correctly_classified[:n_samples_evaluation])
        )
        validation_accuracy_boundary_corners = float(
            np.mean(boundary_correctly_classified[n_samples_evaluation:])
        )
        random_attack_success_inner = (
            np.mean(inner_correctly_classified[:n_samples_asr_evaluation]) < 1.0
        )
        random_attack_success_boundary_surface = (
            np.mean(
                boundary_correctly_classified[:n_samples_evaluation][
                    :n_samples_asr_evaluation
                ]
            )
            > 0.0
        )
        random_attack_success_boundary_corners = (
            np.mean(
                boundary_correctly_classified[n_samples_evaluation:][
                    :n_samples_asr_evaluation
                ]
            )
            > 0.0
        )
        random_attack_success_boundary_corners = np.logical_or(
            random_attack_success_inner, random_attack_success_boundary_corners
        )
        random_attack_success_boundary_surface = np.logical_or(
            random_attack_success_inner, random_attack_success_boundary_surface
        )
        validation_accuracies_and_asr = (
            validation_accuracy_inner,
            validation_accuracy_boundary,
            validation_accuracy_boundary_surface,
            validation_accuracy_boundary_corners,
            random_attack_success_boundary_surface,
            random_attack_success_boundary_corners,
        )
    else:
        validation_accuracies_and_asr = None

    return (
        (linearized_model, binary_classifier),
        (raw_train_loader, train_loader),
        validation_accuracies_and_asr,
    )


def __wrap_assert_get_boundary_adversarials_fn(
    fn: Callable[[torch.Tensor, torch.Tensor, int, float], np.ndarray],
    norm: ut.NormType,
) -> Callable[[torch.Tensor, torch.Tensor, int, float], np.ndarray]:
    """Make sure adversarial examples really lie on the epsilon ball boundary
    (or are within a relative distance of 1%)."""

    def inner(x: torch.Tensor, y: torch.Tensor, n: int, epsilon: float):
        x_ = fn(x, y, n, epsilon)
        delta = (x_ - x).cpu()
        if norm == "linf":
            distance = torch.abs(delta).flatten(1).max(1)[0].numpy()
        elif norm in ("l2", "l1"):
            distance = torch.norm(
                delta, p=1 if norm == "l1" else 2, keepdim=False, dim=[1, 2, 3]
            ).numpy()
        else:
            raise ValueError(f"Unknown norm: {norm}")
        # TODO(zimmerrol): Verify whether 1% tolerance is sensible.
        assert np.isclose(distance, epsilon, atol=0.01 * epsilon), (
            f"Magnitude of boundary adversarial examples ({distance}) "
            f"does not match target distance ({epsilon}"
        )
        return x_

    return inner


def interior_boundary_discrimination_attack(
    classifier: torch.nn.Module,
    test_loader: torch.utils.data.DataLoader,
    attack_fn: Callable[
        [torch.nn.Module, torch.utils.data.DataLoader, dict],
        Tuple[np.ndarray, Tuple[torch.Tensor, torch.Tensor]],
    ],
    linearization_settings: aut.DecisionBoundaryBinarizationSettings,
    n_samples: int,
    device: str,
    batch_size: int = 512,
    rescale_logits: LogitRescalingType = "fixed",
    n_samples_evaluation: int = 0,
    n_samples_asr_evaluation: int = 0,
    verify_valid_inner_training_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_boundary_training_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_inner_validation_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    verify_valid_boundary_validation_data_fn: Optional[
        Callable[[torch.Tensor], np.ndarray]
    ] = None,
    get_boundary_adversarials_fn: Optional[
        Callable[[torch.Tensor, torch.Tensor, int, float], np.ndarray]
    ] = None,
    fill_batches_for_verification: bool = True,
    far_off_distance: float = 1.50,
    rejection_resampling_max_repetitions: int = 10,
    train_classifier_fn: Callable[
        [int, DataLoader, torch.Tensor, str, LogitRescalingType], torch.nn.Module
    ] = None,
    fail_on_exception: bool = False,
    decision_boundary_closeness: Optional[float] = None,
    n_inference_repetitions_boundary: Optional[int] = None,
    n_inference_repetitions_inner: Optional[int] = None,
    relative_inner_boundary_gap: Optional[float] = 0.05,
    sample_training_data_from_corners: bool = False,
) -> List[Tuple[bool, float, float, Tuple[float, float, float, float, bool, bool]]]:

    """
    Performs the binarization test. This means, replacing the last linear layer
    of the classifier with a binary classifier distinguishing between images
    of different perturbation magnitude.

    Args:
        classifier: The classifier that will be used as a feature encoder.
        test_loader: Data loader of the data to run the test on.
        attack_fn: Function performing an adversarial attack on a classifier and
            dataset passed as arguments.
        linearization_settings: How to construct the binarized classifier.
        n_samples: Number of samples to perform this test on.
        device: torch device
        batch_size: Max batch size allowed to use
        rescale_logits: Rescale weights of linear classifier such that logits
          have a max scale of 1
        n_samples_evaluation: Number of random samples to use for evaluation
        n_samples_asr_evaluation: Number of random samples used to calculate
          the ASR of a random attacker
        verify_valid_inner_training_data_fn: Can be used for e.g.
          detector-based defenses.
          Check whether all input points used for training/testing are actually valid
          and won't get filtered out be the model/detector.
        verify_valid_boundary_training_data_fn: See
          verify_valid_inner_training_data_fn but for boundary samples.
        verify_valid_inner_validation_data_fn: See
          verify_valid_inner_training_data_fn but for calculating the validation
          scores, i.e. the random ASR.
        verify_valid_boundary_validation_data_fn: See
          verify_valid_boundary_training_data_fn but for calculating the validation
          scores, i.e. the random ASR.
        get_boundary_adversarials_fn: If given, use this function to
          generate all but one of the boundary samples. This can be used for
          evaluating detector-based evaluation functions.
        fill_batches_for_verification: If computational cost of verification_fn
          does not depend on the batch size, set this to True.
        far_off_distance: Relative multiplier (in terms of epsilon) controlling
          the distance between clean and far off training samples.
        rejection_resampling_max_repetitions: How often to resample to satisfy
            constraints on training samples.
        train_classifier_fn: Callback that trains a readout classifier on a set of
            features.
        fail_on_exception: Raise exception if a single samples fails or keep
            running and report about this later.
        decision_boundary_closeness: (optional) The larger this value, the closer
          the decision boundary will be placed to the boundary sample(s).
        n_inference_repetitions_boundary: (optional) How often to repeat
          inference for boundary samples for obtaining their features.
        n_inference_repetitions_inner: (optional) How often to repeat
          inference for inner samples for obtaining their features.
        relative_inner_boundary_gap: (optional) Gap between interior and
          boundary data (in pixel space) relative to epsilon (i.e. a value of 0 means
          boundary points can lie directly next to inner points)
        sample_training_data_from_corners: Sample training data from the corners
          of the epsilon ball; this setting is only possible when using
    Returns:
        List containing tuples of (attack successful, logit diff of results of attack_fn,
        logit diff of best training sample, (validation accuracy inner,
        validation accuracy boundary, random ASR)))
    """

    if get_boundary_adversarials_fn is not None and (
        linearization_settings.n_boundary_adversarial_points == 0
        and linearization_settings.n_far_off_adversarial_points == 0
    ):
        warnings.warn(
            "get_boundary_adversarials_fn is set but number of boundary "
            "and far-off adversarial examples is set to 0",
            UserWarning,
        )

    results = []
    data_iterator = iter(test_loader)
    current_batch_x = None
    current_batch_y = None
    current_batch_index = 0

    if get_boundary_adversarials_fn is not None:
        # Make sure this function really returns boundary adversarials.
        get_boundary_adversarials_fn = __wrap_assert_get_boundary_adversarials_fn(
            get_boundary_adversarials_fn, linearization_settings.norm
        )

    # Show warnings only once
    warnings_shown_for_messages = []
    for i in tqdm.tqdm(range(n_samples)):
        if current_batch_x is None or current_batch_index == len(current_batch_x) - 1:
            try:
                # Only use input and label.
                current_batch_x, current_batch_y = next(data_iterator)
            except StopIteration:
                warnings.warn(
                    f"Could only gather {i} and not the "
                    f"{n_samples} requested samples."
                )
                break
            current_batch_index = 0
        else:
            current_batch_index += 1

        # Get current item/input data.
        x = current_batch_x[current_batch_index]
        y = current_batch_y[current_batch_index]

        setup_successful = False
        with warnings.catch_warnings(record=True) as ws:
            try:
                (
                    (binary_discriminator, binary_linear_layer),
                    (image_loader, feature_loader),
                    validation_accuracies,
                ) = _get_interior_boundary_discriminator_and_dataloaders(
                    classifier,
                    x,
                    y,
                    linearization_settings,
                    device,
                    rescale_logits=rescale_logits,
                    n_samples_evaluation=n_samples_evaluation,
                    n_samples_asr_evaluation=n_samples_asr_evaluation,
                    batch_size=batch_size,
                    verify_valid_inner_training_data_fn=verify_valid_inner_training_data_fn,
                    verify_valid_boundary_training_data_fn=verify_valid_boundary_training_data_fn,
                    verify_valid_inner_validation_data_fn=verify_valid_inner_validation_data_fn,
                    verify_valid_boundary_validation_data_fn=verify_valid_boundary_validation_data_fn,
                    get_boundary_adversarials_fn=get_boundary_adversarials_fn,
                    fill_batches_for_verification=fill_batches_for_verification,
                    far_off_distance=far_off_distance,
                    rejection_resampling_max_repetitions=rejection_resampling_max_repetitions,
                    train_classifier_fn=train_classifier_fn,
                    decision_boundary_closeness=decision_boundary_closeness,
                    n_inference_repetitions_boundary=n_inference_repetitions_boundary,
                    n_inference_repetitions_inner=n_inference_repetitions_inner,
                    relative_inner_boundary_gap=relative_inner_boundary_gap,
                    sample_training_data_from_corners=sample_training_data_from_corners,
                )
                setup_successful = True
            except RuntimeError as ex:
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname, lineno, fnname, code = traceback.extract_tb(exc_tb)[-1]
                if fail_on_exception:
                    raise ex
                else:
                    warnings.warn(f"Exception caught: {fname}:{lineno}({fnname}): {ex}")

        for w in ws:
            if str(w.message) not in warnings_shown_for_messages:
                warnings_shown_for_messages.append(str(w.message))
                warnings.warn(str(w.message), w.category)

        if not setup_successful:
            continue

        attack_loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(
                torch.unsqueeze(x, 0), torch.zeros(1, dtype=torch.long)
            ),
            shuffle=False,
            batch_size=1,
        )

        if linearization_settings.n_far_off_boundary_points == 0:
            attack_kwargs = {}
        else:
            attack_kwargs = dict(
                reference_points_x=image_loader.dataset.tensors[0][
                    -linearization_settings.n_far_off_boundary_points * 1 :
                ],
                reference_points_y=image_loader.dataset.tensors[1][
                    -linearization_settings.n_far_off_boundary_points * 1 :
                ],
            )

        with warnings.catch_warnings(record=True) as ws:
            attack_successful, (x_adv, logits_adv) = attack_fn(
                binary_discriminator, attack_loader, attack_kwargs
            )
        for w in ws:
            if str(w.message) not in warnings_shown_for_messages:
                warnings_shown_for_messages.append(str(w.message))
                warnings.warn(f"{w.filename}:{w.lineno}:{w.message}", w.category)

        logit_diff_adv = (logits_adv[:, 1] - logits_adv[:, 0]).item()

        # Now compare the result of the attack (x_adv) with the training samples
        # in terms of their confidence.
        # For this, first get logits of binary discriminator for data it was
        # trained on, but only do this for samples of the adversarial class (y = 1).
        logits_training = []
        if train_classifier_fn is None:
            for x, y in feature_loader:
                with torch.no_grad():
                    x = x[y == 1]
                    if len(x) == 0:
                        continue
                    logits_training.append(binary_linear_layer(x.to(device)).cpu())
        else:
            for x, y in image_loader:
                with torch.no_grad():
                    x = x[y == 1]
                    if len(x) == 0:
                        continue
                    logits_training.append(binary_discriminator(x.to(device)).cpu())
        logits_training = torch.cat(logits_training, 0)

        # Now get training sample with max confidence (alternatively we could also
        # just compute the distance to the linear boundary for all sample and pick
        # the one with max distance).
        logit_diff_training = torch.max(
            logits_training[:, 1] - logits_training[:, 0]
        ).item()

        result = (
            attack_successful,
            logit_diff_adv,
            logit_diff_training,
            validation_accuracies,
        )

        results.append(result)

    return results


def format_result(
    scores_logit_differences_and_validation_accuracies,
    n_samples,
    indent=0,
    title="interior-vs-boundary discrimination",
):
    """Formats the result of the interior-vs-boundary discriminator test"""
    if len(scores_logit_differences_and_validation_accuracies) == 0:
        test_result = (np.nan, np.nan, np.nan, np.nan)
    else:
        scores = [it[0] for it in scores_logit_differences_and_validation_accuracies]
        validation_scores = [
            it[3] for it in scores_logit_differences_and_validation_accuracies
        ]
        if validation_scores[0] is None:
            validation_scores = (np.nan, np.nan, np.nan)
        else:
            validation_scores = np.array(validation_scores)
            validation_scores = tuple(np.mean(validation_scores, 0))
        logit_differences = [
            (it[1], it[2]) for it in scores_logit_differences_and_validation_accuracies
        ]
        logit_differences = np.array(logit_differences)
        relative_performance = (logit_differences[:, 0] - logit_differences[:, 1]) / (
            logit_differences[:, 1] + 1e-12
        )

        test_result = (
            np.mean(scores),
            np.mean(relative_performance),
            np.std(relative_performance),
            validation_scores,
        )

    indent = "\t" * indent

    return (
        "{0}{1}, ASR: {2}\n”, "
        "{0}\tNormalized Logit-Difference-Improvement: {3} +- {4}\n"
        "{0}\tValidation Accuracy (I, B, BS, BC, R. ASR S, R. ASR C): {5}\n"
        "{0}\tSetup failed for {6}/{7} samples".format(
            indent,
            title,
            *test_result,
            n_samples - len(scores_logit_differences_and_validation_accuracies),
            n_samples,
        )
    )
