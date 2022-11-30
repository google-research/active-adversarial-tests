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

import random
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import torch
import torch.utils.data
from typing_extensions import Literal

NormType = Union[Literal["linf"], Literal["l2"], Literal["l1"]]
LabelRandomization = Tuple[Literal["random"], Literal["systematically"], Literal[None]]


def clipping_aware_rescaling_l2_torch(
    x0: torch.Tensor, delta: torch.Tensor, target_l2: Union[float, torch.Tensor]
):
    """Rescale delta such that it exactly lies target_l2 away in l2 from x0 after clipping.

    Adapted from https://github.com/jonasrauber/clipping-aware-rescaling/.

    Args:
        x0: Tensor containing the base samples.
        delta: Tensor containing the perturbations to add to x0.
        target_l2: Target l2 distance.

    Returns:
        Tensor containing required rescaling factors.
    """
    N = x0.shape[0]
    assert delta.shape[0] == N

    delta2 = delta.pow(2).reshape((N, -1))
    space = torch.where(delta >= 0, 1 - x0, x0).reshape((N, -1)).type(delta.dtype)
    f2 = space.pow(2) / torch.max(delta2, 1e-20 * torch.ones_like(delta2))
    f2_sorted, ks = torch.sort(f2, dim=-1)
    m = torch.cumsum(delta2.gather(dim=-1, index=ks.flip(dims=(1,))), dim=-1).flip(
        dims=(1,)
    )
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = torch.cat((f2_sorted[:, :1], dx), dim=-1)
    dy = m * dx
    y = torch.cumsum(dy, dim=-1)

    if not issubclass(type(target_l2), torch.Tensor):
        target_l2 = torch.ones(len(x0)).to(x0.device) * target_l2

    assert len(target_l2) == len(
        x0
    ), f"Inconsistent length of `target_l2`. Must have length {len(x0)}."
    assert len(target_l2.shape) == 1, "Inconsistent shape of `target_l2` (must be 1D)."

    target_l2 = target_l2.view((-1, 1, 1, 1)).expand(*x0.shape)
    target_l2 = target_l2.view(len(target_l2), -1)

    target_l2 = target_l2.type(delta.dtype)

    c = y >= target_l2**2

    # work-around to get first nonzero element in each row
    f = torch.arange(c.shape[-1], 0, -1, device=c.device)
    v, j = torch.max(c.long() * f, dim=-1)

    rows = torch.arange(0, N)

    eps2 = f2_sorted[rows, j] - (y[rows, j] - target_l2[rows, j] ** 2) / m[rows, j]
    # it can happen that for certain rows even the largest j is not large enough
    # (i.e. v == 0), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem)
    eps2 = torch.where(v == 0, f2_sorted[:, -1], eps2)

    eps = torch.sqrt(eps2)
    eps = eps.reshape((-1,) + (1,) * (len(x0.shape) - 1))

    return eps


def clipping_aware_rescaling_l1_torch(
    x0: torch.Tensor, delta: torch.Tensor, target_l1: Union[float, torch.Tensor]
):
    """Rescale delta such that it exactly lies target_l1 away in l1 from x0 after clipping.

    Adapted from https://github.com/jonasrauber/clipping-aware-rescaling/.

    Args:
        x0: Tensor containing the base samples.
        delta: Tensor containing the perturbations to add to x0.
        target_l1: Target l1 distance.

    Returns:
        Tensor containing required rescaling factors.
    """
    N = x0.shape[0]
    assert delta.shape[0] == N

    delta2 = delta.abs().reshape((N, -1))
    space = torch.where(delta >= 0, 1 - x0, x0).reshape((N, -1)).type(delta.dtype)
    f2 = space.abs() / torch.max(delta2, 1e-20 * torch.ones_like(delta2))
    f2_sorted, ks = torch.sort(f2, dim=-1)
    m = torch.cumsum(delta2.gather(dim=-1, index=ks.flip(dims=(1,))), dim=-1).flip(
        dims=(1,)
    )
    dx = f2_sorted[:, 1:] - f2_sorted[:, :-1]
    dx = torch.cat((f2_sorted[:, :1], dx), dim=-1)
    dy = m * dx
    y = torch.cumsum(dy, dim=-1)
    # c = y >= target_l2

    if not issubclass(type(target_l1), torch.Tensor):
        target_l1 = torch.ones(len(x0)).to(x0.device) * target_l1

    assert len(target_l1) == len(
        x0
    ), f"Inconsistent length of `target_l2`. Must have length {len(x0)}."
    assert len(target_l1.shape) == 1, "Inconsistent shape of `target_l2` (must be 1D)."

    target_l1 = target_l1.view((-1, 1, 1, 1)).expand(*x0.shape)
    target_l1 = target_l1.view(len(target_l1), -1)

    target_l1 = target_l1.type(delta.dtype)

    c = y >= target_l1

    # Work-around to get first nonzero element in each row.
    f = torch.arange(c.shape[-1], 0, -1, device=c.device)
    v, j = torch.max(c.long() * f, dim=-1)

    rows = torch.arange(0, N)

    eps2 = f2_sorted[rows, j] - (y[rows, j] - target_l1[rows, j]) / m[rows, j]
    # It can happen that for certain rows even the largest j is not large enough
    # (i.e. v == 0), then we will just use it (without any correction) as it's
    # the best we can do (this should also be the only cases where m[j] can be
    # 0 and they are thus not a problem).
    eps = torch.where(v == 0, f2_sorted[:, -1], eps2)

    eps = eps.reshape((-1,) + (1,) * (len(x0.shape) - 1))
    return eps


def clipping_aware_rescaling_linf_torch(
    x0: torch.Tensor, delta: torch.Tensor, target_linf: Union[float, torch.Tensor]
):
    """Rescale delta such that it exactly lies target_linf away in l2inf from x0 after clipping.

    Adapted from https://github.com/jonasrauber/clipping-aware-rescaling/.

    Args:
        x0: Tensor containing the base samples.
        delta: Tensor containing the perturbations to add to x0.
        target_linf: Target l2 distance.

    Returns:
        Tensor containing required rescaling factors.
    """
    N = x0.shape[0]
    assert delta.shape[0] == N

    if not issubclass(type(target_linf), torch.Tensor):
        target_linf = torch.ones(len(x0)).to(x0.device) * target_linf

    assert len(target_linf) == len(
        x0
    ), f"Inconsistent length of `target_linf`. Must have length {len(x0)}."
    assert (
        len(target_linf.shape) == 1
    ), "Inconsistent shape of `target_linf` (must be 1D)."

    target_linf = target_linf.view((-1, 1, 1, 1)).expand(*x0.shape)
    target_linf = target_linf.view(len(target_linf), -1)

    target_linf = target_linf.type(delta.dtype)

    delta2 = delta.abs().reshape((N, -1))
    space = torch.where(delta >= 0, 1 - x0, x0).reshape((N, -1)).type(delta.dtype)

    space_mask = space < target_linf

    if torch.any(torch.all(space_mask, dim=-1)):
        print("Not possible to rescale delta yield set Linf distance")

    delta2[space_mask] = 0

    delta2_sorted, _ = torch.sort(delta2, dim=-1, descending=True)

    eps = target_linf[:, 0] / delta2_sorted[:, 0]

    eps = eps.view(-1, 1, 1, 1)

    return eps


def clipping_aware_rescaling(
    x0: torch.Tensor,
    delta: torch.Tensor,
    target_distance: Union[float, torch.Tensor],
    norm: NormType,
    growing: bool = True,
    shrinking: bool = True,
    return_delta: bool = False,
):
    """Rescale delta such that it exactly lies target_distance away from x0 after clipping.

    Adapted from https://github.com/jonasrauber/clipping-aware-rescaling/.

    Args:
        x0: Tensor containing the base samples.
        delta: Tensor containing the perturbations to add to x0.
        target_distance: Target distance.
        norm: Norm for measuring the distance between x0 and delta.
        growing: If True, delta is allowed to grow.
        shrinking: If True, delta is allowed to shrink.
        return_delta: Return rescaled delta in addition to x0
          plus rescaled delta.

    Returns:
        If return_delta, Tuple of (x0 plus rescaled delta, rescaled delta), otherwise
        only x0 plus rescaled delta.
    """
    if norm == "linf":
        eps = clipping_aware_rescaling_linf_torch(x0, delta, target_distance)
    elif norm == "l2":
        eps = clipping_aware_rescaling_l2_torch(x0, delta, target_distance)
    elif norm == "l1":
        eps = clipping_aware_rescaling_l1_torch(x0, delta, target_distance)
    else:
        raise ValueError("Invalid norm")

    if not shrinking:
        eps = torch.clamp_min(eps, 1.0)
    if not growing:
        eps = torch.clamp_max(eps, 1.0)

    x = x0 + eps * delta
    x = torch.clamp(x, 0, 1)

    if return_delta:
        return x, eps * delta
    else:
        return x


def normalize(x: torch.Tensor, norm: NormType):
    """Normalize data to have unit norm.

    Args:
        x: Data to normalize.
        norm: Norm to use.
    Returns:
        Normalized x0.
    """
    if norm == "linf":
        x = torch.sign(x)
    elif norm in ("l2", "l1"):
        x /= torch.norm(x, p=1 if norm == "l1" else 2, keepdim=True, dim=(1, 2, 3))
    else:
        raise ValueError("Invalid norm:", norm)
    return x


class RandomizeLabelsDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base: torch.utils.data.Dataset,
        mode: LabelRandomization,
        label_map: Optional[Dict[int, int]] = None,
        n_classes: int = 10,
    ):
        if not n_classes > 0:
            raise ValueError("n_classes must be > 0.")
        if mode is None and label_map is None:
            raise ValueError("If mode is None, label_map must not be None.")
        if not mode in (None, "random", "systematically"):
            raise ValueError("mode must be one of None, random, systematically.")

        self.base = base
        self.mode = mode
        if label_map is None:
            if mode == "random":
                labels = np.random.randint(low=0, high=n_classes, size=len(base))
            elif mode == "systematically":
                labels = [
                    (a + b) % n_classes for a, b in enumerate(list(range(n_classes)))
                ]
            random.shuffle(labels)
            label_map = {i: labels[i] for i in range(len(labels))}
        self.label_map = label_map

    def __getitem__(self, item):
        x, y = self.base[item]
        if self.mode == "random":
            y = self.label_map[item]
        elif self.mode == "systematically":
            y = self.label_map[y]
        else:
            raise ValueError()
        return x, y

    def __len__(self):
        return len(self.base)

    def __repr__(self):
        return f"RandomizeLabelsDataset(base_dataset: {repr(self.base)}, mode: {self.mode})"


def build_dataloader_from_arrays(x: np.ndarray, y: np.ndarray, batch_size: int = 1):
    """Wrap two arrays in a dataset and data loader.

    Args:
        x: Array containing input data.
        y: Array containing target data.
        batch_size: Batch size of the newly created data loader.

    Returns:
        Dataloader based on x,y.
    """
    x_tensor = torch.tensor(x, device="cpu", dtype=torch.float32)
    y_tensor = torch.tensor(y, device="cpu", dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size)

    return dataloader
