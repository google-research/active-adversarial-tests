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
import importlib
import re
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
from typing import Union

from typing_extensions import Literal

import utils as ut


@dataclass
class AdversarialAttackSettings:
    epsilon: float
    norm: ut.NormType
    step_size: float
    n_steps: int = 20
    n_averages: int = 1
    attack: Tuple[Literal["pgd", "kwta"]] = "pgd"
    random_start: bool = True

    def __repr__(self):
        return (
            f"{self.attack}_{self.norm}_{self.epsilon}_{self.step_size}_"
            f"{self.n_steps}_{self.n_averages}_{self.random_start}"
        )


@dataclass
class DecisionBoundaryBinarizationSettings:
    epsilon: float
    norm: ut.NormType
    n_inner_points: int
    n_boundary_points: int
    adversarial_attack_settings: Optional[AdversarialAttackSettings]
    n_boundary_adversarial_points: int = 0
    n_far_off_boundary_points: int = 0
    n_far_off_adversarial_points: int = 0
    optimizer: str = "adam"
    lr: float = 5e-2
    class_weight: Optional[Union[Literal["balanced"], dict]] = None

    def __repr__(self):
        return (
            f"{self.norm}_{self.epsilon}_{self.n_inner_points}_"
            f"{self.n_boundary_points}_{self.n_far_off_boundary_points}_"
            f"{self.adversarial_attack_settings}_{self.optimizer}_{self.lr}"
        )


def __parse_structure_argument(
    value,
    argument_type: Union[Callable[[str], Any], type],
    known_flags: Dict[str, Tuple[str, bool]],
    argument_types: Dict[str, Callable],
):
    """
    Recursively parses structured arguments encoded as a string.

    Args:
        argument_type: Class to store values in.
        known_flags: Map between name and default value of flags.
        argument_types: Map between argument names and argument constructors
            for variables.

    Returns:
        Object created based on string.
    """
    arguments = re.findall(r'(?:[^\s,"]|"(?:\\.|[^"])*")+', value)
    kwargs = {}
    for argument in arguments:
        parts = argument.split("=")
        if len(parts) > 2:
            parts = [parts[0], "=".join(parts[1:])]
        if len(parts) != 2:
            # argument is a flag
            if argument not in known_flags:
                raise argparse.ArgumentTypeError(
                    "invalid argument/unknown flag:", argument
                )
            else:
                kwargs[known_flags[argument][0]] = known_flags[argument][1]
        else:
            key, value = parts
            value = value.replace(r"\"", '"')
            if value.startswith('"') and value.endswith('"'):
                value = value[1:-1]
            if key in argument_types:
                kwargs[key] = argument_types[key](value)
            else:
                raise argparse.ArgumentTypeError(
                    f"invalid argument `{argument}` for type `{argument_type}`"
                )

    try:
        return argument_type(**kwargs)
    except Exception as ex:
        raise argparse.ArgumentTypeError("Could not create type:", ex)


def parse_adversarial_attack_argument(value):
    """Parse a string defining a AdversarialAttackSettings object."""
    return __parse_structure_argument(
        value,
        AdversarialAttackSettings,
        {},
        {
            "norm": str,
            "n_steps": int,
            "epsilon": float,
            "step_size": float,
            "attack": str,
            "n_averages": int,
            "random_start": lambda x: x.lower() == "true",
        },
    )


def parse_classifier_argument(value):
    """Parse a string describing a classifier object."""
    class_name = value.split(".")[-1]
    module_path = ".".join(value.split(".")[:-1])
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def parse_decision_boundary_binarization_argument(value):
    """Parse a string defining a DecisionBoundaryBinarizationSettings object."""
    return __parse_structure_argument(
        value,
        DecisionBoundaryBinarizationSettings,
        {},
        {
            "norm": str,
            "epsilon": float,
            "n_boundary_points": int,
            "n_inner_points": int,
            "adversarial_attack_settings": lambda x: parse_adversarial_attack_argument(
                x
            ),
            "optimizer": str,
            "lr": float,
            "class_weight": str,
        },
    )
