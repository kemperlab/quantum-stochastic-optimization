from serde import serde
from dataclasses import dataclass, field
from typing import Literal

from ...utils import NormalDistribution, Distribution


@serde
@dataclass
class TightBindingParameters:
    n_atoms: int = 5
    orbitals: set[Literal['s']] = field(default_factory=lambda: {'s'})
    alpha: Distribution = field(
        default_factory=lambda: NormalDistribution(10., 1.5))

    layers: int = 5
    trotter_steps: int = 1

