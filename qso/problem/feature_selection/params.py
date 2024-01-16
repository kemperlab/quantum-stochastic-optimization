from serde import serde
from dataclasses import dataclass


@serde
@dataclass
class FeatureSelectionParameters:
    redundancy_matrix: list[list[float]]
    response_vector: list[float]

    k_real: int = 2
    k_fake: int = 2
    k_redundant: int = 2

    samples: int = 1024
    betas: list[float] | float = 0.05
    gamma: float = 0.05

    alpha: float = 0.5

    layers: int = 5

