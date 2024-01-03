from __future__ import annotations
from typing import Any, Callable, TYPE_CHECKING

import pennylane as qml
import jax
import math

from jax import numpy as np, Array
from jax.random import PRNGKey
from abc import ABC, abstractmethod


if TYPE_CHECKING:
    from ..problem import ResamplingParameters


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


Circuit = Callable[[Array, list[qml.Hamiltonian], int], list[Array]]
StateCircuit = Callable[[Array], None]


class Optimizer(ABC):
    """
    The `Optimizer` class generalizes optimizers to be able to swap them out
    spontaneously.
    """

    params: Array
    key: Array

    def __init__(
        self,
        circuit: Circuit,
        param_count: int,
        key: Array | None,
    ) -> None:
        self.circuit = circuit
        self.param_count = param_count
        self.key = key if key is not None else PRNGKey(0)

        self.key, subkey = jax.random.split(self.key)
        self.params = jax.random.normal(subkey, (param_count, ))
        self.cost = np.inf

        self.iterations = 0

    def _evaluate_cost(
        self,
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ) -> Array:
        return np.array(
            self.circuit(params, hamiltonians, shots_per_hamiltonian)).mean()

    def step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        self.cost = float(
            self._evaluate_cost(self.params, hamiltonians,
                                shots_per_hamiltonian))

        self.optimizer_step(hamiltonians, shots_per_hamiltonian)

        self.iterations += 1

    @abstractmethod
    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian],
                       shots_per_hamiltonian: int):
        ...

    def log_info(self) -> dict[str, Any]:
        return {
            "params": self.params.tolist(),
            "cost": self.cost,
        }

    def sample_count(self, resampling_params: ResamplingParameters) -> int:
        """
        Gets the number of samples requested for the next step.

        Parameters
        ---
        - `resampling_params` (`qso.problem.runs.ResamplingParameters`): The parameters
          that define how to resample from the distribution.

        Returns
        ---
        - `sample_count` (`int`): The number of requested samples.
        """

        if resampling_params.resample:
            epsilon = resampling_params.epsilon
            hamiltonians = resampling_params.hamiltonians

            return math.ceil(
                hamiltonians *
                math.log2(max(3., self.iterations))**(1 + epsilon))

        else:
            return 1
