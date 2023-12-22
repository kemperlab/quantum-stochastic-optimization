from typing import Any, Callable

import pennylane as qml
import jax

from jax import numpy as np, Array
from jax.random import PRNGKey
from abc import ABC, abstractmethod


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


Circuit = Callable[[Array, list[qml.Hamiltonian], int], list[Array]]


class Optimizer(ABC):
    """
    The `Optimizer` class generalizes optimizers to be able to swap them out
    spontaneously.
    """

    def __init__(
        self,
        qnode: Circuit,
        param_count: int,
        key: Array | None,
    ) -> None:
        self.circuit = qnode
        self.param_count = param_count
        self.key = key if key is not None else PRNGKey(0)

        self.key, subkey = jax.random.split(self.key)
        self.params = jax.random.normal(subkey, (param_count, ))
        self.cost = np.inf

        self.iterations = 0

        self.log: dict[str, Any] = {"iterations": []}

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

    @abstractmethod
    def sample_count(self) -> int:
        """
        Gets the number of samples requested for the next step.

        Returns
        ---
        - `sample_count` (`int`): The number of requested samples.
        """

        ...
