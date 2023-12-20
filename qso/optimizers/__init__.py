from typing import Any, Callable

import pennylane as qml
import jax

from jax import numpy as np, Array
from jax.random import KeyArray, PRNGKey
from abc import ABC, abstractmethod


def mean(vals: list[float]) -> float:
    return sum(vals) / len(vals)


class Optimizer(ABC):
    """
    The `Optimizer` class generalizes optimizers to be able to swap them out
    spontaneously.
    """

    def __init__(self, qnode: Callable[[Array, list[qml.Hamiltonian]],
                                       list[Array]], param_count: int,
                 key: KeyArray | None) -> None:
        self.circuit = qnode
        self.param_count = param_count
        self.key = key if key is not None else PRNGKey(0)

        self.key, subkey = jax.random.split(self.key)
        self.params = jax.random.normal(subkey, (param_count, ))
        self.cost = np.inf

        self.iterations = 0

        self.log: dict[str, Any] = {"iterations": []}

    def _evaluate_cost(self, params: Array,
                       hamiltonians: list[qml.Hamiltonian]) -> Array:
        return np.array(self.circuit(params, hamiltonians)).mean()

    def step(self, hamiltonians: list[qml.Hamiltonian]):
        self.cost = float(self._evaluate_cost(self.params, hamiltonians))

        self.optimizer_step(hamiltonians)

        self.iterations += 1

    @abstractmethod
    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian]):
        ...

    def log_info(self) -> dict[str, Any]:
        return {
            "params": self.params,
            "cost": self.cost,
        }
