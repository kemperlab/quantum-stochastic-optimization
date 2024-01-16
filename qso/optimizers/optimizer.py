from __future__ import annotations
from typing import Any, Callable

import pennylane as qml
import jax
import math

from jax import numpy as np, Array
from jax.random import PRNGKey
from abc import ABC, abstractmethod

from ..problem.resampling_params import ResamplingParameters, MultipleHamiltonians

Circuit = Callable[[Array, list[qml.Hamiltonian], int | None], list[Array]]
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

        self.jacobian: Circuit = jax.jacobian(self.circuit, argnums=0)

        self.key, subkey = jax.random.split(self.key)
        self.params = jax.random.normal(subkey, (param_count, ))
        self.cost = np.inf

        self.iterations = 0

    def _evaluate_cost(
        self,
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ) -> Array:
        """
        This method computes the average Hamiltonian before computing the
        expectation value.
        """
        if len(hamiltonians) > 1:
            hamiltonians = [sum(hamiltonians) / len(hamiltonians)
                            ]  # type: ignore
        elif len(hamiltonians) == 0:
            return np.array(0.)

        return self.circuit(params, hamiltonians, shots_per_hamiltonian)[0]

    def _evaluate_costs(
        self,
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ) -> Array:
        return np.array(
            self.circuit(params, hamiltonians, shots_per_hamiltonian))

    def _evaluate_grad(
        self,
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ) -> Array:
        """
        This method computes the average Hamiltonian before computing the
        gradient of the expectation value.
        """
        if len(hamiltonians) > 1:
            hamiltonians = [sum(hamiltonians) / len(hamiltonians)
                            ]  # type: ignore
        elif len(hamiltonians) == 0:
            return np.zeros_like(params)

        return self.jacobian(params, hamiltonians, shots_per_hamiltonian)[0]

    def _evaluate_grads(
        self,
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ) -> Array:
        return np.array(
            self.jacobian(params, hamiltonians, shots_per_hamiltonian))

    def step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ):
        self.cost = float(
            self._evaluate_cost(self.params, hamiltonians,
                                shots_per_hamiltonian))

        self.optimizer_step(hamiltonians, shots_per_hamiltonian)

        self.iterations += 1

    @abstractmethod
    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian],
                       shots_per_hamiltonian: int | None):
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
        - `resampling_params` (`qso.problem.runs.ResamplingParameters`): The
          parameters that define how to resample from the distribution.

        Returns
        ---
        - `sample_count` (`int`): The number of requested samples.
        """

        match resampling_params:
            case MultipleHamiltonians():
                epsilon = resampling_params.epsilon
                hamiltonians = resampling_params.hamiltonians

                return math.ceil(
                    hamiltonians *
                    math.log2(max(3., self.iterations))**(1 + epsilon))

            case _:
                return 1

    def uses_individual_hamiltonians(self) -> bool:
        return False
