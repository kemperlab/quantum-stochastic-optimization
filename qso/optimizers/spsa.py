from __future__ import annotations
from typing import Any, TYPE_CHECKING

import pennylane as qml
import jax

from typing import Any
from jax import numpy as np, Array
from serde import serde
from dataclasses import dataclass

from .optimizer import Optimizer


if TYPE_CHECKING:
    from jax import Array
    from .optimizer import Circuit

@serde
@dataclass
class SpsaParameters:
    step_size: float = 0.01
    repeat_grads: int = 20


class Spsa(Optimizer):

    def __init__(
        self,
        circuit: Circuit,
        param_count: int,
        spsa_params: SpsaParameters,
        key: Array | None = None,
    ) -> None:
        super().__init__(circuit, param_count, key)

        self.hyperparams = spsa_params
        self.step_size = spsa_params.step_size

    def optimizer_step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        gradient = np.zeros_like(self.params)

        for _ in range(self.hyperparams.repeat_grads):
            self.key, subkey = jax.random.split(self.key)
            perturbation = 2. * jax.random.bernoulli(
                subkey, shape=(self.param_count, )) - 1.

            dcosts = (self._evaluate_cost(
                self.params + perturbation * self.step_size,
                hamiltonians,
                shots_per_hamiltonian,
            ) - self._evaluate_cost(
                self.params - perturbation * self.step_size,
                hamiltonians,
                shots_per_hamiltonian,
            ))

            gradient += dcosts / (2 * perturbation * self.step_size *
                                  self.hyperparams.repeat_grads)

        norm: float = np.linalg.norm(gradient).item()**2
        step = -self.step_size * gradient / norm

        new_cost = self._evaluate_cost(
            self.params + step,
            hamiltonians,
            shots_per_hamiltonian,
        )

        if new_cost >= self.cost:
            self.step_size *= 0.9
        else:
            self.params = self.params + step
            self.step_size *= 1.1

    def log_info(self) -> dict[str, Any]:
        return {'cost': self.cost, 'step_size': self.step_size}
