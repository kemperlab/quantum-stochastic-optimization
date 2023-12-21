import pennylane as qml
import jax
import math

from typing import Any
from jax import numpy as np, Array

from .optimizer import Optimizer, Circuit


class SPSA(Optimizer):

    def __init__(
        self,
        qnode: Circuit,
        param_count: int,
        key: Array | None = None,
        step_size: float = 0.01,
        repeat_grads: int = 20,
        epsilon: float = 0.1,
        n_hamiltonians: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(qnode, param_count, key)

        self.step_size = step_size
        self.repeat_grads = repeat_grads

        self.hyperparams = {
            'epsilon': epsilon,
            'step_size': step_size,
            'repeat_grads': repeat_grads,
            'n_hamiltonians': n_hamiltonians,
        }

        self.log['hyperparams'] = self.hyperparams

    def optimizer_step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        gradient = np.zeros_like(self.params)

        for _ in range(self.repeat_grads):
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
                                  self.repeat_grads)

        norm = np.linalg.norm(gradient)**2
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

    def sample_count(self) -> int:
        epsilon = self.hyperparams['epsilon']
        n_hamiltonians = self.hyperparams['n_hamiltonians']

        return math.ceil(n_hamiltonians *
                         math.log2(max(3., self.iterations))**(1 + epsilon))
