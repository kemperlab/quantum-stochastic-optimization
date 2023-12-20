import pennylane as qml
import jax.numpy as np
import jax

from jax import Array
from jax.random import KeyArray
from typing import Any, Callable

from . import Optimizer


class SPSA(Optimizer):

    def __init__(self,
                 qnode: Callable[[Array, list[qml.Hamiltonian]], list[Array]],
                 param_count: int,
                 key: KeyArray | None = None,
                 **kwargs) -> None:
        super().__init__(qnode, param_count, key)

        self.step_size = 0.01
        self.repeat_grads = 20

    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian]):
        gradient = np.zeros_like(self.params)

        for _ in range(self.repeat_grads):
            self.key, subkey = jax.random.split(self.key)
            perturbation = 2. * jax.random.bernoulli(
                subkey, shape=(self.param_count, )) - 1.

            dcosts = (self._evaluate_cost(
                self.params + perturbation * self.step_size,
                hamiltonians) - self._evaluate_cost(
                    self.params - perturbation * self.step_size, hamiltonians))

            gradient += dcosts / (2 * perturbation * self.step_size *
                                  self.repeat_grads)

        norm = np.linalg.norm(gradient)**2
        step = -self.step_size * gradient / norm

        new_cost = self._evaluate_cost(self.params + step, hamiltonians)

        if new_cost >= self.cost:
            self.step_size *= 0.9
        else:
            self.params = self.params + step
            self.step_size *= 1.1

    def log_info(self) -> dict[str, Any]:
        return {'cost': self.cost, 'step_size': self.step_size}
