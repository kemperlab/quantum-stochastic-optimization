from operator import itemgetter
from typing import Any, Callable

import pennylane as qml
import math
import jax

from jax.random import KeyArray
from jax import numpy as np, Array

from . import Optimizer


class Adam(Optimizer):

    def __init__(self,
                 qnode: Callable[[Array, list[qml.Hamiltonian]], list[Array]],
                 param_count: int,
                 alpha: float = 1e-2,
                 beta_1: float = 0.9,
                 beta_2: float = 0.999,
                 epsilon: float = 1e-8,
                 key: KeyArray | None = None,
                 resample: bool = False,
                 **kwargs) -> None:
        super().__init__(qnode, param_count, key)

        self.jacobian: Callable[[Array, list[qml.Hamiltonian]],
                                list[Array]] = jax.jacobian(self.circuit,
                                                            argnums=0)
        self.hessian = jax.hessian(self.circuit, argnums=0)

        self.hyperparams: dict[str, float] = {
            'alpha': alpha,
            'beta_1': beta_1,
            'beta_2': beta_2,
            'epsilon': epsilon
        }

        epsilon = self.hyperparams['epsilon']
        self.resample = resample

        if self.resample:
            self.n_t = math.ceil(
                math.log2(max(3., self.iterations))**(1 + epsilon))
        else:
            self.n_t = 1

        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)

        self.grad_norm = 0.
        self.log["hyperparams"] = self.hyperparams

        self.iters = 0

    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian]):
        self.iters += 1

        alpha = self.hyperparams['alpha']
        beta_1 = self.hyperparams['beta_1']
        beta_2 = self.hyperparams['beta_2']
        epsilon = self.hyperparams['epsilon']

        jacobians = np.array(self.jacobian(self.params, hamiltonians))

        mean_gradient = jacobians.mean(axis=0)
        mean_gradient_norm = np.linalg.norm(mean_gradient)

        self.grad_norm = float(mean_gradient_norm)

        self.m = beta_1 * self.m + (1 - beta_1) * mean_gradient
        self.v = beta_2 * self.v + (1 - beta_2) * mean_gradient**2

        m_hat = self.m / (1 - beta_1**self.iters)
        v_hat = self.v / (1 - beta_2**self.iters)

        step = -alpha * m_hat / (np.sqrt(v_hat) + epsilon)
        self.step_norm = float(np.linalg.norm(step))

        self.params = self.params + step

        if self.resample:
            self.n_t = math.ceil(
                math.log2(max(3., self.iterations))**(1 + epsilon))
        else:
            self.n_t = 1

    def log_info(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "n_t": self.n_t,
            "gradient_norm": self.grad_norm,
            "step_norm": self.step_norm,
        }
