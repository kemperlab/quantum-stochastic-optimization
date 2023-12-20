from operator import itemgetter
from typing import Any, Callable

import pennylane as qml
import math
import jax

from jax.random import KeyArray
from jax import numpy as np, Array

from . import Optimizer


class AdaptiveTrustRegion(Optimizer):

    def __init__(self,
                 qnode: Callable[[Array, list[qml.Hamiltonian]], list[Array]],
                 param_count: int,
                 rho: float = 0.8,
                 gamma_1: float = 1.1,
                 gamma_2: float = 0.9,
                 epsilon: float = 0.1,
                 mu: float = 1000.,
                 delta_0: float = 0.2,
                 key: KeyArray | None = None,
                 **kwargs) -> None:
        super().__init__(qnode, param_count, key)

        self.jacobian: Callable[[Array, list[qml.Hamiltonian]],
                                list[Array]] = jax.jacobian(self.circuit,
                                                            argnums=0)
        self.hessian = jax.hessian(self.circuit, argnums=0)

        self.hyperparams = {
            'rho': rho,
            'gamma_1': gamma_1,
            'gamma_2': gamma_2,
            'epsilon': epsilon,
            'delta_0': delta_0,
            'mu': mu,
        }

        eps = self.hyperparams['epsilon']
        self.n_t = math.ceil(math.log2(max(3., self.iterations))**(1 + eps))

        self.delta_t = delta_0
        self.grad_norm = 0.

        self.log["hyperparams"] = self.hyperparams

    def optimizer_step(self, hamiltonians: list[qml.Hamiltonian]):
        rho, gamma_1, gamma_2, epsilon, mu = itemgetter(
            'rho',
            'gamma_1',
            'gamma_2',
            'epsilon',
            'mu',
        )(self.hyperparams)

        jacobians = np.array(self.jacobian(self.params, hamiltonians))
        hessians = np.array(self.hessian(self.params, hamiltonians))

        mean_gradient = jacobians.mean(axis=0)
        mean_gradient_norm = np.linalg.norm(mean_gradient)

        mean_hessian = hessians.mean(axis=0)

        self.grad_norm = float(mean_gradient_norm)

        # cauchy point
        step_scalar = mean_gradient.T @ mean_hessian @ mean_gradient
        if step_scalar <= 0:
            step_scalar = 1.
        else:
            step_scalar = min(
                1.,
                mean_gradient_norm**3 / (self.delta_t * step_scalar),
            )

        step = -step_scalar * self.delta_t / mean_gradient_norm * mean_gradient
        self.step_norm = float(np.linalg.norm(step))
        self.step_scalar = float(step_scalar)

        # compute improvement ratio
        predicted_cost = (self.cost + np.dot(step, mean_gradient) +
                          0.5 * np.dot(step, mean_hessian @ step))

        new_params = self.params + step
        new_cost = self._evaluate_cost(new_params, hamiltonians)

        self.predicted_cost = float(predicted_cost)
        self.new_cost = float(new_cost)

        pred_improvement = self.cost - predicted_cost
        true_improvement = self.cost - new_cost

        # conditionally step
        if true_improvement > rho * pred_improvement and self.step_norm < mu * self.grad_norm:
            self.params = new_params
            self.delta_t *= gamma_1
        else:
            self.delta_t *= gamma_2

        if len(hamiltonians) > 1:
            sigma_t2 = np.trace(np.cov(jacobians, rowvar=False))
        else:
            sigma_t2 = 0.

        self.sigma_t2 = float(sigma_t2)

        self.n_t = math.ceil(
            math.log2(max(3., self.iterations))**(1 + epsilon) *
            max(1., sigma_t2 / self.delta_t))

    def log_info(self) -> dict[str, Any]:
        return {
            "cost":
            self.cost,
            "n_t":
            self.n_t,
            "delta_t":
            self.delta_t,
            "gradient_norm":
            self.grad_norm,
            "step_norm":
            self.step_norm,
            "step_scalar":
            self.step_scalar,
            "sigma_t2":
            self.sigma_t2,
            "improvement_ratio":
            ((self.cost - self.new_cost) / (self.cost - self.predicted_cost)),
        }
