from operator import itemgetter
from typing import Any

import pennylane as qml
import math
import jax

from jax import numpy as np, Array

from .optimizer import Optimizer, Circuit


class AdaptiveTrustRegion(Optimizer):

    def __init__(
        self,
        qnode: Circuit,
        param_count: int,
        rho: float = 0.8,
        gamma_1: float = 1.1,
        gamma_2: float = 0.9,
        epsilon: float = 0.1,
        mu: float = 1000.,
        delta_0: float = 0.2,
        n_hamiltonians: int = 1,
        key: Array | None = None,
        **kwargs,
    ) -> None:
        super().__init__(qnode, param_count, key)

        self.jacobian: Circuit = jax.jacrev(self.circuit, argnums=0)

        self.hyperparams = {
            'rho': rho,
            'gamma_1': gamma_1,
            'gamma_2': gamma_2,
            'epsilon': epsilon,
            'delta_0': delta_0,
            'mu': mu,
            'n_hamiltonians': n_hamiltonians,
        }

        self.delta_t = delta_0
        self.sigma_t2 = 0.

        self.log["hyperparams"] = self.hyperparams

    def optimizer_step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        rho, gamma_1, gamma_2, mu = itemgetter(
            'rho',
            'gamma_1',
            'gamma_2',
            'mu',
        )(self.hyperparams)

        jacobians = np.array(
            self.jacobian(self.params, hamiltonians, shots_per_hamiltonian))

        mean_gradient = jacobians.mean(axis=0)
        mean_gradient_norm = np.linalg.norm(mean_gradient)

        mean_hessian = np.eye(self.param_count)

        self.grad_norm = float(mean_gradient_norm)

        # cauchy point
        gradient_hessian_prod = mean_hessian @ mean_gradient

        step_scalar = (mean_gradient.T @ gradient_hessian_prod).item()
        if step_scalar <= 0:
            step_scalar = 1.
        else:
            step_scalar = min(
                1.,
                mean_gradient_norm**3 / (self.delta_t * step_scalar),
            )

        step = -step_scalar * self.delta_t / mean_gradient_norm * mean_gradient  # a scaled version of the gradient
        self.step_norm = float(np.linalg.norm(step))
        self.step_scalar = float(step_scalar)

        # compute improvement ratio
        predicted_cost = (self.cost + np.dot(step, mean_gradient) +
                          0.5 * np.dot(
                              step, -step_scalar * self.delta_t /
                              mean_gradient_norm * gradient_hessian_prod))

        new_params = self.params + step
        new_cost = self._evaluate_cost(new_params, hamiltonians,
                                       shots_per_hamiltonian)

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
            sigma_t2 = np.trace(np.cov(jacobians, rowvar=False)).item()
        else:
            sigma_t2 = 0.

        self.sigma_t2 = float(sigma_t2)

    def log_info(self) -> dict[str, Any]:
        return {
            "cost":
            self.cost,
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

    def sample_count(self) -> int:
        epsilon = self.hyperparams['epsilon']
        n_hamiltonians = self.hyperparams['n_hamiltonians']

        return math.ceil(n_hamiltonians *
                         math.log2(max(3., self.iterations))**(1 + epsilon) *
                         max(1., self.sigma_t2 / self.delta_t))
