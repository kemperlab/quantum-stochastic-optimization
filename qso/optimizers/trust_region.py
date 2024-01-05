from __future__ import annotations
from typing import Any, TYPE_CHECKING

import pennylane as qml
import math
import jax

from jax import numpy as np, Array
from serde import serde
from dataclasses import dataclass

from .optimizer import Optimizer

if TYPE_CHECKING:
    from .optimizer import Circuit
    from ..problem import ResamplingParameters


@serde
@dataclass
class TrustRegionParameters:
    delta_0: float
    rho: float
    gamma_1: float
    gamma_2: float
    mu: float


class TrustRegion(Optimizer):

    def __init__(
        self,
        qnode: Circuit,
        param_count: int,
        trust_region_params: TrustRegionParameters,
        key: Array | None = None,
    ) -> None:
        super().__init__(qnode, param_count, key)

        self.jacobian: Circuit = jax.jacrev(self.circuit, argnums=0)

        self.hyperparams = trust_region_params
        self.delta_t = trust_region_params.delta_0
        self.sigma_t2 = 0.

    def optimizer_step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int | None,
    ):
        rho = self.hyperparams.rho
        gamma_1 = self.hyperparams.gamma_1
        gamma_2 = self.hyperparams.gamma_2
        mu = self.hyperparams.mu

        jacobians = np.array(
            self.jacobian(self.params, hamiltonians, shots_per_hamiltonian))

        mean_gradient = jacobians.mean(axis=0)
        mean_gradient_norm: float = np.linalg.norm(mean_gradient).item()

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

    def sample_count(self,
                     resampling_params: ResamplingParameters | None) -> int:

        if resampling_params is not None:
            epsilon = resampling_params.epsilon
            hamiltonians = resampling_params.hamiltonians

            return math.ceil(
                hamiltonians *
                math.log2(max(3., self.iterations))**(1 + epsilon) *
                max(1., self.sigma_t2 / self.delta_t))
        else:
            return 1
