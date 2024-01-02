from typing import Any

import pennylane as qml
import jax

from jax import numpy as np, Array
from serde import serde
from dataclasses import dataclass

from .optimizer import Optimizer, Circuit


@serde
@dataclass
class AdamParameters:
    alpha: float = 0.01
    beta_1: float = 0.9
    beta_2: float = 0.999
    epsilon: float = 1e-8


class Adam(Optimizer):

    def __init__(
        self,
        circuit: Circuit,
        param_count: int,
        adam_params: AdamParameters,
        key: Array | None = None,
    ) -> None:
        super().__init__(circuit, param_count, key)

        self.jacobian: Circuit = jax.jacobian(self.circuit, argnums=0)
        self.hyperparams = adam_params

        self.m = np.zeros_like(self.params)
        self.v = np.zeros_like(self.params)

        self.grad_norm = 0.

        self.iters = 0

    def optimizer_step(
        self,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        self.iters += 1

        alpha = self.hyperparams.alpha
        beta_1 = self.hyperparams.beta_1
        beta_2 = self.hyperparams.beta_2
        epsilon = self.hyperparams.epsilon

        jacobians = np.array(
            self.jacobian(self.params, hamiltonians, shots_per_hamiltonian))

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

    def log_info(self) -> dict[str, Any]:
        return {
            "cost": self.cost,
            "gradient_norm": self.grad_norm,
            "step_norm": self.step_norm,
        }
