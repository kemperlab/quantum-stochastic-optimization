from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml
import jax

from jax import numpy as np, Array
from serde import serde
from dataclasses import dataclass

from pennylane.qaoa import x_mixer

from qso.optimizers.optimizer import StateCircuit

from ..problem import QSOProblem

from ...data.feature_selection import random_linearly_correlated_data
from ...utils import resample_data
from ...utils.validation import check_ndarray
from ...utils.ansatz import hamiltonian_ansatz

if TYPE_CHECKING:
    from ..runs import OptimizationRun

N_LAYERS = 5


@serde
@dataclass
class FeatureSelectionParameters:
    redundancy_matrix: list[list[float]]
    response_vector: list[float]

    k_real: int = 2
    k_fake: int = 2
    k_redundant: int = 2

    samples: int = 1024
    betas: list[float] | float = 0.05
    gamma: float = 0.05

    alpha: float = 0.5


def objective_matrix(feature_data: Array,
                     response_data: Array,
                     alpha: float = 0.5) -> Array:
    pre_objective = np.corrcoef(feature_data,
                                response_data[:, None],
                                rowvar=False)
    objective_body = np.abs(pre_objective[:-1, :-1]) * (1. - alpha)
    objective_body -= alpha * np.diagflat(np.abs(pre_objective[-1, :-1]))

    return objective_body


def qubo_hamiltonian(objective: Array) -> qml.Hamiltonian:
    j = objective / 4.
    h = -objective.sum(axis=0) / 2.

    n = objective.shape[0]
    check_ndarray("objective", objective, shape=(n, n))

    coeffs = []
    terms = []
    for m in range(n):
        for n in range(n):
            if m != n:
                coeffs.append(j[m, n])
                terms.append(qml.PauliZ(m) @ qml.PauliZ(n))
            else:
                coeffs.append(h[m])
                terms.append(qml.PauliZ(m))

    return qml.Hamiltonian(coeffs, terms).simplify()


class FeatureSelectionProblem(QSOProblem):
    """
    This class describes the feature selection problem.
    """

    def __init__(self,
                 run_params: OptimizationRun,
                 problem_params: FeatureSelectionParameters,
                 key: Array | None = None) -> None:
        """
        Initialize an instance of the feature selection problem.

        Parameters
        ---
        - `run_params` (`qso.OptimizationRun`): The collection of parameters
          defining the current optimization run.
        - `problem_params`
          (`qso.problem.feature_selection.FeatureSelectionParameters`): The set
          of parameters that describe the feature selection problem.
        - `key` (`jax.Array`): A generator to deterministically generate the
          pseudo-random numbers used.
        """
        super().__init__(run_params, key=key)

        self.problem_params = problem_params

        self.n_var = problem_params.k_real + problem_params.k_redundant + problem_params.k_fake

        if isinstance(problem_params.betas, list):
            if self.n_var > 1 and len(problem_params.betas) > 1:
                betas: float | Array = np.array(problem_params.betas)
            else:
                betas = problem_params.betas[0]
        else:
            betas = problem_params.betas

        self.key, new_data_key = jax.random.split(self.key)

        feature_data, response_data = random_linearly_correlated_data(
            problem_params.samples,
            problem_params.k_real,
            problem_params.k_fake,
            problem_params.k_redundant,
            betas,
            problem_params.gamma,
            response_vector=np.array(problem_params.response_vector),
            redundancy_matrix=np.array(problem_params.redundancy_matrix),
            key=new_data_key,
        )

        self.feature_data = feature_data
        self.response_data = response_data

        check_ndarray("feature_data",
                      self.feature_data,
                      shape=(problem_params.samples, self.n_var))

        check_ndarray("response_data",
                      self.response_data,
                      shape=(problem_params.samples, ))

    def sample_hamiltonian(self) -> qml.Hamiltonian:
        self.key, key = jax.random.split(self.key)

        feature_data, response_data = resample_data(
            self.feature_data,
            self.response_data,
            samples=self.problem_params.samples,
            key=key)

        objective = objective_matrix(feature_data,
                                     response_data,
                                     alpha=self.problem_params.alpha)

        return qubo_hamiltonian(objective)

    def default_hamiltonian(self) -> qml.Hamiltonian:
        objective = objective_matrix(self.feature_data,
                                     self.response_data,
                                     alpha=self.problem_params.alpha)
        return qubo_hamiltonian(objective)

    def get_ansatz(self) -> tuple[int, int, StateCircuit]:
        n_var = self.n_var
        x_hamiltonian = x_mixer(range(n_var))

        def qaoa_layer(times: Array, params: Array):
            qml.CommutingEvolution(
                hamiltonian_ansatz(params, 'z', 'z', n_var),
                times[0],
            )
            qml.CommutingEvolution(x_hamiltonian, times[1])

        def state_circuit(params: Array):
            for wire in range(n_var):
                qml.PauliX(wire)
                qml.Hadamard(wire)

            times = params[:2 * N_LAYERS].reshape(N_LAYERS, 2)
            params = params[2 * N_LAYERS:]

            qml.layer(qaoa_layer, N_LAYERS, times, params=params)

        return 2 * N_LAYERS + 2 * n_var - 1, n_var, state_circuit
