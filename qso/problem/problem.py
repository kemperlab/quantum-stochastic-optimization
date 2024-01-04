from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml

from abc import ABC, abstractmethod
from jax import Array
from jax.random import PRNGKey

from qso.utils import get_qdev

from ..optimizers import (AdamParameters, Adam, Spsa, SpsaParameters,
                          TrustRegion, TrustRegionParameters)

if TYPE_CHECKING:
    from ..loggers import Logger
    from ..optimizers import Circuit, StateCircuit, Optimizer, OptimizerParameters
    from .runs import OptimizationRun


def get_optimizer(circuit: Circuit, param_count: int,
                  parameters: OptimizerParameters) -> Optimizer:
    match parameters:
        case AdamParameters():
            return Adam(circuit, param_count, parameters)

        case SpsaParameters():
            return Spsa(circuit, param_count, parameters)

        case TrustRegionParameters():
            return TrustRegion(circuit, param_count, parameters)


class QSOProblem(ABC):
    """
    An instance of a quantum stochastic optimization problem to optimizer for.
    Instances of this class are responsible for generating the Hamiltonians to
    optimize over.
    """

    run_params: OptimizationRun

    def __init__(self,
                 run_params: OptimizationRun,
                 key: Array | None = None) -> None:
        self.common_random_hamiltonians: list[qml.Hamiltonian] = []

        self.run_params = run_params
        self.key = key if key is not None else PRNGKey(run_params.seed.seed)

    @abstractmethod
    def sample_hamiltonian(self) -> qml.Hamiltonian:
        """
        This method should sample a Hamiltonian uniformly from the underlying
        distribution.

        Returns
        ---
        - hamiltonian (`pennylane.Hamiltonian`): A random Hamiltonian.
        """
        ...

    @abstractmethod
    def default_hamiltonian(self) -> qml.Hamiltonian:
        """
        This method should sample the default Hamiltonian for the problem.
        In the case of feature selection, for example, the Hamiltonian produced
        by not bootstrapping and taking the full distribution.

        Returns
        ---
        - hamiltonian (`pennylane.Hamiltonian`): A random Hamiltonian.
        """
        ...

    def get_hamiltonians(self, n: int) -> list[qml.Hamiltonian]:
        """
        Gets the first-`n` hamiltonians saved in the set of
        `common_random_hamiltonians`.
        """
        if n > len(self.common_random_hamiltonians):
            for _ in range(n - len(self.common_random_hamiltonians)):
                self.common_random_hamiltonians.append(
                    self.sample_hamiltonian())

        return self.common_random_hamiltonians[:n]

    @abstractmethod
    def get_ansatz(self) -> tuple[int, int, StateCircuit]:
        """
        This method should get a circuit ansatz associated with this problem.

        Returns
        ---
        - `param_count` (`int`): The number of parameters to be passed into the
          circuit as a `jax.Array`.
        - `wire_count` (`int`): The number of wires used by the circuit.
        - `circuit` (`qso.optimizer.StateCircuit`): A circuit ansatz that takes
          a parameter array (`jax.Array`) and prepares a state. This will
          typically be further composed with `qml.expval` calls for some
          `qml.Hamiltonian` instances to get expectation values.
        """
        ...

    def get_cost_circuit(self) -> Circuit:
        """
        This method should get the cost circuit constructed from the abstract
        `QSOProblem.get_ansatz` method.

        Returns
        ---
        - `circuit` (`qso.optimizer.Circuit`): A cost circuit that takes a
          parameter array (`jax.Array`), a list of hamiltonians
          (`list[qml.Hamiltonian]`), and the shots per hamiltonians (`int`) and
          gets the expected cost value for each hamiltonian.
        """
        _, wire_count, state_circuit = self.get_ansatz()
        qdev = get_qdev(wire_count)

        @qml.qnode(qdev)
        def single_cost_circuit(params: Array,
                                hamiltonian: qml.Hamiltonian) -> Array:
            state_circuit(params)
            return qml.expval(hamiltonian)  # type: ignore

        def cost_circuit(params: Array, hamiltonians: list[qml.Hamiltonian],
                         shots_per_hamiltonian: int) -> list[Array]:
            return [
                single_cost_circuit(params,
                                    hamiltonian,
                                    shots=shots_per_hamiltonian)
                for hamiltonian in hamiltonians
            ]

        return cost_circuit

    def solve_problem(
        self,
        logger: Logger,
    ) -> tuple[float, Array]:
        """
        Determines the ground state of the expectation value of system's
        Hamiltonian.

        Parameters
        ---
        - optimizer (`qso.optimizers.Optimizer`): The optimizer to optimize
          using.
        - logger (`qso.logger.Logger`): The storage for the current status of
          the problem solving.

        Returns
        ---
        - ground_state_energy (`float`): The ground state energy of the system.
        - parameters (`jax.Array`): The parameters that produce a ground state
          for the given optimizer ansatz.
        """

        param_count, _, _ = self.get_ansatz()
        cost_circuit = self.get_cost_circuit()

        optimizer = get_optimizer(cost_circuit, param_count,
                                  self.run_params.optimizer)

        for _ in range(self.run_params.steps):
            samples = optimizer.sample_count(self.run_params.resampling)

            if self.run_params.resampling is not None:
                hamiltonians = self.get_hamiltonians(samples)

                if self.run_params.resampling.split_shots:
                    step_shots = self.run_params.shots // samples
                else:
                    step_shots = self.run_params.shots

            else:
                hamiltonians = [self.default_hamiltonian()]
                step_shots = self.run_params.shots

            optimizer.step(hamiltonians, step_shots)

            logger.log_step(
                optimizer.log_info()
                | {
                    "shots_per_hamiltonian": step_shots,
                    "samples": samples,
                    "params": optimizer.params.tolist(),
                })

        return optimizer.cost, optimizer.params
