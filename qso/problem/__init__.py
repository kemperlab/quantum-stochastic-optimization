from math import ceil, e, log
import pennylane as qml

from abc import ABC, abstractmethod
from jax import Array
from jax.random import PRNGKey, KeyArray

from operator import itemgetter

from qso.optimizers.trust_region import AdaptiveTrustRegion

from ..optimizers import Optimizer
from ..loggers import Logger


class QSOProblem(ABC):
    """
    An instance of a quantum stochastic optimization problem to optimizer for.
    Instances of this class are responsible for generating the Hamiltonians to
    optimize over.
    """

    def __init__(self, key: KeyArray | None = None) -> None:
        self.common_random_hamiltonians: list[qml.Hamiltonian] = []
        self.key = key if key is not None else PRNGKey(0)

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

    def solve_problem(self, optimizer: Optimizer,
                      logger: Logger) -> tuple[float, Array]:
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

        n_steps, samples_0, epsilon, resample = itemgetter(
            "n_steps",
            "n_hamiltonians",
            "epsilon",
            "resample",
        )(logger)

        for t in range(n_steps):
            if isinstance(optimizer, AdaptiveTrustRegion) and resample:
                samples = optimizer.n_t
            elif resample:
                samples = ceil(samples_0 *
                               log(max(e + 1, t), e)**(1 + epsilon))
            else:
                samples = 1

            hamiltonians = self.get_hamiltonians(samples)
            optimizer.step(hamiltonians)

            logger.log_step(optimizer.log_info())

        return optimizer.cost, optimizer.params
