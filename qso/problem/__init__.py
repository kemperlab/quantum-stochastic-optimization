import pennylane as qml

from abc import ABC, abstractmethod
from jax import Array
from jax.random import PRNGKey

from operator import itemgetter

from ..optimizers import Optimizer
from ..loggers import Logger


class QSOProblem(ABC):
    """
    An instance of a quantum stochastic optimization problem to optimizer for.
    Instances of this class are responsible for generating the Hamiltonians to
    optimize over.
    """

    def __init__(self, key: Array | None = None) -> None:
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

    def solve_problem(
        self,
        optimizer: Optimizer,
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

        (
            n_steps,
            resample,
            shots,
            split_shots,
        ) = itemgetter(
            "n_steps",
            "resample",
            "shots",
            "split_shots",
        )(logger)

        for _ in range(n_steps):
            samples = optimizer.sample_count() if resample else 1

            if split_shots:
                step_shots = shots // samples
            else:
                step_shots = shots

            hamiltonians = self.get_hamiltonians(samples)
            optimizer.step(hamiltonians, step_shots)

            logger.log_step(optimizer.log_info()
                            | {
                                'shots_per_hamiltonian': step_shots,
                                "samples": samples,
                            })

        return optimizer.cost, optimizer.params
