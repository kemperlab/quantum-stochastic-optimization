from __future__ import annotations
from typing import Literal, TYPE_CHECKING

import pennylane as qml
import jax

from jax import Array
from pennylane.qaoa import x_mixer
from pennylane.fermi import FermiSentence
from pennylane.qchem import qubit_observable

from ..problem import QSOProblem
from ...optimizers import StateCircuit
from ...utils.ansatz import hamiltonian_ansatz

if TYPE_CHECKING:
    from .params import TightBindingParameters
    from ..runs import OptimizationRun


def potential_energy(orbital: Literal['s'], distance: float):
    assert orbital == 's'

    # distance should be in angstroms
    return -1.4 * 7.62 / distance**2  # eV
    # return 1 / distance**2


def tight_binding_hamiltonian(
    n_atoms: int,
    orbitals: set[Literal['s']],
    distances: Array,
) -> qml.Hamiltonian:
    n_orbitals = len(orbitals)
    orbital_list = list(orbitals)

    sentence = FermiSentence({})
    for i in range(n_atoms - 1):
        j = i + 1

        distance = distances[i].item()
        for k, orbital in enumerate(orbital_list):
            i_orbital = n_orbitals * i + k
            j_orbital = n_orbitals * j + k

            sentence += potential_energy(orbital, distance) * (
                qml.FermiC(i_orbital) * qml.FermiA(j_orbital) +
                qml.FermiC(j_orbital) * qml.FermiA(i_orbital))

    return qubit_observable(sentence)  # type: ignore


class TightBindingProblem(QSOProblem):
    """
    This class describes the tight binding problem.
    """

    def __init__(self,
                 run_params: OptimizationRun,
                 problem_params: TightBindingParameters,
                 key: Array | None = None) -> None:
        """
        Initialize an instance of the tight binding problem.

        Parameters
        ---
        - `run_params` (`qso.problem.runs.OptimizationRun`): The structure of the
          current experiment run.
        - `problem_params`
          (`qso.problem.tight_binding.TightBindingParameters`): The set of
          parameters that describe the tight binding problem.
        - `key` (`jax.Array`): A generator to deterministically
          generate the pseudo-random numbers used.
        """
        super().__init__(run_params, key=key)

        self.problem_params = problem_params
        self.n_var = self.problem_params.n_atoms * len(
            self.problem_params.orbitals)

    def sample_hamiltonian(self) -> qml.Hamiltonian:
        self.key, key = jax.random.split(self.key)
        alpha = self.problem_params.alpha.sample(
            key, (self.problem_params.n_atoms - 1, ))

        return tight_binding_hamiltonian(self.problem_params.n_atoms,
                                         self.problem_params.orbitals, alpha)

    def default_hamiltonian(self) -> qml.Hamiltonian:
        alpha = self.problem_params.alpha.expected(
            (self.problem_params.n_atoms - 1, ))
        return tight_binding_hamiltonian(self.problem_params.n_atoms,
                                         self.problem_params.orbitals, alpha)

    def get_ansatz(self) -> tuple[int, int, StateCircuit]:
        n_var = self.n_var
        x_hamiltonian = x_mixer(range(n_var))
        layers = self.problem_params.layers

        def qaoa_layer(times: Array, params: Array):
            qml.ApproxTimeEvolution(
                hamiltonian_ansatz(params, 'z', 'x', n_var) +
                hamiltonian_ansatz(params, 'z', 'y', n_var),
                times[0],
                self.problem_params.trotter_steps,
            )
            qml.CommutingEvolution(x_hamiltonian, times[1])

        def state_circuit(params: Array):
            for wire in range(n_var):
                qml.PauliX(wire)
                qml.Hadamard(wire)

            times = params[:2 * layers].reshape(layers, 2)
            params = params[2 * layers:]

            qml.layer(qaoa_layer, layers, times, params=params)

        return 2 * layers + 2 * n_var - 1, n_var, state_circuit
