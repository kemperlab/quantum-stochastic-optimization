from __future__ import annotations
from typing import Literal, TYPE_CHECKING

import pennylane as qml
import jax

from jax import Array
from serde import serde
from dataclasses import dataclass, field

from pennylane.qaoa import x_mixer
from pennylane.fermi import FermiSentence
from pennylane.qchem import qubit_observable

from ..problem import QSOProblem
from ...optimizers import StateCircuit
from ...utils.ansatz import hamiltonian_ansatz
from ...utils import NormalDistribution, Distribution

if TYPE_CHECKING:
    from jax import Array
    from ..runs import OptimizationRun

N_LAYERS = 5
TROTTER_STEPS = 5


@serde
@dataclass
class TightBindingParameters:
    n_atoms: int = 5
    orbitals: set[Literal['s']] = field(default_factory=lambda: {'s'})
    alpha: Distribution = field(default_factory=lambda: NormalDistribution(10., 1.5))


def potential_energy(orbital: Literal['s'], distance: float):
    assert orbital == 's'

    # distance should be in angstroms
    return -1.4 * 7.62 / distance**2  # eV


def tight_binding_hamiltonian(
    n_atoms: int,
    orbitals: set[Literal['s']],
    distance: float,
) -> qml.Hamiltonian:
    n_orbitals = len(orbitals)
    orbital_list = list(orbitals)

    sentence = FermiSentence({})
    for i in range(n_atoms - 1):
        j = i + 1

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
        alpha = self.problem_params.alpha.sample(key)

        return tight_binding_hamiltonian(self.problem_params.n_atoms,
                                         self.problem_params.orbitals, alpha)

    def default_hamiltonian(self) -> qml.Hamiltonian:
        alpha = self.problem_params.alpha.expected()
        return tight_binding_hamiltonian(self.problem_params.n_atoms,
                                         self.problem_params.orbitals, alpha)

    def get_ansatz(self) -> tuple[int, int, StateCircuit]:
        n_var = self.n_var
        x_hamiltonian = x_mixer(range(n_var))

        def qaoa_layer(times: Array, params: Array):
            qml.ApproxTimeEvolution(
                hamiltonian_ansatz(params, 'z', 'x', n_var) +
                hamiltonian_ansatz(params, 'z', 'y', n_var),
                times[0],
                TROTTER_STEPS,
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
