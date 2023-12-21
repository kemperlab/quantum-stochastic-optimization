import os
import pennylane as qml
import jax

from typing import Callable, Literal, Set
from jax import Array
from jax.random import PRNGKey

from pennylane.qaoa import x_mixer
from pennylane.fermi import FermiSentence
from pennylane.qchem import qubit_observable

from argparse import ArgumentParser, Namespace

from . import QSOProblem
from ..loggers import PrettyPrint
from ..utils.ansatz import hamiltonian_ansatz


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


def tight_binding_ansatz(
    n_var: int,
    n_layers: int = 5,
    trotter_steps: int = 5,
) -> tuple[int, Callable[[Array], None]]:
    x_hamiltonian = x_mixer(range(n_var))

    def qaoa_layer(times: Array, params: Array):
        qml.ApproxTimeEvolution(
            hamiltonian_ansatz(params, 'z', 'x', n_var) +
            hamiltonian_ansatz(params, 'z', 'y', n_var),
            times[0],
            trotter_steps,
        )
        qml.CommutingEvolution(x_hamiltonian, times[1])

    def state_circuit(params: Array):
        for wire in range(n_var):
            qml.PauliX(wire)
            qml.Hadamard(wire)

        times = params[:2 * n_layers].reshape(n_layers, 2)
        params = params[2 * n_layers:]

        qml.layer(qaoa_layer, n_layers, times, params=params)

    return 2 * n_layers + 2 * n_var - 1, state_circuit


class TightBindingProblem(QSOProblem):
    """
    This class describes the tight binding problem.
    """

    def __init__(self,
                 n_atoms: int,
                 orbitals: Set[Literal['s']],
                 alpha: tuple[float, float] = (10, 0.1),
                 key: Array | None = None) -> None:
        """
        Initialize an instance of the tight binding problem.

        Parameters
        ---
        - `alpha` (`tuple[float, float]`): A tuple of the mean and standard
          deviation, respectively, of the lattice spacing in the tight binding
          model.
        - `key` (`jax.Array`): A generator to deterministically
          generate the pseudo-random numbers used.
        """
        super().__init__(key)

        self.n_atoms = n_atoms
        self.orbitals = orbitals
        self.alpha = alpha

        self.n_var = self.n_atoms * len(self.orbitals)

    def sample_hamiltonian(self) -> qml.Hamiltonian:
        alpha_mu, alpha_sigma = self.alpha

        self.key, key = jax.random.split(self.key)
        alpha = alpha_mu + jax.random.normal(key).item() * alpha_sigma

        return tight_binding_hamiltonian(self.n_atoms, self.orbitals, alpha)


def get_parser(parser: ArgumentParser):
    parser.add_argument("--n_atoms", type=int, default=5)
    parser.add_argument("--alpha_mu", type=float, default=10.)
    parser.add_argument("--alpha_sigma", type=float, default=0.1)


def run(args: Namespace):
    key = PRNGKey(args.seed)

    key, problem_key, optimizer_key = jax.random.split(key, 3)
    problem = TightBindingProblem(args.n_atoms, {'s'},
                                  alpha=(args.alpha_mu, args.alpha_sigma),
                                  key=problem_key)

    n_var = problem.n_var
    param_count, ansatz = tight_binding_ansatz(n_var)

    os.environ['OPENBLAS_NUM_THREADS'] = '8'
    os.environ['OMP_NUM_THREADS'] = '8'
    os.environ['MKL_NUM_THREADS'] = '8'
    qdev: qml.QubitDevice = qml.device(
        "default.qubit",
        wires=n_var,
        shots=args.shots,
        max_workers=os.cpu_count() // 8,  # type: ignore
    )  # type: ignore

    @qml.qnode(qdev, diff_method="best")
    def single_cost_circuit(params: Array, hamiltonian: qml.Hamiltonian):
        ansatz(params)

        return qml.expval(hamiltonian)

    def cost_circuit(
        params: Array,
        hamiltonians: list[qml.Hamiltonian],
        shots_per_hamiltonian: int,
    ):
        return [
            single_cost_circuit(params,
                                hamiltonian,
                                shots=shots_per_hamiltonian)
            for hamiltonian in hamiltonians
        ]

    optimizer = args.optimizer(cost_circuit,
                               param_count,
                               **vars(args),
                               key=optimizer_key)

    logger = PrettyPrint(**vars(args))
    logger.register_hook(lambda x: x.save_json(args.data_file, overwrite=True))

    problem.solve_problem(optimizer, logger)
