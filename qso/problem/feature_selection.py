import pennylane as qml
import jax

from typing import Callable
from jax import numpy as np, Array
from jax.random import PRNGKey

from pennylane.qaoa import x_mixer

from argparse import ArgumentParser, Namespace

from qso.loggers import PrettyPrint

from . import QSOProblem
from ..data.feature_selection import random_linearly_correlated_data
from ..optimizers.trust_region import AdaptiveTrustRegion
from ..utils import get_qdev, resample_data, ProblemHamiltonian
from ..utils.validation import check_ndarray
from ..utils.ansatz import hamiltonian_ansatz

N_LAYERS = 5
TROTTER_STEPS = 5


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


def feature_selection_ansatz(
        n_var: int) -> tuple[int, Callable[[Array], None]]:
    x_hamiltonian = x_mixer(range(n_var))

    def qaoa_layer(times: Array, params: Array):
        qml.ApproxTimeEvolution(
            hamiltonian_ansatz(params, 'z', 'z', n_var),
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

    return 2 * N_LAYERS + 2 * n_var - 1, state_circuit


class FeatureSelectionProblem(QSOProblem):
    """
    This class describes the feature selection problem.
    """

    def __init__(self,
                 feature_data: Array,
                 response_data: Array,
                 alpha: float = 0.5,
                 key: Array | None = None) -> None:
        """
        Initialize an instance of the feature selection problem.

        Parameters
        ---
        - `feature_data` (`numpy.ndarray`): An array of the shape `(N, k)`, where
          `k` is the number of features and `N` is the number of samples.
        - `response_data` (`numpy.ndarray`): An array of the shape `(N, )`
          where `N` is the number of samples.
        - `alpha` (`float`): Determines the weight to give the redundancy and
          importance matrix components of the objective matrix.
        - `key` (`jax.Array`): A generator to deterministically generate the
          pseudo-random numbers used.
        """
        super().__init__(key)
        self.feature_data = feature_data
        self.response_data = response_data
        self.alpha = alpha

        assert np.ndim(feature_data) == 2, (
            "Expected `feature_data` to have exactly 2 dimensions, "
            f"but found: {np.ndim(feature_data)}")

        self.n, self.k = self.feature_data.shape
        check_ndarray("response_data", self.response_data, shape=(self.n, ))

    def sample_hamiltonian(self) -> qml.Hamiltonian:
        self.key, key = jax.random.split(self.key)
        feature_data, response_data = resample_data(self.feature_data,
                                                    self.response_data,
                                                    samples=self.n,
                                                    key=key)
        objective = objective_matrix(feature_data,
                                     response_data,
                                     alpha=self.alpha)
        return qubo_hamiltonian(objective)

    def default_hamiltonian(self) -> qml.Hamiltonian:
        objective = objective_matrix(self.feature_data,
                                     self.response_data,
                                     alpha=self.alpha)
        return qubo_hamiltonian(objective)


def get_parser(parser: ArgumentParser):
    parser.add_argument("--k_real", type=int, default=2)
    parser.add_argument("--k_fake", type=int, default=2)
    parser.add_argument("--k_redundant", type=int, default=2)

    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--betas", nargs='+', type=float, default=[0.05])
    parser.add_argument("--gamma", type=float, default=0.05)

    parser.add_argument("--data_description", type=str, default=None)
    parser.add_argument("--data_description_file", type=str, default=None)

    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--print_hamiltonian",
                        action="store_true",
                        help="Just print Hamiltonian and exit.")


def run(args: Namespace):
    key = PRNGKey(args.seed)

    n_var = args.k_real + args.k_fake + args.k_redundant

    if args.data_description_file is not None:
        with open(args.data_description_file, 'r') as f:
            data_str = f.read()

        data = np.fromstring(data_str, sep=',')
        response_vector = data[:args.k_real]
        redundant_matrix = data[args.k_real:].reshape(args.k_redundant,
                                                      args.k_real)

    elif args.data_description is not None:
        data = np.fromstring(args.data_description, sep=',')
        response_vector = data[:args.k_real]
        redundant_matrix = data[args.k_real:].reshape(args.k_redundant,
                                                      args.k_real)

    else:
        key, resp_key, redund_key = jax.random.split(key, 3)
        response_vector = jax.random.normal(resp_key, shape=(args.k_real, ))
        redundant_matrix = jax.random.normal(redund_key,
                                             shape=(args.k_redundant,
                                                    args.k_real))

    if n_var > 1 and len(args.betas) > 1:
        betas = np.array(args.betas)
    else:
        betas = args.betas[0]

    key, new_data_key, problem_key, optimizer_key = jax.random.split(key, 4)
    feature_data, response_data = random_linearly_correlated_data(
        args.samples,
        args.k_real,
        args.k_fake,
        args.k_redundant,
        betas,
        args.gamma,
        response_vector=response_vector,
        redundant_matrix=redundant_matrix,
        key=new_data_key,
    )

    problem = FeatureSelectionProblem(feature_data,
                                      response_data,
                                      alpha=args.alpha,
                                      key=problem_key)

    if args.print_hamiltonian:
        print(ProblemHamiltonian(problem.default_hamiltonian()))
        print(
            ProblemHamiltonian(
                sum([
                    problem.sample_hamiltonian()
                    for _ in range(args.n_hamiltonians)
                ]) / args.n_hamiltonians))  # type: ignore
        exit(0)

    param_count, ansatz = feature_selection_ansatz(n_var)
    qdev = get_qdev(n_var)

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

    optimizer = AdaptiveTrustRegion(cost_circuit,
                                    param_count,
                                    **vars(args),
                                    key=optimizer_key)

    logger = PrettyPrint(**vars(args))
    logger.register_hook(lambda x: x.save_json(args.data_file, overwrite=True))

    problem.solve_problem(optimizer, logger)
