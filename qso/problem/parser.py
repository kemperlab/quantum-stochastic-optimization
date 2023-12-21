from argparse import ArgumentParser

from . import qchem, feature_selection
from ..optimizers import AdaptiveTrustRegion, Optimizer, OPTIMIZER_CATALOG


def get_optimizer(name: str) -> type[Optimizer]:
    if name in OPTIMIZER_CATALOG:
        return OPTIMIZER_CATALOG[name]
    else:
        raise ValueError(f"Could not find optimizer `{name}`")


def get_main_parser() -> ArgumentParser:
    """
    Constructs the main parser for the `__main__.py` script for the root module.
    """
    parser = ArgumentParser()

    parser.add_argument("--seed",
                        type=int,
                        default=0,
                        help="The seed for the random number generator.")

    parser.add_argument("--n_steps",
                        type=int,
                        default=100,
                        help="The number of steps to take in iterations.")

    parser.add_argument("--shots",
                        type=lambda x: int(x) if x != "None" else None,
                        default=1024,
                        help="The number of samples to get from each circuit.")

    parser.add_argument("--n_hamiltonians",
                        type=int,
                        default=1,
                        help="The number of starting Hamiltonians")

    parser.add_argument("--split_shots",
                        action='store_true',
                        help="Reallocate shots amongst sampled Hamiltonians.")

    parser.add_argument("--resample",
                        action="store_true",
                        help="Whether to resample Hamiltonians")

    parser.add_argument("--rho",
                        type=float,
                        default=0.8,
                        help="The percent of predicted model improvement "
                        "necessary for the parameter update to be kept. (TR)")

    parser.add_argument("--gamma_1",
                        type=float,
                        default=1.1,
                        help="The size by which to scale the trust region "
                        "on a successful parameter update step. (TR)")

    parser.add_argument("--gamma_2",
                        type=float,
                        default=0.9,
                        help="The size by which to scale the trust region "
                        "on an unsuccessful parameter update step. (TR)")

    parser.add_argument("--epsilon",
                        type=float,
                        default=0.1,
                        help="The rate at which the number of iterations "
                        "scales the number of Hamiltonian resamplings. "
                        "(TR | SPSA | Adam) ")

    parser.add_argument("--eps",
                        type=float,
                        default=0.1,
                        help="The offset to avoid division by zero. (Adam)")

    parser.add_argument("--delta_0",
                        type=float,
                        default=0.05,
                        help="The initial neighborhood size. (TR)")

    parser.add_argument("--mu",
                        type=float,
                        default=1000.,
                        help="The maximal step-to-gradient norm ratio. (TR)")

    parser.add_argument("--alpha",
                        type=float,
                        default=0.05,
                        help="The learning rate. (Adam)")

    parser.add_argument("--beta_1",
                        type=float,
                        default=0.9,
                        help="The exponential decay rate for the first-"
                        "moment. (Adam)")

    parser.add_argument("--beta_2",
                        type=float,
                        default=0.999,
                        help="The exponential decay rate for the second-"
                        "moment. (Adam)")

    parser.add_argument("--step_size",
                        type=float,
                        default=0.05,
                        help="The step size. (SPSA)")

    parser.add_argument("--data_file",
                        type=str,
                        default="data_output.json",
                        help="The location to save logging data.")

    parser.add_argument("--optimizer",
                        type=get_optimizer,
                        default=AdaptiveTrustRegion,
                        help="The algorithm to use for optimization")

    subparsers = parser.add_subparsers(required=True)

    qchem_parser = subparsers.add_parser(
        "qchem",
        help="The quantum chemistry problem, the tight-binding model, "
        "where lattice spacing lies on some distribution.")
    qchem.get_parser(qchem_parser)

    feature_selection_parser = subparsers.add_parser(
        "feature_selection",
        help="The feature selection problem, a pairing of "
        "cosine-similarity and bootstrap where the bootstrapping "
        "introduces some unknown correlation matrix distribution.")
    feature_selection.get_parser(feature_selection_parser)

    qchem_parser.set_defaults(func=qchem.run)
    feature_selection_parser.set_defaults(func=feature_selection.run)

    return parser
