# from argparse import ArgumentParser
#
# parser = ArgumentParser()
#
# from . import tight_binding, feature_selection
# from .. import plot
#
#
# def get_main_parser() -> ArgumentParser:
#     """
#     Constructs the main parser for the `__main__.py` script for the root module.
#     """
#     parser = ArgumentParser()
#
#     parser.add_argument("--seed",
#                         type=int,
#                         default=0,
#                         help="The seed for the random number generator.")
#
#     parser.add_argument("--n_steps",
#                         type=int,
#                         default=100,
#                         help="The number of steps to take in iterations.")
#
#     parser.add_argument("--shots",
#                         type=lambda x: int(x) if x != "None" else None,
#                         default=1024,
#                         help="The number of samples to get from each circuit.")
#
#     parser.add_argument("--n_hamiltonians",
#                         type=int,
#                         default=1,
#                         help="The number of starting Hamiltonians")
#
#     parser.add_argument("--split_shots",
#                         action='store_true',
#                         help="Reallocate shots amongst sampled Hamiltonians.")
#
#     parser.add_argument("--resample",
#                         action="store_true",
#                         help="Whether to resample Hamiltonians")
#
#     parser.add_argument("--resample_single",
#                         action="store_true",
#                         help="Whether to pick resample a new Hamiltonian at "
#                         "each step.")
#
#     parser.add_argument("--rho",
#                         type=float,
#                         default=0.8,
#                         help="The percent of predicted model improvement "
#                         "necessary for the parameter update to be kept. (TR)")
#
#     parser.add_argument("--gamma_1",
#                         type=float,
#                         default=1.1,
#                         help="The size by which to scale the trust region "
#                         "on a successful parameter update step. (TR)")
#
#     parser.add_argument("--gamma_2",
#                         type=float,
#                         default=0.9,
#                         help="The size by which to scale the trust region "
#                         "on an unsuccessful parameter update step. (TR)")
#
#     parser.add_argument("--epsilon",
#                         type=float,
#                         default=0.1,
#                         help="The rate at which the number of iterations "
#                         "scales the number of Hamiltonian resamplings. "
#                         "(TR | SPSA | Adam) ")
#
#     parser.add_argument("--eps",
#                         type=float,
#                         default=0.1,
#                         help="The offset to avoid division by zero. (Adam)")
#
#     parser.add_argument("--delta_0",
#                         type=float,
#                         default=0.05,
#                         help="The initial neighborhood size. (TR)")
#
#     parser.add_argument("--mu",
#                         type=float,
#                         default=1000.,
#                         help="The maximal step-to-gradient norm ratio. (TR)")
#
#     parser.add_argument("--alpha",
#                         type=float,
#                         default=0.05,
#                         help="The learning rate. (Adam)")
#
#     parser.add_argument("--beta_1",
#                         type=float,
#                         default=0.9,
#                         help="The exponential decay rate for the first-"
#                         "moment. (Adam)")
#
#     parser.add_argument("--beta_2",
#                         type=float,
#                         default=0.999,
#                         help="The exponential decay rate for the second-"
#                         "moment. (Adam)")
#
#     parser.add_argument("--step_size",
#                         type=float,
#                         default=0.05,
#                         help="The step size. (SPSA)")
#
#     parser.add_argument("--data_file",
#                         type=str,
#                         default="data_output.json",
#                         help="The location to save logging data.")
#
#     parser.add_argument("--optimizer",
#                         choices=OPTIMIZER_CATALOG.keys(),
#                         default='tr',
#                         help="The algorithm to use for optimization")
#
#     subparsers = parser.add_subparsers(required=True)
#
#     tight_binding_parser = subparsers.add_parser(
#         "tight_binding",
#         help="The quantum chemistry problem, the tight-binding model, "
#         "where lattice spacing lies on some distribution.")
#     tight_binding.get_parser(tight_binding_parser)
#
#     feature_selection_parser = subparsers.add_parser(
#         "feature_selection",
#         help="The feature selection problem, a pairing of "
#         "cosine-similarity and bootstrap where the bootstrapping "
#         "introduces some unknown correlation matrix distribution.")
#     feature_selection.get_parser(feature_selection_parser)
#
#     plot_parser = subparsers.add_parser("plot",
#                                         help="Helps plot experiment results")
#     plot.get_parser(plot_parser)
#
#     tight_binding_parser.set_defaults(func=tight_binding.run)
#     feature_selection_parser.set_defaults(func=feature_selection.run)
#     plot_parser.set_defaults(func=plot.run)
#
#     return parser
