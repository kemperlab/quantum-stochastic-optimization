import pennylane as qml

from argparse import ArgumentParser, FileType, Namespace
from jax import numpy as np
from matplotlib import use as mpl_use, pyplot as plt
from pathlib import Path
from scipy.linalg import eigvalsh
from serde import json as serde_json

from . import confidence_plot
from ..problem import OptimizationDescription
from .runs import ExperimentRun


def run(args: Namespace):
    optimization_run_data = args.problem_spec.read()
    run_description = serde_json.from_json(OptimizationDescription,
                                           optimization_run_data)

    if args.sixel:
        mpl_use("module://matplotlib-backend-sixel")

    fig = plt.figure()
    ax = fig.add_subplot()

    for folder in args.folders:
        assert type(folder) == Path

        experiment_name = str(folder)
        list_of_runs = []
        min_eigvals = []
        expected_hamiltonians = []

        for run in folder.glob("*.json"):
            exp_run = ExperimentRun(run)
            run_number: int = exp_run.run_number

            optim_run = run_description.get_run(run_number)
            problem = optim_run.get_problem()

            expected_hamiltonian = sum([
                problem.sample_hamiltonian() for _ in range(args.hamiltonians)
            ]) / args.hamiltonians

            eigvals = np.array(eigvalsh(qml.matrix(expected_hamiltonian)))

            min_eigvals.append(np.min(eigvals))
            list_of_runs.append(optim_run)
            expected_hamiltonians.append(expected_hamiltonian)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f',
                        '--folders',
                        nargs='+',
                        type=Path,
                        dest='folders')
    parser.add_argument('-p',
                        '--problem_spec',
                        type=FileType('r'),
                        dest='problem_spec')
    parser.add_argument('-n', '--hamiltonians', type=int, dest='hamiltonians')
    parser.add_argument('--sixel', action='store_true')

    run(parser.parse_args())
