from __future__ import annotations
from typing import TYPE_CHECKING

import pennylane as qml

from argparse import ArgumentParser, FileType, Namespace
from jax import numpy as np
from jax.numpy.linalg import eigvalsh
from matplotlib import use as mpl_use, pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from serde import json as serde_json
from tqdm import tqdm

from qso.plot.style import COLORS

from ..problem import OptimizationDescription
from .runs import ExperimentRun

if TYPE_CHECKING:
    from jax import Array


def run(args: Namespace):
    optimization_run_data = args.problem_spec.read()
    run_description = serde_json.from_json(OptimizationDescription,
                                           optimization_run_data)

    if args.sixel:
        mpl_use("module://matplotlib-backend-sixel")

    fig: plt.Figure = plt.figure()  # type: ignore
    ax: Axes = fig.add_subplot()

    min_eigvals = []

    folders = list(map(Path, args.folders))
    for color, folder in zip(COLORS, tqdm(folders, desc="Folders")):
        experiment_name = str(folder)

        all_xs: list[Array] = []
        all_ys: list[Array] = []

        for run in tqdm(list(folder.glob("*.json")),
                        desc=f"{experiment_name} runs"):
            exp_run = ExperimentRun(run)
            run_number: int = exp_run.run_number

            optim_run = run_description.get_run(run_number)
            problem = optim_run.get_problem()

            expected_hamiltonian = sum([
                problem.sample_hamiltonian() for _ in range(args.hamiltonians)
            ]) / args.hamiltonians

            eigvals = eigvalsh(qml.matrix(expected_hamiltonian))

            min_eigvals.append(np.min(eigvals))

            xs = exp_run.get_x_axis(args.x_axis)

            circuit = problem.get_cost_circuit()

            if args.true_costs:
                ys = np.array([
                    circuit(
                        param,
                        [expected_hamiltonian],
                        args.shots,
                    )[0].item() for param in tqdm(exp_run.get_params(),
                                                  desc="True Cost computation")
                ])
            else:
                ys = exp_run.get_costs()
            ax.plot(xs, ys, color=color, alpha=0.05)

            all_xs.append(xs)
            all_ys.append(ys)

        start_x = max(x.min().item() for x in all_xs)
        end_x = min(x.max().item() for x in all_xs)

        xs = np.linspace(start_x, end_x, 1000)
        sample_ys = np.stack(
            [
                np.interp(xs, run_x, run_y)
                for run_x, run_y in zip(all_xs, all_ys)
            ],
            axis=0,
        )

        mean_ys = sample_ys.mean(axis=0)
        ci_ys = sample_ys.std(axis=0) / len(all_xs)**0.5

        ax.plot(xs, mean_ys, color=color, label=f"{experiment_name}")
        ax.fill_between(xs,
                        mean_ys - ci_ys,
                        mean_ys + ci_ys,
                        color=color,
                        alpha=0.2)

    ax.axhline(np.mean(np.array(min_eigvals)).item(),
               color='black',
               linestyle='--')
    ax.legend()
    ax.set_xlabel(args.x_axis)
    ax.set_ylabel("Cost")
    ax.set_title(f"Costs vs. {args.x_axis.title()}")

    if args.sixel:
        fig.show()

    fig.savefig(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-f',
                        '--folders',
                        nargs='+',
                        dest='folders',
                        required=True)
    parser.add_argument('-p',
                        '--problem-spec',
                        type=FileType('r'),
                        dest='problem_spec',
                        required=True)
    parser.add_argument('-o', '--output', dest='output', default='plot.pdf')
    parser.add_argument('-n',
                        '--hamiltonians',
                        type=int,
                        dest='hamiltonians',
                        required=True)
    parser.add_argument('--true-costs', action='store_true', dest='true_costs')
    parser.add_argument('--shots', type=int, default=None)
    parser.add_argument('--sixel', action='store_true')
    parser.add_argument('--x-axis', choices=['iterations', 'hamiltonians'], default='iterations')

    run(parser.parse_args())
