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

from ..problem.runs import OptimizationDescription
from .runs import ExperimentRun

if TYPE_CHECKING:
    from jax import Array


def run(args: Namespace):
    if args.names:
        assert len(args.names) == len(args.folders)

    optimization_run_data = args.problem_spec.read()
    run_description = serde_json.from_json(OptimizationDescription,
                                           optimization_run_data)

    if args.sixel:
        mpl_use("module://matplotlib-backend-sixel")

    fig: plt.Figure = plt.figure()  # type: ignore
    ax: Axes = fig.add_subplot()

    min_eigvals = []
    lowest_points = []
    ranges = []

    if not args.names:
        folders = list(map(Path, args.folders))
        experiments = list(zip(folders, map(str, folders)))
    else:
        experiments = list(zip(map(Path, args.folders), args.names))

    for color, (folder,
                experiment_name) in zip(COLORS,
                                        tqdm(experiments, desc="Folders")):

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
            expected_hamiltonian = expected_hamiltonian.simplify()

            eigvals = eigvalsh(qml.matrix(expected_hamiltonian))
            min_eigvals.append(np.min(eigvals))

            xs = exp_run.get_x_axis(args.x_axis)

            circuit = problem.get_cost_circuit()

            if args.true_costs:
                ys = np.array([
                    circuit(
                        param,
                        [expected_hamiltonian],
                        None,
                    )[0].item() for param in tqdm(exp_run.get_params(),
                                                  desc="True Cost computation")
                ])
            else:
                ys = exp_run.get_costs()
            ax.plot(xs, ys, color=color, alpha=0.01)

            all_xs.append(xs)
            all_ys.append(ys)

        start_x = max(x.min().item() for x in all_xs)
        end_x = min(x.max().item() for x in all_xs)

        ranges.append((start_x, end_x))

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

        lowest_points.append(mean_ys.min().item())

    ground_energy = np.mean(np.array(min_eigvals)).item()
    ax.axhline(ground_energy, color='black', linestyle='--')

    ax.set_ylim(bottom=min(ground_energy, *lowest_points) - 0.05)

    x_min, x_max = max(map(lambda x: x[0],
                           ranges)), min(map(lambda x: x[1], ranges))
    x_range = x_max - x_min

    ax.set_xlim(left=x_min - 0.01 * x_range, right=x_max + 0.01 * x_range)

    ax.legend()
    ax.grid()

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
    parser.add_argument('--names', nargs='*')
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
    parser.add_argument('--x-axis',
                        choices=['iterations', 'shots'],
                        default='iterations')

    run(parser.parse_args())
