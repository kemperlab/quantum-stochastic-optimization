import pennylane as qml

from argparse import ArgumentParser, Namespace
from itertools import cycle
from jax import numpy as np, Array
from matplotlib import use as mpl_use, pyplot as plt
from matplotlib.axes import Axes
from pathlib import Path
from tqdm import tqdm
from typing import Callable, Literal

from .runs import ExperimentRun
from ..problem.qchem import TightBindingProblem, tight_binding_ansatz

Circuit = Callable[[Array], Array]

colors = cycle(['b', 'g', 'r', 'c', 'm', 'y', 'k'])


def get_parser(parser: ArgumentParser):
    parser.add_argument('--plot-folders', nargs='+', type=Path, required=True)
    parser.add_argument('--plot-type',
                        choices=['confidence'],
                        type=str,
                        required=True)
    parser.add_argument('--problem',
                        choices=['qchem', 'feature_selection'],
                        type=str,
                        required=True)
    parser.add_argument('--sixel', action='store_true')
    parser.add_argument('--n_hamiltonians', type=int, default=100)
    parser.add_argument('--no-show', action='store_false', dest='show')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--true-cost', action='store_true')
    parser.add_argument('--x-axis',
                        choices=['iterations', 'hamiltonians'],
                        type=str,
                        required=True)


def confidence_plot(
    runs: dict[Path, list[ExperimentRun]],
    x_axis: Literal['iterations', 'hamiltonians'],
    ax: Axes,
    circuit: Circuit | None = None,
):
    for color, (experiment, run_list) in zip(tqdm(colors), runs.items()):
        all_xs: list[Array] = []
        all_ys: list[Array] = []
        for run in run_list:
            xs = run.get_x_axis(x_axis)
            ys = run.get_costs()

            if circuit is not None:
                ys = np.array([
                    circuit(param).item() for param in tqdm(run.get_params())
                ])

            all_xs.append(xs)
            all_ys.append(ys)

            ax.plot(xs, ys, color=color, alpha=0.02)

        start_x = max([x.min().item() for x in all_xs])
        end_x = min([x.max().item() for x in all_xs])

        xs = np.linspace(start_x, end_x, 1000)
        sample_ys = np.stack(
            [
                np.interp(xs, run_x, run_y)
                for run_x, run_y in zip(all_xs, all_ys)
            ],
            axis=0,
        )

        mean_ys = sample_ys.mean(axis=0)
        ci_ys = sample_ys.std(axis=0) / len(run_list)**0.5

        ax.plot(xs, mean_ys, color=color, label=f"{experiment}")
        ax.fill_between(xs,
                        mean_ys - ci_ys,
                        mean_ys + ci_ys,
                        color=color,
                        alpha=0.2)


def run(args: Namespace):
    if args.sixel:
        mpl_use('module://matplotlib-backend-sixel')

    folders: list[Path] = args.plot_folders
    plot_type = args.plot_type
    x_axis = args.x_axis
    problem = args.problem
    n_hamiltonians = args.n_hamiltonians

    runs = {
        folder: [ExperimentRun(run) for run in folder.glob("*.json")]
        for folder in folders
    }

    match problem:
        case 'qchem':
            _, circuit = tight_binding_ansatz(5)
            tb_problem = TightBindingProblem(5, {'s'}, (10, 1.5))
            exp_hamiltonian = sum([
                tb_problem.sample_hamiltonian() for _ in range(n_hamiltonians)
            ]) / n_hamiltonians

        case 'feature_selection':
            exit()

        case _:
            raise ValueError("Not a valid problem")

    qdev = qml.device('lightning.qubit', wires=5)

    def cost_circuit(params: Array):
        circuit(params)
        return qml.expval(exp_hamiltonian)

    cost_circuit_qnode: qml.QNode | None = qml.QNode(cost_circuit, qdev)

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    if not args.true_cost:
        cost_circuit_qnode = None

    match plot_type:
        case 'confidence':
            confidence_plot(runs, x_axis, ax, circuit=cost_circuit_qnode)

    ax.set_title(f"{plot_type} ({problem})")
    ax.set_ylabel(f"Cost")
    ax.set_xlabel(f"{x_axis}")

    ground_cost = np.linalg.eigvalsh(qml.matrix(exp_hamiltonian)).min().item()
    ax.axhline(ground_cost, color='black', linestyle='--')
    ax.set_ylim(bottom=ground_cost - 0.1, top=None)

    ax.grid(True)
    ax.legend()

    if args.output is not None:
        figure.savefig(args.output)

    if args.show:
        figure.show()
