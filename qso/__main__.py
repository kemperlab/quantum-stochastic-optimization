from argparse import ArgumentParser, FileType
from pathlib import Path

from serde import json as serde_json
from .loggers import PrettyPrint

from .problem import OptimizationDescription
from .utils import ProblemHamiltonian

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("run_specs",
                        type=FileType('rb'),
                        help="Path to run specification file.")
    parser.add_argument("-r", "--run", type=int, default=0, dest="run_number")
    parser.add_argument("-i", "--info", action="store_true", dest="info")
    parser.add_argument("-n",
                        "--n_hamiltonians",
                        type=int,
                        default=10,
                        help="This is the number of hamiltonians to use for"
                        "info. It is only relevant if `info` flag is enabled.")
    parser.add_argument("-o",
                        "--log",
                        type=Path,
                        default="output.json",
                        dest="log")

    args = parser.parse_args()

    run_desc = serde_json.from_json(OptimizationDescription,
                                    args.run_specs.read())

    run = run_desc.get_run(args.run_number)
    problem = run.get_problem()

    if args.info:
        n = args.n_hamiltonians

        print("Default Hamiltonian:")
        print(ProblemHamiltonian(problem.default_hamiltonian()))
        print(f"Expected Hamiltonian ({n} samples):")
        print(
            ProblemHamiltonian(
                sum([problem.default_hamiltonian() for _ in range(n)]) / n))

    logger = PrettyPrint(run=run,
                         run_number=args.run_number,
                         log_file=str(args.log),
                         steps=run.steps)
    logger.register_hook(lambda x: x.save_json(args.log, overwrite=True))

    problem.solve_problem(logger)
