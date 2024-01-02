from argparse import ArgumentParser, FileType
from pathlib import Path

from serde import json as serde_json
from .loggers import PrettyPrint

from .runs import OptimizationDescription

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("run_specs",
                        type=FileType('rb'),
                        help="Path to run specification file.")
    parser.add_argument("-r", "--run", type=int, default=0, dest="run_number")
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

    logger = PrettyPrint(run=run, run_number=args.run, log_file=str(args.log))
    logger.register_hook(lambda x: x.save_json(args.log, overwrite=True))

    problem.solve_problem(logger)
