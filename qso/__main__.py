from .optimizers import get_optimizer
from .problem.parser import get_main_parser

if __name__ == "__main__":
    parser = get_main_parser()
    arguments = parser.parse_args()

    arguments.optimizer = get_optimizer(arguments.optimizer)
    arguments.func(arguments)
