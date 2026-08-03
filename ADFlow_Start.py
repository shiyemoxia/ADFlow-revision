"""Command-line entry point for one ADFlow category run."""

from config import get_args
from main import main


def run_experiment():
    config = get_args()
    main(config)


if __name__ == '__main__':
    run_experiment()
