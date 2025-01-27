"""The functions to be used to run infretis via the terminal."""

import argparse

from infretis.scheduler import scheduler
from infretis.setup import setup_config


def infretisrun():
    """Read input and runs infretis."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Location of infretis input file", required=True
    )

    args_dict = vars(parser.parse_args())
    input_file = args_dict["input"]

	# Run the infretis scheduler
    internalrun(input_file)


def internalrun(input_file):
    """Run internal runner.

    infretis can now be called directly without argparse.
    """
    config = setup_config(input_file)
    if config is None:
        return
    scheduler(config)
