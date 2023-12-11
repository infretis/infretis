"""The functions to be used to run infretis via the terminal."""
import argparse
import os

import tomli

from infretis.scheduler import scheduler
from infretis.setup import setup_config
from infretis.tools.Wham_Pcross import run_analysis


def infretisrun():
    """Read input and runs infretis."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--input", help="Location of infretis input file", required=True
    )

    args_dict = vars(parser.parse_args())
    input_file = args_dict["input"]
    config = setup_config(input_file)
    if config is None:
        return
    scheduler(config)


def infretisanalyze():
    """Run Titus0 wham script."""
    parser = argparse.ArgumentParser()
    args = [
        "-toml",
        "-data",
        "-nskip",
        "-lamres",
        "-nblock",
        "-fener",
        "-folder",
    ]
    helps = [
        "the toml file for the simulation",
        "the infretis data.txt file",
        "number of skipped lines",
        "resolution along the lambda-CV",
        "minimal number of blocks",
        "calculate free energy",
        "output folder",
    ]
    defaults = [
        "infretis.toml",
        "infretis_data.txt",
        100,
        "(intf_1-intf0)/10)",
        5,
        False,
        "wham",
    ]
    # fill defaults
    # for defs, (key, value) in zip(defaults, args.items()):
    for arg, help0, def0 in zip(args, helps, defaults):
        parser.add_argument(
            arg,
            help=help0 + f" (default: {def0})",
            default=def0,
        )

    # get user input
    imps = vars(parser.parse_args())

    # if no toml or data file: print help
    if not os.path.isfile(imps["toml"]) or not os.path.isfile(imps["data"]):
        parser.print_help()
        return

    with open(imps["toml"], "rb") as toml_file:
        config = tomli.load(toml_file)
    imps["intfs"] = config["simulation"]["interfaces"]

    if imps["lamres"] == "(intf_1-intf0)/10)":
        imps["lamres"] = (imps["intfs"][1] - imps["intfs"][0]) / 10

    run_analysis(imps)


def infretisinit():
    """To generate initial *toml template and other features."""
    return
