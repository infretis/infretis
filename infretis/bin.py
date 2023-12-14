"""The functions to be used to run infretis via the terminal."""
import argparse
import os

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
        ("-toml", "the toml file for the simulation", "infretis.toml"),
        ("-data", "the infretis data.txt file", "infretis_data.txt"),
        ("-nskip", "number of lines to skip in infretis_data.txt", 100),
        (
            "-lamres",
            "resolution along the orderparameter",
            "(intf1-intf0)/10)",
        ),
        ("-nblock", "minimal number of blocks in the block-error analysis", 5),
        (
            "-fener",
            "calculate free energy. See settings in tools/Free_energy.py",
            False,
        ),
        ("-folder", "output folder", "wham"),
    ]
    # fill defaults
    for arg, info, default in args:
        parser.add_argument(
            arg,
            help=info + f" (default: {default})",
            default=default,
        )

    # get user input
    imps = vars(parser.parse_args())

    # if no toml or data file: print help
    if not os.path.isfile(imps["toml"]) or not os.path.isfile(imps["data"]):
        parser.print_help()
        return

    config = setup_config(imps["toml"])
    imps["intfs"] = config["simulation"]["interfaces"]

    if imps["lamres"] == "(intf1-intf0)/10)":
        imps["lamres"] = (imps["intfs"][1] - imps["intfs"][0]) / 10

    if imps["fener"]:
        imps["trajdir"] = config["simulation"]["load_dir"]

    run_analysis(imps)


def infretisinit():
    """To generate initial *toml template and other features."""
    return
