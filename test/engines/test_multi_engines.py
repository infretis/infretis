"""Test what?."""
import pathlib

import tomli

from infretis.classes.engines.factory import create_engines

HERE = pathlib.Path(__file__).resolve().parent


def get_quantis_engines():
    """Set up a turtlemd engine for the H2 system where
    [N+] potential is Lennard-Jones and [0-] is PairDoubleWell.
    """
    input_path = HERE / "../../examples/turtlemd/H2_quantis/"
    toml_file = input_path / "infretis.toml"
    with open(toml_file, "rb") as rfile:
        config = tomli.load(rfile)
    config["simulation"]["tis_set"]["multi_engine"] = False
    config["simulation"]["tis_set"]["quantis"] = True
    config["simulation"]["tis_set"]["accept_all"] = False
    config["engine0"]["input_path"] = input_path
    config["engine1"]["input_path"] = input_path
    engines = create_engines(config)
    return engines


def test_quantis_potentials():
    """Test that we actually create two different potentials with quantis."""
    engines = get_quantis_engines()
    # We should have 2 engines
    assert len(engines) == 2
    # The two engines should have different potentials
    assert engines[-1].potential[0].desc != engines[0].potential[0].desc
