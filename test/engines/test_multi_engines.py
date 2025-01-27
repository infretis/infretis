"""Test what?."""

import pathlib
import os

import numpy as np

from infretis.classes.engines.factory import create_engines
from infretis.setup import setup_config

HERE = pathlib.Path(__file__).resolve().parent


def get_quantis_engines(tmp_path: pathlib.PosixPath):
    """Set up a turtlemd engine for the H2 system where
    [N+] potential is Lennard-Jones and [0-] is PairDoubleWell.
    """
    input_path = HERE / "../../examples/turtlemd/H2_quantis/"
    toml_file = input_path / "infretis.toml"
    os.chdir(tmp_path)
    config = setup_config(toml_file)
    engines = create_engines(config)
    return engines


def test_quantis_potentials(tmp_path: pathlib.PosixPath):
    """Test that we actually create two different potentials with quantis."""
    engines, _ = get_quantis_engines(tmp_path)
    # We should have 2 engines
    assert len(engines) == 2
    # The two engines should have different potentials
    assert (
        engines["engine0"][0].potential[0].desc
        != engines["engine"][0].potential[0].desc
    )


def test_potential_curves(tmp_path: pathlib.PosixPath):
    engines, _ = get_quantis_engines(tmp_path)
    e0 = engines["engine0"][0]
    e1 = engines["engine"][0]
    x = np.linspace(0.3, 1.3, 50)
    e1.system.particles.pos *= 0
    en0 = []
    en1 = []
    for i, xi in enumerate(x):
        e1.system.particles.pos[0, 0] = xi
        en0.append(e0.potential[0].potential(e1.system))
        en1.append(e1.potential[0].potential(e1.system))
    assert not np.allclose(en0, en1)
