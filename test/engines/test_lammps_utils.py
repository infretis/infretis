"""Test some utils from the lammps engine."""

import pathlib

import numpy as np

from infretis.classes.engines.lammps import (
get_atom_masses,
read_energies,
)

HERE = pathlib.Path(__file__).resolve().parent

ATOM_MASSES = np.array([12.011, 35.45, 1.008, 1.008, 1.008, 15.9994, 1.008, 1.008, 35.45])
ENERGIES = np.array([840.63779, 849.66006, 869.09075, 818.6732, 835.96235, 848.68126, 811.74666, 769.69304, 780.0135, 760.957,  748.75368, 734.62429])

def test_read_masses():
    masses = get_atom_masses(
            HERE / "data/lammps_files/unsorted_atom_style_full.data",
            atom_style = "full"
            )
    assert np.all(masses.flatten() == ATOM_MASSES)

def test_read_energies():
    energies = read_energies(HERE / "data/lammps_files/log.lammps")
    assert np.all(energies["KinEng"] == ENERGIES)

